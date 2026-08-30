//+------------------------------------------------------------------+
//|                                                  GarchEngine.mqh |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| QUANTITATIVE FOUNDATION & MATHEMATICAL SPECIFICATION:            |
//|                                                                  |
//| GARCH(1,1) - Generalized Autoregressive Conditional              |
//| Heteroskedasticity (Bollerslev, 1986).                           |
//|                                                                  |
//| 1. Log Returns Formulation:                                      |
//|    r_t = ln(Close_t / Close_{t-1})                               |
//|    Mean Return: mu = (1/N) * sum_{i=1}^N r_i                     |
//|    Sample Variance: s^2 = (1/(N-1)) * sum_{i=1}^N (r_i - mu)^2   |
//|                                                                  |
//| 2. Conditional Variance Recurrence:                              |
//|    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2  |
//|    where e_t = r_t - mu (mean-adjusted shock/innovation)         |
//|    Stationarity Condition: alpha + beta < 1.0 (Persistence)      |
//|    Long-run Unconditional Variance: V_L = omega / (1 - alpha - beta)
//|    => omega = s^2 * (1 - (alpha + beta))                         |
//|                                                                  |
//| 3. Multi-Step Horizon Analytical Variance Forecast:              |
//|    For step h = 1, 2, ..., H:                                    |
//|    E[sigma_{t+h}^2] = V_L + (alpha + beta)^h * (sigma_t^2 - V_L) |
//|    Aggregated Cumulative Variance across horizon H:              |
//|    sigma_agg^2 = sum_{h=1}^H E[sigma_{t+h}^2]                    |
//|    sigma_agg   = sqrt(sigma_agg^2)                               |
//|                                                                  |
//| 4. Dynamic Price Risk & Stop-Level Mapping:                      |
//|    PriceRisk   = CurrentPrice * sigma_agg                        |
//|    RiskPoints  = PriceRisk / _Point                              |
//|    TP_Points   = kTP * RiskPoints                                |
//|    SL_Points   = kSL * RiskPoints                                |
//|                                                                  |
//| 5. Econometric Machine Learning Features (Zero Static Constants):|
//|    - omega: Long-run baseline variance scale                     |
//|    - vol_ratio: sigma_cond / sqrt(s^2) (expansion > 1 / compr < 1)|
//|    - vol_trend: sigma_agg / (sqrt(H) * sigma_cond) (term slope)  |
//|    - sigma_cond: Instantaneous conditional return volatility     |
//|    - sigma_agg: Multi-step cumulative horizon standard deviation |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| CGarchEngine: Analytical GARCH(1,1) Multi-Step Volatility Engine |
//+------------------------------------------------------------------+
class CGarchEngine
{
private:
   int      m_priceSize;     // Historical sample size (bars) for return calculation
   int      m_horizon;       // Multi-step forecast horizon (bars)
   double   m_alpha;         // ARCH parameter: weight on lagged market shock (alpha > 0)
   double   m_beta;          // GARCH parameter: persistence of lagged variance (beta > 0)
   
public:
   //+---------------------------------------------------------------+
   //| Constructors                                                  |
   //+---------------------------------------------------------------+
   CGarchEngine()
      : m_priceSize(500),
        m_horizon(8),
        m_alpha(0.05),
        m_beta(0.92)
   {
   }
   
   CGarchEngine(int priceSize, int horizon, double alpha, double beta)
   {
      SetParameters(priceSize, horizon, alpha, beta);
   }
   
   //+---------------------------------------------------------------+
   //| Destructor                                                    |
   //+---------------------------------------------------------------+
   ~CGarchEngine() {}
   
   //+---------------------------------------------------------------+
   //| SetParameters: Configures GARCH parameters with stationarity  |
   //+---------------------------------------------------------------+
   void SetParameters(int priceSize, int horizon, double alpha, double beta)
   {
      m_priceSize = (priceSize >= 30) ? priceSize : 200;
      m_horizon   = (horizon >= 1) ? horizon : 5;
      m_alpha     = (alpha > 0.0 && alpha < 1.0) ? alpha : 0.05;
      m_beta      = (beta > 0.0 && beta < 1.0) ? beta : 0.92;
      
      // Ensure covariance stationarity: alpha + beta < 1.0
      if(m_alpha + m_beta >= 1.0)
      {
         PrintFormat("[GARCH] Warning: Parameter sum (alpha=%.4f + beta=%.4f >= 1.0) violates stationarity. Clamping to 0.05 / 0.92.",
                     m_alpha, m_beta);
         m_alpha = 0.05;
         m_beta  = 0.92;
      }
   }
   
   //+---------------------------------------------------------------+
   //| Getters                                                       |
   //+---------------------------------------------------------------+
   int    GetPriceSize() const { return m_priceSize; }
   int    GetHorizon()   const { return m_horizon; }
   double GetAlpha()     const { return m_alpha; }
   double GetBeta()      const { return m_beta; }
    
   //+---------------------------------------------------------------+
   //| ComputeGarchMetrics: Returns omega, volRatio, volTrend,       |
   //| sigmaCond, and sigmaAgg for a specific bar shift.             |
   //| Uses strictly closed bars to eliminate lookahead bias.        |
   //+---------------------------------------------------------------+
   bool ComputeGarchMetrics(const string symbol, ENUM_TIMEFRAMES period, int barShift,
                            double &outOmega, double &outVolRatio, double &outVolTrend,
                            double &outSigmaCond, double &outSigmaAgg)
   {
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      
      // Request rates history: sample size + safety buffer starting from barShift
      int barsNeeded = m_priceSize + 10;
      int copied = CopyRates(symbol, period, barShift, barsNeeded, rates);
      if(copied < m_priceSize + 2)
      {
         PrintFormat("[GARCH] [WARMUP] Insufficient rates for %s (shift=%d). Copied: %d, Needed: %d. Waiting for history buffer.",
                     symbol, barShift, copied, m_priceSize + 2);
         return false;
      }
      
      // Step 1: Compute log returns in chronological order (oldest to newest)
      // Note: When barShift == 0 (current incomplete bar), indexing idxNewer = N..1
      // and idxOlder = N+1..2 accesses closed bars rates[1] to rates[N+1],
      // strictly excluding forming bar rates[0] to eliminate lookahead bias.
      int N = m_priceSize;
      double returns[];
      ArrayResize(returns, N);
      
      double meanReturn = 0.0;
      for(int i = 0; i < N; i++)
      {
         int idxNewer = N - i;
         int idxOlder = N - i + 1;
         
         double pNewer = rates[idxNewer].close;
         double pOlder = rates[idxOlder].close;
         
         if(pOlder <= 0.0) pOlder = 1.0;
         returns[i] = MathLog(pNewer / pOlder);
         meanReturn += returns[i];
      }
      meanReturn /= (double)N;
      
      // Step 2: Calculate unconditional sample variance (s^2)
      double sampleVar = 0.0;
      for(int i = 0; i < N; i++)
      {
         double diff = returns[i] - meanReturn;
         sampleVar += diff * diff;
      }
      sampleVar /= (double)(N - 1);
      
      if(sampleVar <= 0.0)
         sampleVar = 1e-6; // Lower bound safety clamp
         
      // Step 3: Compute omega from long-run unconditional variance (Variance Targeting)
      double persistence = m_alpha + m_beta; // Guarantees persistence < 1.0
      double omega = sampleVar * (1.0 - persistence);
      if(omega <= 0.0) omega = 1e-8;
      
      // Step 4: Conditional variance recursion across historical sample window
      // sigma_t^2 = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma_{t-1}^2
      double currentSigma2 = sampleVar; // Initialize at sample variance
      for(int i = 0; i < N; i++)
      {
         double shock = returns[i] - meanReturn;
         double shock2 = shock * shock;
         currentSigma2 = omega + m_alpha * shock2 + m_beta * currentSigma2;
      }
      
      // Step 5: Multi-step aggregated variance forecast over horizon H
      // E[sigma_{t+h}^2] = V_L + (alpha + beta)^h * (sigma_t^2 - V_L)
      double longRunVar = omega / (1.0 - persistence);
      double sumForecastVar = 0.0;
      double persistencePower = persistence;
      
      for(int h = 1; h <= m_horizon; h++)
      {
         double forecastStepVar = longRunVar + persistencePower * (currentSigma2 - longRunVar);
         if(forecastStepVar < 1e-8) forecastStepVar = 1e-8;
         sumForecastVar += forecastStepVar;
         persistencePower *= persistence;
      }
      
      // Step 6: Output analytical econometric metrics
      double sigmaCond = MathSqrt(MathMax(currentSigma2, 1e-8));
      double sigmaAgg  = MathSqrt(MathMax(sumForecastVar, 1e-8));
      if(sigmaAgg <= 0.0) sigmaAgg = 1e-4;
      
      double sampleStd = MathSqrt(MathMax(sampleVar, 1e-8));
      double volRatio  = (sampleStd > 0.0) ? (sigmaCond / sampleStd) : 1.0;
      
      double expectedFlatAgg = MathSqrt((double)m_horizon) * sigmaCond;
      double volTrend  = (expectedFlatAgg > 0.0) ? (sigmaAgg / expectedFlatAgg) : 1.0;
      
      outOmega     = omega;
      outVolRatio  = volRatio;
      outVolTrend  = volTrend;
      outSigmaCond = sigmaCond;
      outSigmaAgg  = sigmaAgg;
      
      ArrayFree(rates);
      ArrayFree(returns);
      return true;
   }
    
   //+---------------------------------------------------------------+
   //| CalculateDynamicRisk: Fits GARCH(1,1), forecasts multi-step   |
   //| aggregated volatility, and computes dynamic TP/SL in points.  |
   //| Uses bar 1 (last closed bar) as reference price.              |
   //+---------------------------------------------------------------+
   bool CalculateDynamicRisk(const string symbol, ENUM_TIMEFRAMES period,
                             double kTP, double kSL,
                             double &outTPPoints, double &outSLPoints, double &outSigmaAgg)
   {
      double omega = 0.0, volRatio = 0.0, volTrend = 0.0, sigmaCond = 0.0;
      if(!ComputeGarchMetrics(symbol, period, 0, omega, volRatio, volTrend, sigmaCond, outSigmaAgg))
         return false;
      
      // Copy bar 1 (the bar that just closed at bar open event)
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(symbol, period, 1, 1, rates) < 1)
      {
         ArrayFree(rates);
         return false;
      }
      
      // Step 7: Dynamic price risk calculation in broker points
      double currentPrice = rates[0].close; // Close of bar 1
      if(currentPrice <= 0.0)
      {
         ArrayFree(rates);
         return false;
      }
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(point <= 0.0) point = 0.00001;
      
      double priceRisk = currentPrice * outSigmaAgg;
      double riskPoints = priceRisk / point;
      
      // Broker Stop-Level Compliance
      long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      long spread    = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
      double minStopPoints = (double)MathMax(stopLevel, spread * 2);
      if(minStopPoints < 10.0) minStopPoints = 10.0;
      
      // Dynamic TP & SL Points
      outTPPoints = kTP * riskPoints;
      outSLPoints = kSL * riskPoints;
      
      // Ensure stops exceed broker minimum threshold
      if(outTPPoints < minStopPoints) outTPPoints = minStopPoints;
      if(outSLPoints < minStopPoints) outSLPoints = minStopPoints;
      
      ArrayFree(rates);
      return true;
   }
};
