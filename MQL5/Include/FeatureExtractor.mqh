//+------------------------------------------------------------------+
//|                                             FeatureExtractor.mqh |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| FEATURE EXTRACTION ARCHITECTURE:                                 |
//|                                                                  |
//| 1. Modularity & Zero Train-Serving Skew:                         |
//|    This single header file is consumed by BOTH:                  |
//|    - DMatrix-EA.mq5 (Historical Dataset Collector)               |
//|    - LiveONNX-EA.mq5 (Live Microsecond Inference Engine)         |
//|    Guaranteeing bit-for-bit parity of input feature tensors.     |
//|                                                                  |
//| 2. Sequential Horizon Lookback Shift:                            |
//|    For each active feature F_i and lookback lag h:               |
//|    Row Vector = [ F_i(t), F_i(t-1), F_i(t-2), ..., F_i(t-H) ]   |
//|    Total Vector Length = BaseFeaturesCount * (Lookback + 1)      |
//|                                                                  |
//| 3. Feature Categories (13 Toggleable Groups):                    |
//|    A. Technical Indicators: ADX, ATR, Bands, MACD, Fast/Slow MA, |
//|       RSI, Stochastic. (Handles & copy buffers).                 |
//|    B. Price Action / Candlesticks: Type (0/1/2), Body, Shadows   |
//|    C. Temporal Context: Weekday (0-4), Quarter Day (0-3)         |
//|    D. Market Microstructure: Open Session (0-7), Spread (Points) |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| SFeatureConfig: Master Configuration struct for all feature flags|
//+------------------------------------------------------------------+
#include "GarchEngine.mqh"

struct SFeatureConfig
{
   //--- Indicator Toggles (13 Toggleable Groups)
   bool useADX;            // iADX (Main, +DI, -DI)
   bool useATR;            // iATR (Normalized volatility)
   bool useBands;          // Bollinger Bands (Diff to mid, Bandwidth)
   bool useMACD;           // iMACD (Main, Signal)
   bool useFastMA;         // Fast Moving Average distance
   bool useSlowMA;         // Slow Moving Average distance
   bool useRSI;            // Relative Strength Index
   bool useStochastic;     // Stochastic Oscillator (%K, %D)
   
   //--- Candlestick / Price Action Toggle
   bool useCandlestick;    // Candlestick Geometry & Type
   
   //--- Temporal & Microstructure Toggles
   bool useTimestampWeek;  // Day of week (0f=Mon ... 4f=Fri)
   bool useTimestampDay;   // Quarter of day (0f=00-06h, 1f=06-12h, 2f=12-18h, 3f=18-24h)
   bool useOpenMarkets;    // Active Forex Session Code (0f=Syd ... 7f=NY+Syd)
   bool useSpread;         // Spread in points
   
   //--- GARCH Volatility Feature Toggle & Settings
   bool   useGarchFeatures; // Include GARCH(1,1) features (omega, alpha, beta, sigma_cond, sigma_agg)
   int    garchHorizon;     // Forecast horizon for GARCH
   double garchAlpha;       // ARCH shock parameter
   double garchBeta;        // GARCH persistence parameter
   
   //--- Feature Lookback & Historical Sizing
   int  featureLookback;   // Feature Lookback Lags (t to t-N)
   int  priceSize;         // Sample size for historical GARCH modeling
   
   //--- Indicator Parameters
   // iADX
   int                adxPeriod;
   // iATR
   int                atrPeriod;
   // iBands
   int                bandsPeriod;
   int                bandsShift;
   double             bandsDeviation;
   ENUM_APPLIED_PRICE bandsAppliedPrice;
   // iMACD
   int                macdFastPeriod;
   int                macdSlowPeriod;
   int                macdSignalPeriod;
   ENUM_APPLIED_PRICE macdAppliedPrice;
   // Fast iMA
   int                fastMAPeriod;
   int                fastMAShift;
   ENUM_MA_METHOD     fastMAMethod;
   ENUM_APPLIED_PRICE fastMAAppliedPrice;
   // Slow iMA
   int                slowMAPeriod;
   int                slowMAShift;
   ENUM_MA_METHOD     slowMAMethod;
   ENUM_APPLIED_PRICE slowMAAppliedPrice;
   // iRSI
   int                rsiPeriod;
   ENUM_APPLIED_PRICE rsiAppliedPrice;
   // iStochastic
   int                stochKPeriod;
   int                stochDPeriod;
   int                stochSlowing;
   ENUM_MA_METHOD     stochMethod;
   ENUM_STO_PRICE     stochPriceField;
   
   //--- Default Constructor: Standard Baseline Parameters
   void SetDefaults()
   {
      useADX            = true;
      useATR            = true;
      useBands          = true;
      useMACD           = true;
      useFastMA         = true;
      useSlowMA         = true;
      useRSI            = true;
      useStochastic     = true;
      
      useCandlestick    = true;
      
      useTimestampWeek  = true;
      useTimestampDay   = true;
      useOpenMarkets    = true;
      useSpread         = true;
      
      useGarchFeatures  = true;
      garchHorizon      = 8;
      garchAlpha        = 0.05;
      garchBeta         = 0.92;
      
      featureLookback   = 4;
      priceSize         = 500;
      
      adxPeriod         = 14;
      atrPeriod         = 14;
      bandsPeriod       = 20;
      bandsShift        = 0;
      bandsDeviation    = 2.0;
      bandsAppliedPrice = PRICE_CLOSE;
      macdFastPeriod    = 12;
      macdSlowPeriod    = 26;
      macdSignalPeriod  = 9;
      macdAppliedPrice  = PRICE_CLOSE;
      fastMAPeriod      = 20;
      fastMAShift       = 0;
      fastMAMethod      = MODE_EMA;
      fastMAAppliedPrice= PRICE_CLOSE;
      slowMAPeriod      = 50;
      slowMAShift       = 0;
      slowMAMethod      = MODE_EMA;
      slowMAAppliedPrice= PRICE_CLOSE;
      rsiPeriod         = 14;
      rsiAppliedPrice   = PRICE_CLOSE;
      stochKPeriod      = 8;
      stochDPeriod      = 3;
      stochSlowing      = 3;
      stochMethod       = MODE_SMA;
      stochPriceField   = STO_LOWHIGH;
   }
};

//+------------------------------------------------------------------+
//| CFeatureExtractor: Multi-Indicator Extraction & Lookback Engine  |
//+------------------------------------------------------------------+
class CFeatureExtractor
{
private:
   string          m_symbol;
   ENUM_TIMEFRAMES m_period;
   SFeatureConfig  m_config;
   CGarchEngine    m_garch;
   
   // MT5 Built-in Indicator Handles
   int             m_hADX;
   int             m_hATR;
   int             m_hBands;
   int             m_hMACD;
   int             m_hFastMA;
   int             m_hSlowMA;
   int             m_hRSI;
   int             m_hStoch;
   
   int             m_baseFeatureCount; // Number of atomic features per single bar
   int             m_totalVectorSize;  // m_baseFeatureCount * (featureLookback + 1)
   string          m_baseFeatureNames[];
   
public:
   //+---------------------------------------------------------------+
   //| Constructor                                                   |
   //+---------------------------------------------------------------+
   CFeatureExtractor()
   {
      m_hADX             = INVALID_HANDLE;
      m_hATR             = INVALID_HANDLE;
      m_hBands           = INVALID_HANDLE;
      m_hMACD            = INVALID_HANDLE;
      m_hFastMA          = INVALID_HANDLE;
      m_hSlowMA          = INVALID_HANDLE;
      m_hRSI             = INVALID_HANDLE;
      m_hStoch           = INVALID_HANDLE;
      m_baseFeatureCount = 0;
      m_totalVectorSize  = 0;
   }
   
   //+---------------------------------------------------------------+
   //| Destructor: Safely frees all allocated indicator resources    |
   //+---------------------------------------------------------------+
   ~CFeatureExtractor()
   {
      ReleaseHandles();
   }
   
   //+---------------------------------------------------------------+
   //| Init: Configures indicators, validates horizon, builds schema |
   //+---------------------------------------------------------------+
   bool Init(const string symbol, ENUM_TIMEFRAMES period, const SFeatureConfig &config)
   {
      m_symbol = symbol;
      m_period = period;
      m_config = config;
      
      ReleaseHandles();
      
      // Initialize GARCH Engine if enabled
      if(m_config.useGarchFeatures)
      {
         m_garch.SetParameters(m_config.priceSize, m_config.garchHorizon, m_config.garchAlpha, m_config.garchBeta);
      }
      
      // Initialize Indicator Handles according to active feature toggles
      if(m_config.useADX)
      {
         m_hADX = iADX(m_symbol, m_period, m_config.adxPeriod);
         if(m_hADX == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iADX handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useATR)
      {
         m_hATR = iATR(m_symbol, m_period, m_config.atrPeriod);
         if(m_hATR == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iATR handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useBands)
      {
         m_hBands = iBands(m_symbol, m_period, m_config.bandsPeriod, m_config.bandsShift, m_config.bandsDeviation, m_config.bandsAppliedPrice);
         if(m_hBands == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iBands handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useMACD)
      {
         m_hMACD = iMACD(m_symbol, m_period, m_config.macdFastPeriod, m_config.macdSlowPeriod, m_config.macdSignalPeriod, m_config.macdAppliedPrice);
         if(m_hMACD == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iMACD handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useFastMA)
      {
         m_hFastMA = iMA(m_symbol, m_period, m_config.fastMAPeriod, m_config.fastMAShift, m_config.fastMAMethod, m_config.fastMAAppliedPrice);
         if(m_hFastMA == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating Fast iMA handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useSlowMA)
      {
         m_hSlowMA = iMA(m_symbol, m_period, m_config.slowMAPeriod, m_config.slowMAShift, m_config.slowMAMethod, m_config.slowMAAppliedPrice);
         if(m_hSlowMA == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating Slow iMA handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useRSI)
      {
         m_hRSI = iRSI(m_symbol, m_period, m_config.rsiPeriod, m_config.rsiAppliedPrice);
         if(m_hRSI == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iRSI handle: %d", GetLastError());
            return false;
         }
      }
      
      if(m_config.useStochastic)
      {
         m_hStoch = iStochastic(m_symbol, m_period, m_config.stochKPeriod, m_config.stochDPeriod, m_config.stochSlowing, m_config.stochMethod, m_config.stochPriceField);
         if(m_hStoch == INVALID_HANDLE)
         {
            PrintFormat("[FeatureExtractor] Error creating iStochastic handle: %d", GetLastError());
            return false;
         }
      }
      
      // Build Atomic Base Feature Schema and calculate total tensor dimension
      BuildFeatureSchema();
      
      PrintFormat("[FeatureExtractor] Initialized. Base Features: %d, Feature Lookback: %d, Total Dimensions: %d",
                  m_baseFeatureCount, m_config.featureLookback, m_totalVectorSize);
      return true;
   }
   
   //+---------------------------------------------------------------+
   //| ReleaseHandles: Releases indicator handles back to terminal   |
   //+---------------------------------------------------------------+
   void ReleaseHandles()
   {
      if(m_hADX != INVALID_HANDLE)    { IndicatorRelease(m_hADX);    m_hADX = INVALID_HANDLE; }
      if(m_hATR != INVALID_HANDLE)    { IndicatorRelease(m_hATR);    m_hATR = INVALID_HANDLE; }
      if(m_hBands != INVALID_HANDLE)  { IndicatorRelease(m_hBands);  m_hBands = INVALID_HANDLE; }
      if(m_hMACD != INVALID_HANDLE)   { IndicatorRelease(m_hMACD);   m_hMACD = INVALID_HANDLE; }
      if(m_hFastMA != INVALID_HANDLE) { IndicatorRelease(m_hFastMA); m_hFastMA = INVALID_HANDLE; }
      if(m_hSlowMA != INVALID_HANDLE) { IndicatorRelease(m_hSlowMA); m_hSlowMA = INVALID_HANDLE; }
      if(m_hRSI != INVALID_HANDLE)    { IndicatorRelease(m_hRSI);    m_hRSI = INVALID_HANDLE; }
      if(m_hStoch != INVALID_HANDLE)  { IndicatorRelease(m_hStoch);  m_hStoch = INVALID_HANDLE; }
   }
   
   //+---------------------------------------------------------------+
   //| Getters                                                       |
   //+---------------------------------------------------------------+
   int  GetTotalVectorSize() const { return m_totalVectorSize; }
   int  GetBaseFeatureCount() const { return m_baseFeatureCount; }
   int  GetFeatureLookback() const { return m_config.featureLookback; }
   void GetConfig(SFeatureConfig &outConfig) const { outConfig = m_config; }
   
   //+---------------------------------------------------------------+
   //| ExtractFlattenedVector: Extracts and flattens all active      |
   //| features across lookback [t, t-1, ... t-H] into a 1D vectorf. |
   //+---------------------------------------------------------------+
   bool ExtractFlattenedVector(int baseShift, vectorf &outVector)
   {
      if(m_totalVectorSize <= 0) return false;
      
      outVector.Resize(m_totalVectorSize);
      
      // Request price rates across full lookback horizon
      int barsNeeded = baseShift + m_config.featureLookback + 10;
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      int copiedRates = CopyRates(m_symbol, m_period, 0, barsNeeded, rates);
      if(copiedRates < barsNeeded)
      {
         PrintFormat("[FeatureExtractor] [WARMUP] Insufficient historical rates. Copied: %d, Needed: %d. Waiting for history buffer.", copiedRates, barsNeeded);
         return false;
      }
      
      double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      if(point <= 0.0) point = 0.00001;
      
      int vecIdx = 0;
      
      // Sequential Horizon Flattening: Step h=0 (current base t) to h=featureLookback (lag t-N)
      for(int h = 0; h <= m_config.featureLookback; h++)
      {
         int currentShift = baseShift + h;
         
         // 1. Technical Indicators
         if(m_config.useADX)
         {
            double bufMain[], bufPDI[], bufNDI[];
            CopyBuffer(m_hADX, 0, currentShift, 1, bufMain);
            CopyBuffer(m_hADX, 1, currentShift, 1, bufPDI);
            CopyBuffer(m_hADX, 2, currentShift, 1, bufNDI);
            
            outVector[vecIdx++] = (bufMain.Size() > 0) ? (float)bufMain[0] : 0.0f;
            outVector[vecIdx++] = (bufPDI.Size() > 0)  ? (float)bufPDI[0]  : 0.0f;
            outVector[vecIdx++] = (bufNDI.Size() > 0)  ? (float)bufNDI[0]  : 0.0f;
         }
         
         if(m_config.useATR)
         {
            double bufATR[];
            CopyBuffer(m_hATR, 0, currentShift, 1, bufATR);
            outVector[vecIdx++] = (bufATR.Size() > 0) ? (float)(bufATR[0] / point) : 0.0f;
         }
         
         if(m_config.useBands)
         {
            double bufBase[], bufUpper[], bufLower[];
            CopyBuffer(m_hBands, 0, currentShift, 1, bufBase);
            CopyBuffer(m_hBands, 1, currentShift, 1, bufUpper);
            CopyBuffer(m_hBands, 2, currentShift, 1, bufLower);
            
            double closeP = rates[currentShift].close;
            double baseP  = (bufBase.Size() > 0)  ? bufBase[0]  : closeP;
            double upP    = (bufUpper.Size() > 0) ? bufUpper[0] : closeP;
            double lowP   = (bufLower.Size() > 0) ? bufLower[0] : closeP;
            
            outVector[vecIdx++] = (float)((closeP - baseP) / point);
            outVector[vecIdx++] = (float)((upP - lowP) / point);
         }
         
         if(m_config.useMACD)
         {
            double bufMain[], bufSig[];
            CopyBuffer(m_hMACD, 0, currentShift, 1, bufMain);
            CopyBuffer(m_hMACD, 1, currentShift, 1, bufSig);
            
            outVector[vecIdx++] = (bufMain.Size() > 0) ? (float)(bufMain[0] / point) : 0.0f;
            outVector[vecIdx++] = (bufSig.Size() > 0)  ? (float)(bufSig[0] / point)  : 0.0f;
         }
         
         if(m_config.useFastMA)
         {
            double bufFast[];
            CopyBuffer(m_hFastMA, 0, currentShift, 1, bufFast);
            double maVal = (bufFast.Size() > 0) ? bufFast[0] : rates[currentShift].close;
            outVector[vecIdx++] = (float)((rates[currentShift].close - maVal) / point);
         }
         
         if(m_config.useSlowMA)
         {
            double bufSlow[];
            CopyBuffer(m_hSlowMA, 0, currentShift, 1, bufSlow);
            double maVal = (bufSlow.Size() > 0) ? bufSlow[0] : rates[currentShift].close;
            outVector[vecIdx++] = (float)((rates[currentShift].close - maVal) / point);
         }
         
         if(m_config.useRSI)
         {
            double bufRSI[];
            CopyBuffer(m_hRSI, 0, currentShift, 1, bufRSI);
            outVector[vecIdx++] = (bufRSI.Size() > 0) ? (float)bufRSI[0] : 50.0f;
         }
         
         if(m_config.useStochastic)
         {
            double bufK[], bufD[];
            CopyBuffer(m_hStoch, 0, currentShift, 1, bufK);
            CopyBuffer(m_hStoch, 1, currentShift, 1, bufD);
            
            outVector[vecIdx++] = (bufK.Size() > 0) ? (float)bufK[0] : 50.0f;
            outVector[vecIdx++] = (bufD.Size() > 0) ? (float)bufD[0] : 50.0f;
         }
         
         // 2. Candlestick Geometry & Price Action
         if(m_config.useCandlestick)
         {
            double openP  = rates[currentShift].open;
            double highP  = rates[currentShift].high;
            double lowP   = rates[currentShift].low;
            double closeP = rates[currentShift].close;
            
            // Type: 0f = Neutral (doji), 1f = Bullish (Close > Open), 2f = Bearish (Close < Open)
            float candleType = 0.0f;
            if(closeP > openP)      candleType = 1.0f;
            else if(closeP < openP) candleType = 2.0f;
            
            float bodySize    = (float)(MathAbs(closeP - openP) / point);
            float upperShadow = (float)(MathMax(0.0, (highP - MathMax(openP, closeP))) / point);
            float lowerShadow = (float)(MathMax(0.0, (MathMin(openP, closeP) - lowP)) / point);
            
            outVector[vecIdx++] = candleType;
            outVector[vecIdx++] = bodySize;
            outVector[vecIdx++] = upperShadow;
            outVector[vecIdx++] = lowerShadow;
         }
         
         // 3. Temporal Context & Market Microstructure
         datetime barTime = rates[currentShift].time;
         MqlDateTime dt;
         TimeToStruct(barTime, dt);
         
         if(m_config.useTimestampWeek)
         {
            // Timestamp Week: (0f=Mon ... 4f=Fri)
            float weekDay = (float)(dt.day_of_week - 1);
            if(weekDay < 0.0f) weekDay = 4.0f;
            if(weekDay > 4.0f) weekDay = 4.0f;
            outVector[vecIdx++] = weekDay;
         }
         
         if(m_config.useTimestampDay)
         {
            // Quarter of Day: (0f=00-06h, 1f=06-12h, 2f=12-18h, 3f=18-24h)
            float dayQuarter = (float)(dt.hour / 6);
            if(dayQuarter > 3.0f) dayQuarter = 3.0f;
            outVector[vecIdx++] = dayQuarter;
         }
         
         if(m_config.useOpenMarkets)
         {
            // Active Market Sessions Code: (0f=Sydney ... 7f=NY+Syd)
            float marketCode = GetMarketSessionCode(dt.hour);
            outVector[vecIdx++] = marketCode;
         }
         
         if(m_config.useSpread)
         {
            // Spread in points
            long sp = rates[currentShift].spread;
            if(sp <= 0) sp = SymbolInfoInteger(m_symbol, SYMBOL_SPREAD);
            outVector[vecIdx++] = (float)sp;
         }
         
         // 4. GARCH(1,1) Volatility Dynamics
         if(m_config.useGarchFeatures)
         {
            double omega = 0.0, volRatio = 0.0, volTrend = 0.0, sigmaCond = 0.0, sigmaAgg = 0.0;
            if(!m_garch.ComputeGarchMetrics(m_symbol, m_period, currentShift, omega, volRatio, volTrend, sigmaCond, sigmaAgg))
            {
               return false;
            }
            outVector[vecIdx++] = (float)omega;
            outVector[vecIdx++] = (float)volRatio;
            outVector[vecIdx++] = (float)volTrend;
            outVector[vecIdx++] = (float)sigmaCond;
            outVector[vecIdx++] = (float)sigmaAgg;
         }
      }
      
      return (vecIdx == m_totalVectorSize);
   }
   
   //+---------------------------------------------------------------+
   //| GetFeatureColumnNames: Returns full list of flattened names   |
   //+---------------------------------------------------------------+
   void GetFeatureColumnNames(string &outNames[]) const
   {
      ArrayResize(outNames, m_totalVectorSize);
      int idx = 0;
      for(int h = 0; h <= m_config.featureLookback; h++)
      {
         string suffix = (h == 0) ? "_t" : "_t_minus_" + IntegerToString(h);
         
         for(int b = 0; b < m_baseFeatureCount; b++)
         {
            outNames[idx++] = m_baseFeatureNames[b] + suffix;
         }
      }
   }
   
   //+---------------------------------------------------------------+
   //| GetCSVHeader: Returns standard comma-separated CSV header     |
   //+---------------------------------------------------------------+
   string GetCSVHeader() const
   {
      string names[];
      GetFeatureColumnNames(names);
      string header = "";
      int total = ArraySize(names);
      for(int i = 0; i < total; i++)
      {
         if(i > 0) header += ",";
         header += names[i];
      }
      header += ",label";
      return header;
   }
   
private:
   //+---------------------------------------------------------------+
   //| BuildFeatureSchema: Constructs the ordered list of base names |
   //+---------------------------------------------------------------+
   void BuildFeatureSchema()
   {
      m_baseFeatureCount = 0;
      ArrayFree(m_baseFeatureNames);
      
      if(m_config.useADX)
      {
         AddBaseFeature("adx_main");
         AddBaseFeature("adx_pdi");
         AddBaseFeature("adx_ndi");
      }
      if(m_config.useATR)
      {
         AddBaseFeature("atr");
      }
      if(m_config.useBands)
      {
         AddBaseFeature("bands_diff_mid");
         AddBaseFeature("bands_bandwidth");
      }
      if(m_config.useMACD)
      {
         AddBaseFeature("macd_main");
         AddBaseFeature("macd_signal");
      }
      if(m_config.useFastMA)
      {
         AddBaseFeature("ma_fast_diff");
      }
      if(m_config.useSlowMA)
      {
         AddBaseFeature("ma_slow_diff");
      }
      if(m_config.useRSI)
      {
         AddBaseFeature("rsi");
      }
      if(m_config.useStochastic)
      {
         AddBaseFeature("stoch_k");
         AddBaseFeature("stoch_d");
      }
      if(m_config.useCandlestick)
      {
         AddBaseFeature("candle_type");
         AddBaseFeature("candle_body");
         AddBaseFeature("candle_upper_shadow");
         AddBaseFeature("candle_lower_shadow");
      }
      if(m_config.useTimestampWeek)
      {
         AddBaseFeature("timestamp_week");
      }
      if(m_config.useTimestampDay)
      {
         AddBaseFeature("timestamp_day");
      }
      if(m_config.useOpenMarkets)
      {
         AddBaseFeature("open_markets");
      }
      if(m_config.useSpread)
      {
         AddBaseFeature("spread");
      }
      if(m_config.useGarchFeatures)
      {
         AddBaseFeature("garch_omega");
         AddBaseFeature("garch_vol_ratio");
         AddBaseFeature("garch_vol_trend");
         AddBaseFeature("garch_sigma_cond");
         AddBaseFeature("garch_sigma_agg");
      }
      
      m_totalVectorSize = m_baseFeatureCount * (m_config.featureLookback + 1);
   }
   
   void AddBaseFeature(const string name)
   {
      ArrayResize(m_baseFeatureNames, m_baseFeatureCount + 1);
      m_baseFeatureNames[m_baseFeatureCount] = name;
      m_baseFeatureCount++;
   }

public:
   //+---------------------------------------------------------------+
   //| GetMarketSessionCode: Maps MT5 Server Time (EET/EEST) hour to |
   //| active trading session regimes.                                |
   //| (0f=Sydney, 1f=Syd+Tky, 2f=Tokyo, 3f=Tky+Lon, 4f=London,      |
   //|  5f=Lon+NY, 6f=NY, 7f=NY+Syd)                                 |
   //+---------------------------------------------------------------+
   float GetMarketSessionCode(int hour) const
   {
      switch(hour)
      {
         case 0:  case 1:
            return 0.0f; // Sydney
         case 2:  case 3:  case 4:  case 5:  case 6:  case 7:  case 8:
            return 1.0f; // Syd+Tky
         case 9:
            return 2.0f; // Tokyo
         case 10:
            return 3.0f; // Tky+Lon
         case 11: case 12: case 13: case 14:
            return 4.0f; // London
         case 15: case 16: case 17: case 18:
            return 5.0f; // Lon+NY
         case 19: case 20: case 21: case 22:
            return 6.0f; // NY
         case 23:
            return 7.0f; // NY+Syd
         default:
            return 0.0f;
      }
   }
};

