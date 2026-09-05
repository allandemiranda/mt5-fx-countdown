//+------------------------------------------------------------------+
//|                                                 DMatrix-EA.mq5   |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "MQL5 Machine Learning Dataset Collector & GARCH Dynamic Risk Engine"

//+------------------------------------------------------------------+
//| MODULE RESPONSIBILITY & EXECUTION LIFECYCLE:                     |
//|                                                                  |
//| 1. Strategy Tester Mode: Runs on 'Every Tick' with zero latency. |
//| 2. On New Bar (IsNewBar):                                        |
//|    - Extracts flattened feature vector across lookback [t..t-H]. |
//|    - Computes GARCH(1,1) dynamic Take Profit and Stop Loss.      |
//|    - Simultaneously executes 1 BUY and 1 SELL position.          |
//|    - Maps tickets in RAM to bypass MT5's 31-char comment limit.  |
//| 3. On Trade Transaction (OnTradeTransaction):                    |
//|    - Records TP closures as 1.0f (OPEN).                         |
//|    - Records SL closures as 0.0f (NOT_OPEN).                     |
//| 4. On Deinitialization (OnDeinit):                               |
//|    - Evaluates remaining open positions using unresolved metric. |
//|    - Chronologically sorts all rows oldest-to-newest.            |
//|    - Strips timestamps and exports CSV datasets directly.        |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
#include "..\Include\GarchEngine.mqh"
#include "..\Include\FeatureExtractor.mqh"
#include "..\Include\OrderTracker.mqh"

#ifndef TRADE_RETCODE_OFFQUOTES
#define TRADE_RETCODE_OFFQUOTES 10004
#endif
#ifndef TRADE_RETCODE_INVALID_STOPS
#define TRADE_RETCODE_INVALID_STOPS 10016
#endif
#ifndef TRADE_RETCODE_TRADE_DISABLED
#define TRADE_RETCODE_TRADE_DISABLED 10017
#endif
#ifndef TRADE_RETCODE_MARKET_CLOSED
#define TRADE_RETCODE_MARKET_CLOSED 10018
#endif
#ifndef TRADE_RETCODE_PRICE_OFF
#define TRADE_RETCODE_PRICE_OFF 10021
#endif

//+------------------------------------------------------------------+
//| INPUT PARAMETERS & FEATURE TOGGLES                               |
//+------------------------------------------------------------------+
//--- 1. Indicator Feature Toggles
input group "=== Indicator Feature Toggles ==="
input bool   InpUseADX         = true;        // Include iADX (Main, +DI, -DI)
input bool   InpUseATR         = true;        // Include iATR (Volatility)
input bool   InpUseBands       = true;        // Include iBands (Diff Mid, Bandwidth)
input bool   InpUseMACD        = true;        // Include iMACD (Main, Signal)
input bool   InpUseFastMA      = true;        // Include Fast iMA Diff
input bool   InpUseSlowMA      = true;        // Include Slow iMA Diff
input bool   InpUseRSI         = true;        // Include iRSI (Momentum)
input bool   InpUseStochastic  = true;        // Include iStochastic (K, D)

//--- 2. Candlestick & Price Action Toggles
input group "=== Candlestick Feature Toggles ==="
input bool   InpUseCandlestick = true;        // Include Candlestick (Type, Body, Shadows)

//--- 3. Temporal & Market Microstructure Toggles
input group "=== Temporal & Market Toggles ==="
input bool   InpUseTimestampWeek = true;      // Include Weekday (0f=Mon ... 4f=Fri)
input bool   InpUseTimestampDay  = true;      // Include Quarter Day (0f=00-06h ... 3f=18-24h)
input bool   InpUseOpenMarkets   = true;      // Include Open Markets Session Code (0f-7f)
input bool   InpUseSpread        = true;      // Include Current Spread in Points

//--- 4. Feature Lookback Settings
input group "=== Feature Lookback Settings ==="
input int    InpFeatureLookback = 4;          // Feature Lookback Lags (t, t-1..t-N)

//--- 5. GARCH(1,1) Volatility Forecast Settings
input group "=== GARCH(1,1) Risk Forecast Settings ==="
input bool   InpUseGarchFeatures    = true;       // Include GARCH Features in Dataset
input int    InpGarchHorizon        = 8;          // GARCH Forecast Horizon (Future bars: t+1..t+H)
input int    InpPriceSize           = 500;        // GARCH Historical Sample Size (bars)
input double InpGarchAlpha          = 0.05;       // GARCH Alpha (ARCH shock weight)
input double InpGarchBeta           = 0.92;       // GARCH Beta (GARCH variance persistence)

//--- 6. Momentum Labeling & Barrier Risk Settings (Triple Barrier Method)
input group "=== Momentum Labeling & Barrier Risk Settings ==="
input int    InpLabelHorizonBars      = 12;       // Vertical Barrier: Horizon in bars (Holding period)
input int    InpLabelMinPoints        = 150;      // Upper Barrier: Min favorable points for TP
input int    InpLabelMaxAdversePoints = 150;      // Lower Barrier: Max adverse points for SL
input double InpLotSize               = 0.01;     // Trade Lot Size
input ulong  InpMagicNumber           = 111100;   // EA Magic Number

//--- 7. Daily Schedule & Session Filter Settings (MT5 Server Time)
input group "=== Daily Schedule & Session Filter Settings ==="
input bool   InpTradeMonday        = true;        // Monday: Enable Trading
input string InpMondayStartTime    = "11:00:00";  // Monday: Start Time (HH:MM:SS)
input string InpMondayEndTime      = "18:00:00";  // Monday: End Time (HH:MM:SS, 00:00:00 = 24h)

input bool   InpTradeTuesday       = true;        // Tuesday: Enable Trading
input string InpTuesdayStartTime   = "10:00:00";  // Tuesday: Start Time (HH:MM:SS)
input string InpTuesdayEndTime     = "18:00:00";  // Tuesday: End Time (HH:MM:SS, 00:00:00 = 24h)

input bool   InpTradeWednesday     = true;        // Wednesday: Enable Trading
input string InpWednesdayStartTime = "10:00:00";  // Wednesday: Start Time (HH:MM:SS)
input string InpWednesdayEndTime   = "18:00:00";  // Wednesday: End Time (HH:MM:SS, 00:00:00 = 24h)

input bool   InpTradeThursday      = true;        // Thursday: Enable Trading
input string InpThursdayStartTime  = "10:00:00";  // Thursday: Start Time (HH:MM:SS)
input string InpThursdayEndTime    = "18:00:00";  // Thursday: End Time (HH:MM:SS, 00:00:00 = 24h)

input bool   InpTradeFriday        = true;        // Friday: Enable Trading
input string InpFridayStartTime    = "10:00:00";  // Friday: Start Time (HH:MM:SS)
input string InpFridayEndTime      = "16:00:00";  // Friday: End Time (HH:MM:SS, 00:00:00 = 24h)

//--- 8. Anomaly & Pandemic Blackout Filter (EET/EEST MT5 Server Time)
input group "=== Anomaly / Pandemic Blackout Filter (EET/EEST Server Time) ==="
input bool     InpAvoidPandemicTime = false;                  // Enable Pandemic / Blackout Period Filter
input datetime InpPandemicStartTime = D'2020.01.01 00:00:00'; // Blackout Start Date in EET/EEST (Inclusive)
input datetime InpPandemicEndTime   = D'2021.06.01 00:00:00'; // Blackout End Date in EET/EEST (Exclusive)

//--- 9. Indicator Calculation Periods, Shifts & Methods
input group "=== Indicator Parameters ==="
input int                InpADXPeriod           = 14;              // ADX Period
input int                InpATRPeriod           = 14;              // ATR Period
input int                InpBandsPeriod         = 20;              // Bollinger Bands Period
input int                InpBandsShift          = 0;               // Bollinger Bands Shift
input double             InpBandsDev            = 2.0;             // Bollinger Bands Deviation
input ENUM_APPLIED_PRICE InpBandsAppliedPrice   = PRICE_CLOSE;     // Bollinger Bands Applied Price
input int                InpMACDFastPeriod      = 12;              // MACD Fast EMA Period
input int                InpMACDSlowPeriod      = 26;              // MACD Slow EMA Period
input int                InpMACDSignalPeriod    = 9;               // MACD Signal SMA Period
input ENUM_APPLIED_PRICE InpMACDAppliedPrice    = PRICE_CLOSE;     // MACD Applied Price
input int                InpFastMAPeriod        = 20;              // Fast MA Period
input int                InpFastMAShift         = 0;               // Fast MA Shift
input ENUM_MA_METHOD     InpFastMAMethod        = MODE_EMA;        // Fast MA Method
input ENUM_APPLIED_PRICE InpFastMAAppliedPrice  = PRICE_CLOSE;     // Fast MA Applied Price
input int                InpSlowMAPeriod        = 50;              // Slow MA Period
input int                InpSlowMAShift         = 0;               // Slow MA Shift
input ENUM_MA_METHOD     InpSlowMAMethod        = MODE_EMA;        // Slow MA Method
input ENUM_APPLIED_PRICE InpSlowMAAppliedPrice  = PRICE_CLOSE;     // Slow MA Applied Price
input int                InpRSIPeriod           = 14;              // RSI Period
input ENUM_APPLIED_PRICE InpRSIAppliedPrice     = PRICE_CLOSE;     // RSI Applied Price
input int                InpStochK              = 8;               // Stochastic %K Period
input int                InpStochD              = 3;               // Stochastic %D Period
input int                InpStochSlowing        = 3;               // Stochastic Slowing
input ENUM_MA_METHOD     InpStochMethod         = MODE_SMA;        // Stochastic MA Method
input ENUM_STO_PRICE     InpStochPriceField     = STO_LOWHIGH;     // Stochastic Price Field

//+------------------------------------------------------------------+
//| GLOBAL OBJECTS & STATE VARIABLES                                 |
//+------------------------------------------------------------------+
CTrade            g_trade;
CGarchEngine      g_garch;
CFeatureExtractor g_featureExtractor;
COrderTracker     g_orderTracker;
SFeatureConfig    g_config;
datetime          g_lastBarTime = 0;

//+------------------------------------------------------------------+
//| GetOptimalFillingType: Returns supported execution filling mode  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetOptimalFillingType(const string symbol)
{
   uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| TimeStringToSeconds: Parses "HH:MM:SS" or "HH:MM" into seconds   |
//+------------------------------------------------------------------+
int TimeStringToSeconds(const string timeStr)
{
   string parts[];
   int count = StringSplit(timeStr, ':', parts);
   int h = (count > 0) ? (int)StringToInteger(parts[0]) : 0;
   int m = (count > 1) ? (int)StringToInteger(parts[1]) : 0;
   int s = (count > 2) ? (int)StringToInteger(parts[2]) : 0;
   return (h * 3600 + m * 60 + s);
}

//+------------------------------------------------------------------+
//| IsTradeScheduleAllowed: Checks whether barTime falls into the    |
//| configured daily trading window in MT5 Server Time               |
//+------------------------------------------------------------------+
bool IsTradeScheduleAllowed(const datetime barTime)
{
   MqlDateTime dt;
   TimeToStruct(barTime, dt);
   
   bool dayEnabled = false;
   string startTimeStr = "00:00:00";
   string endTimeStr   = "00:00:00";
   
   switch(dt.day_of_week)
   {
      case 1: // Monday
         dayEnabled   = InpTradeMonday;
         startTimeStr = InpMondayStartTime;
         endTimeStr   = InpMondayEndTime;
         break;
      case 2: // Tuesday
         dayEnabled   = InpTradeTuesday;
         startTimeStr = InpTuesdayStartTime;
         endTimeStr   = InpTuesdayEndTime;
         break;
      case 3: // Wednesday
         dayEnabled   = InpTradeWednesday;
         startTimeStr = InpWednesdayStartTime;
         endTimeStr   = InpWednesdayEndTime;
         break;
      case 4: // Thursday
         dayEnabled   = InpTradeThursday;
         startTimeStr = InpThursdayStartTime;
         endTimeStr   = InpThursdayEndTime;
         break;
      case 5: // Friday
         dayEnabled   = InpTradeFriday;
         startTimeStr = InpFridayStartTime;
         endTimeStr   = InpFridayEndTime;
         break;
      default: // Saturday
         return false;
   }
   
   if(!dayEnabled)
      return false;
      
   // Daily bars and higher open at 00:00:00 MT5 Server Time.
   // If the day is enabled, daily bars are allowed.
   if(_Period >= PERIOD_D1)
      return true;
      
   int startSec = TimeStringToSeconds(startTimeStr);
   int endSec   = TimeStringToSeconds(endTimeStr);
   
   // FIN-02: Resolve midnight ambiguity (00:00:00)
   // 1) 00:00:00 to 00:00:00 means full 24h trading allowed
   if(startSec == 0 && endSec == 0)
      return true;
      
   // 2) startSec > 0 and endSec == 0 means trade from startSec until midnight / end of day (86400 seconds)
   if(startSec > 0 && endSec == 0)
      endSec = 86400;
      
   int barSec = dt.hour * 3600 + dt.min * 60 + dt.sec;
   
   if(endSec > startSec)
   {
      // Standard daytime window [startSec, endSec) - End Time not included
      return (barSec >= startSec && barSec < endSec);
   }
   else
   {
      // Overnight window (e.g. 22:00:00 to 04:00:00)
      return (barSec >= startSec || barSec < endSec);
   }
}

//+------------------------------------------------------------------+
//| IsNewBar: Detects when a new bar has opened                      |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   if(currentBarTime != g_lastBarTime)
   {
      g_lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("[DMatrix-EA] Initializing Data Collector EA...");
   
   // 1. Configure Feature Flags & Indicator Parameters
   g_config.useADX            = InpUseADX;
   g_config.useATR            = InpUseATR;
   g_config.useBands          = InpUseBands;
   g_config.useMACD           = InpUseMACD;
   g_config.useFastMA         = InpUseFastMA;
   g_config.useSlowMA         = InpUseSlowMA;
   g_config.useRSI            = InpUseRSI;
   g_config.useStochastic     = InpUseStochastic;
   g_config.useCandlestick    = InpUseCandlestick;
   g_config.useTimestampWeek  = InpUseTimestampWeek;
   g_config.useTimestampDay   = InpUseTimestampDay;
   g_config.useOpenMarkets    = InpUseOpenMarkets;
   g_config.useSpread         = InpUseSpread;
   g_config.useGarchFeatures  = InpUseGarchFeatures;
   g_config.garchHorizon      = InpGarchHorizon;
   g_config.garchAlpha        = InpGarchAlpha;
   g_config.garchBeta         = InpGarchBeta;
   
   g_config.featureLookback   = InpFeatureLookback;
   g_config.priceSize         = InpPriceSize;
   
   g_config.adxPeriod         = InpADXPeriod;
   g_config.atrPeriod         = InpATRPeriod;
   g_config.bandsPeriod       = InpBandsPeriod;
   g_config.bandsShift        = InpBandsShift;
   g_config.bandsDeviation    = InpBandsDev;
   g_config.bandsAppliedPrice = InpBandsAppliedPrice;
   g_config.macdFastPeriod    = InpMACDFastPeriod;
   g_config.macdSlowPeriod    = InpMACDSlowPeriod;
   g_config.macdSignalPeriod  = InpMACDSignalPeriod;
   g_config.macdAppliedPrice  = InpMACDAppliedPrice;
   g_config.fastMAPeriod      = InpFastMAPeriod;
   g_config.fastMAShift       = InpFastMAShift;
   g_config.fastMAMethod      = InpFastMAMethod;
   g_config.fastMAAppliedPrice= InpFastMAAppliedPrice;
   g_config.slowMAPeriod      = InpSlowMAPeriod;
   g_config.slowMAShift       = InpSlowMAShift;
   g_config.slowMAMethod      = InpSlowMAMethod;
   g_config.slowMAAppliedPrice= InpSlowMAAppliedPrice;
   g_config.rsiPeriod         = InpRSIPeriod;
   g_config.rsiAppliedPrice   = InpRSIAppliedPrice;
   g_config.stochKPeriod      = InpStochK;
   g_config.stochDPeriod      = InpStochD;
   g_config.stochSlowing      = InpStochSlowing;
   g_config.stochMethod       = InpStochMethod;
   g_config.stochPriceField   = InpStochPriceField;
   
   // 2. Initialize GARCH Econometric Engine
   g_garch.SetParameters(InpPriceSize, InpGarchHorizon, InpGarchAlpha, InpGarchBeta);
   
   // 3. Initialize Modular Feature Extractor
   if(!g_featureExtractor.Init(_Symbol, _Period, g_config))
   {
      Print("[DMatrix-EA] Error initializing Feature Extractor!");
      return INIT_FAILED;
   }
   
   // 4. Initialize In-Memory Order Tracker
   g_orderTracker.Init(_Symbol, _Period);
   
   // 5. Configure Trade Execution Handler with Adaptive Filling Mode
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(GetOptimalFillingType(_Symbol));
   
   PrintFormat("[DMatrix-EA] Initialized successfully on %s, %s. Total Feature Dimensions: %d",
               _Symbol, EnumToString(_Period), g_featureExtractor.GetTotalVectorSize());
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   PrintFormat("[DMatrix-EA] Deinitializing (Reason: %d). Processing unresolved positions and exporting datasets...", reason);
   
   // Apply deinit rules, sort chronologically, and export CSVs
   g_orderTracker.ExportDatasets(g_featureExtractor, g_config);
   
   g_featureExtractor.ReleaseHandles();
   Print("[DMatrix-EA] Deinitialization complete.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Execute only once per new bar open
   if(!IsNewBar()) return;
   
   // 1. Enforce Triple Barrier Vertical Timeout on existing active positions
   g_orderTracker.CheckTimeouts(InpLabelHorizonBars, g_trade);
   
   // Base timestamp of the new bar
   datetime baseTimestamp = iTime(_Symbol, _Period, 0);
   
   // 2. Anomaly & Pandemic Blackout Filter
   if(InpAvoidPandemicTime && baseTimestamp >= InpPandemicStartTime && baseTimestamp < InpPandemicEndTime)
   {
      return;
   }
   
   // 3. Filter candidate entry bars by Daily Schedule (MT5 Server Time)
   if(!IsTradeScheduleAllowed(baseTimestamp))
   {
      return;
   }
   
   // 4. Extract the flattened vector of all active features across lookback
   vectorf featureVector;
   if(!g_featureExtractor.ExtractFlattenedVector(0, featureVector))
   {
      return;
   }
   
   // 5. Setup Triple Barrier Price Levels (InpLabelMinPoints and InpLabelMaxAdversePoints)
   double point          = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits            = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   long stopsLevel       = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   long spread           = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double ask            = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid            = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double minAllowedDist = (double)(stopsLevel + spread + 5) * point;
   double targetTP       = (double)InpLabelMinPoints * point;
   double targetSL       = (double)InpLabelMaxAdversePoints * point;
   
   // FIN-01: Discard candidate bar if broker constraints exceed pure theoretical barrier targets
   if(targetTP < minAllowedDist || targetSL < minAllowedDist)
   {
      PrintFormat("[DMatrix-EA] [WARNING] Bar %s skipped: broker minimum distance (%.*f) exceeds pure barrier target (TP: %.*f, SL: %.*f, StopsLevel: %d, Spread: %d). Skipping bar to prevent barrier distortion.",
                  TimeToString(baseTimestamp, TIME_DATE|TIME_MINUTES), digits, minAllowedDist,
                  digits, targetTP, digits, targetSL, (int)stopsLevel, (int)spread);
      return;
   }
   
   // Strictly use pure target distances without MathMax inflation
   double slDist         = targetSL;
   double tpDist         = targetTP;
   
   // 6. Open simultaneous BUY and SELL positions with Triple Barrier TP/SL
   // BUY Order: Execution at Ask, closes at Bid -> SL below Bid, TP above Ask
   double buySL      = NormalizeDouble(bid - slDist, digits);
   double buyTP      = NormalizeDouble(ask + tpDist, digits);
   
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   if(g_trade.Buy(InpLotSize, _Symbol, ask, buySL, buyTP, "DMatrix_BUY"))
   {
      ulong buyTicket = 0;
      ulong dealTicket = g_trade.ResultDeal();
      if(dealTicket > 0 && HistoryDealSelect(dealTicket))
         buyTicket = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      if(buyTicket == 0)
         buyTicket = g_trade.ResultOrder();
      if(buyTicket == 0)
         buyTicket = dealTicket;
      // Register in memory with full feature vector (bypassing MT5 comment limit)
      g_orderTracker.RegisterPosition(buyTicket, POSITION_TYPE_BUY, baseTimestamp, ask, buyTP, buySL, featureVector);
   }
   else
   {
      uint retcode = g_trade.ResultRetcode();
      if(retcode == TRADE_RETCODE_MARKET_CLOSED ||
         retcode == TRADE_RETCODE_OFFQUOTES ||
         retcode == TRADE_RETCODE_PRICE_OFF ||
         retcode == TRADE_RETCODE_TRADE_DISABLED ||
         retcode == TRADE_RETCODE_INVALID_STOPS)
      {
         PrintFormat("[DMatrix-EA] [WARNING] Order %s rejected for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s). Skipping bar.",
                     "BUY", _Symbol, digits, ask, digits, bid, digits, buySL, digits, buyTP,
                     (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription());
      }
      else
      {
         PrintFormat("[DMatrix-EA] [ERROR] Order %s failed for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s, Deal: %I64u, Order: %I64u, LastError: %d)",
                     "BUY", _Symbol, digits, ask, digits, bid, digits, buySL, digits, buyTP,
                     (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription(), g_trade.ResultDeal(), g_trade.ResultOrder(), GetLastError());
      }
   }
   
   // SELL Order: Execution at Bid, closes at Ask -> SL above Ask, TP below Bid
   double sellSL     = NormalizeDouble(ask + slDist, digits);
   double sellTP     = NormalizeDouble(bid - tpDist, digits);
   
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   if(g_trade.Sell(InpLotSize, _Symbol, bid, sellSL, sellTP, "DMatrix_SELL"))
   {
      ulong sellTicket = 0;
      ulong dealTicket = g_trade.ResultDeal();
      if(dealTicket > 0 && HistoryDealSelect(dealTicket))
         sellTicket = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      if(sellTicket == 0)
         sellTicket = g_trade.ResultOrder();
      if(sellTicket == 0)
         sellTicket = dealTicket;
      // Register in memory with full feature vector (bypassing MT5 comment limit)
      g_orderTracker.RegisterPosition(sellTicket, POSITION_TYPE_SELL, baseTimestamp, bid, sellTP, sellSL, featureVector);
   }
   else
   {
      uint retcode = g_trade.ResultRetcode();
      if(retcode == TRADE_RETCODE_MARKET_CLOSED ||
         retcode == TRADE_RETCODE_OFFQUOTES ||
         retcode == TRADE_RETCODE_PRICE_OFF ||
         retcode == TRADE_RETCODE_TRADE_DISABLED ||
         retcode == TRADE_RETCODE_INVALID_STOPS)
      {
         PrintFormat("[DMatrix-EA] [WARNING] Order %s rejected for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s). Skipping bar.",
                     "SELL", _Symbol, digits, ask, digits, bid, digits, sellSL, digits, sellTP,
                     (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription());
      }
      else
      {
         PrintFormat("[DMatrix-EA] [ERROR] Order %s failed for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s, Deal: %I64u, Order: %I64u, LastError: %d)",
                     "SELL", _Symbol, digits, ask, digits, bid, digits, sellSL, digits, sellTP,
                     (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription(), g_trade.ResultDeal(), g_trade.ResultOrder(), GetLastError());
      }
   }
}

//+------------------------------------------------------------------+
//| Trade Transaction handler: Processes deal additions in real-time|
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   g_orderTracker.ProcessTransaction(trans, request, result);
}

