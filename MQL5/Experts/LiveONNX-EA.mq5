//+------------------------------------------------------------------+
//|                                                LiveONNX-EA.mq5   |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "Live Microsecond ONNX Inference EA with Direction Filter & GARCH Dynamic Risk"

//+------------------------------------------------------------------+
//| LIVE TRADING EA ARCHITECTURE & INFERENCE PIPELINE:               |
//|                                                                  |
//| 1. High-Performance Native ONNX Engine:                          |
//|    - Loads 'model_buy.onnx' and 'model_sell.onnx' directly into  |
//|      MT5's high-speed C++ ONNX runtime handle.                   |
//|    - Sets explicit input shape [1, num_features] and output shape|
//|      [1, 2] in OnInit to avoid runtime tensor reallocation.      |
//|                                                                  |
//| 2. Zero-Copy Execution (ONNX_NO_CONVERSION):                     |
//|    - Utilizes native MQL5 'vectorf' single-precision float       |
//|      arrays matching the ONNX FloatTensorType directly.          |
//|    - Microsecond inference latency suitable for institutional    |
//|      automated quantitative execution.                           |
//|                                                                  |
//| 3. Execution & Dynamic Risk Management:                          |
//|    - Direction: BOTH (0), ONLY_BUY (1), ONLY_SELL (2).           |
//|    - Dynamic GARCH(1,1) analytical multi-step horizon variance   |
//|      strictly matching DMatrix-EA training labels (kTP / kSL).   |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
#include "..\Include\GarchEngine.mqh"
#include "..\Include\FeatureExtractor.mqh"
#include "..\Include\ConsecutiveManager.mqh"
#include "..\Include\ExecutionAuditor.mqh"

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
//| CUSTOM ENUMERATIONS                                              |
//+------------------------------------------------------------------+
enum ENUM_TRADE_DIRECTION
{
   DIRECTION_BOTH      = 0, // Both (BUY & SELL)
   DIRECTION_ONLY_BUY  = 1, // Only BUY
   DIRECTION_ONLY_SELL = 2  // Only SELL
};

enum ENUM_SR_ZONE_SELECTION
{
   SR_ZONE_CLOSEST  = 0, // Closest to Entry (First Barrier - Higher Winrate)
   SR_ZONE_FURTHEST = 1  // Furthest within GARCH (Max Reach - Higher Profit)
};

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                 |
//+------------------------------------------------------------------+
//--- 1. Trading & Execution Settings
input group "=== Trading & Execution Settings ==="
input ENUM_TRADE_DIRECTION InpTradeDirection           = DIRECTION_BOTH;          // Trade Direction Mode
input double               InpMinimalLevelAcceptedBuy  = 0.50;                    // Minimum BUY Probability Threshold
input double               InpMinimalLevelAcceptedSell = 0.50;                    // Minimum SELL Probability Threshold
input double               InpLotSize                  = 0.01;                    // Trade Lot Size
input ulong                InpMagicNumber              = 222100;                  // EA Magic Number

//--- 2. Daily Schedule & Session Filter Settings (MT5 Server Time)
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

//--- 3. Structural S&R Snapping (Superimposed over GARCH)
input group "=== Structural S&R Snapping (Over GARCH) ==="
input bool                 InpEnableSRSnapping         = true;                    // Enable S&R Structural Snapping over GARCH
input int                  InpSRLookbackBars           = 12;                      // Candles to Scan for S&R Zones (e.g. 5 to 65)
input int                  InpSRPivotStrength          = 2;                       // S&R Fractal Pivot Radius K (1=3 bars, 2=5 bars, 3=7 bars)
input int                  InpSROffsetPoints           = 30;                      // Zone Adjustment Offset in Points (SL further, TP closer)
input ENUM_SR_ZONE_SELECTION InpSRZoneSelection       = SR_ZONE_CLOSEST;         // S&R Zone Target Selection (Closest vs Furthest)

//--- 4. Risk & Margin Governance (Viability Filter)
input group "=== Risk & Margin Governance (Viability Filter) ==="
input bool                 InpEnableRiskFilter         = true;                    // Enable Risk & Margin Governance Filter
input bool                 InpEnableDynamicLotSizing   = false;                   // Enable Dynamic Lot Sizing & Risk Fitting
input double               InpMaxLotSize               = 0.05;                    // Max Starting Lot Size for Dynamic Sizing
input double               InpMarginSafetyMultiplier   = 1.5;                     // Broker Margin Call Safety Multiplier (e.g. 1.5x)
input double               InpMaxRiskRewardRatio       = 1.5;                     // Max Risk-Reward Ratio (SL_points / TP_points)
input double               InpMaxTradeRiskPct          = 3.0;                     // Maximum Trade Loss Budget (% of Equity)

//--- 5. Consecutive Signal & Position Management
input group "=== Consecutive Signal Management ==="
input ENUM_CONSECUTIVE_SIGNAL_MODE InpConsecutiveMode            = CONSECUTIVE_MODE_LEGACY_INDEPENDENT; // Consecutive Signal Mode
input int                          InpMaxConsecutiveOrders      = 3;                                  // Max Consecutive Orders (0 = Unlimited)
input double                       InpHurdleProfitPct           = 50.0;                               // Hurdle Profit % of Initial TP (Mode 1)
input double                       InpProfitLockPct             = 50.0;                               // Retained Profit % Locked into SL (Mode 1)
input int                          InpAntiChopMinDisplacement   = 150;                                // Anti-Chop Min Displacement in Points (Mode 2)
input int                          InpSafetyOffsetPoints        = 20;                                 // Safety Buffer Offset in Points
input bool                         InpEnableSwapAmortization    = true;                               // Amortize Accrued Swap in SL (Net Breakeven)
input bool                         InpConsecutiveSlotFilter     = false;                              // Require New Slot Amplitude >= Current Slot
input bool                         InpIgnoreConflictingSignals  = true;                               // Ignore Same-Candle Conflicting Signals
input bool                         InpEnableOpposingRegimeFilter= false;                              // Enable Opposing Regime Filter (ML Counter)
input int                          InpOpposingStreakThreshold   = 2;                                  // Opposing Signal Streak Threshold (N Bars)
input ENUM_OPPOSING_DEFENSIVE_ACTION InpOpposingAction          = OPPOSING_ACTION_CLOSE_IF_PROFIT;    // Opposing Defensive Action
input int                          InpOpposingTrailingPoints    = 50;                                 // Opposing Trailing Points (Defensive Trailing)
input double                       InpOpposingRecalculateRatio  = 0.5;                                // Opposing Barrier Recalculate Ratio (0.1 - 0.9)

//--- Static SQLite Macro Database Identifier (Stored in MT5 Common/Files)
const string               MACRO_DATABASE_NAME         = "macro_governance.db";

//--- 6. Economic Calendar Filter (SQLite Database)
input group "=== Economic Calendar Filter (SQLite) ==="
input bool                 InpEnableCalendarFilter     = true;                    // Enable Economic Calendar Filter (Live & Tester)

//--- 6. Global News Blacklist Filter (SQLite Database)
input group "=== Global News Blacklist Filter (SQLite) ==="
input bool                 InpEnableNewsFilter         = true;                    // Enable Global News Blacklist (Live Only)

//--- 7. Dynamic GARCH Risk Parameters (Execution Sizing)
input group "=== Dynamic GARCH Risk Parameters ==="
input int                  InpRiskGarchHorizon         = 8;                       // Dynamic Risk Volatility Forecast Horizon (Bars)
input double               InpKTP                      = 1.5;                     // Dynamic GARCH kTP Multiplier
input double               InpKSL                      = 1.5;                     // Dynamic GARCH kSL Multiplier

//--- 8. GARCH(1,1) Volatility Forecast Settings (Feature Parity with DMatrix)
input group "=== GARCH(1,1) Volatility Forecast Settings ==="
input int                  InpGarchHorizon             = 8;                       // GARCH Forecast Horizon (Future bars: t+1..t+H)
input int                  InpPriceSize                = 500;                     // GARCH Historical Sample Size (bars)
input double               InpGarchAlpha               = 0.05;                    // GARCH Alpha (ARCH shock weight)
input double               InpGarchBeta                = 0.92;                    // GARCH Beta (GARCH variance persistence)

//--- 9. Feature Configuration (Strict Parity with DMatrix-EA)
input group "=== Feature Configuration (Parity with DMatrix) ==="
input int                  InpFeatureLookback       = 4;                       // Feature Lookback Lags (t, t-1..t-N)
input bool                 InpUseADX                = true;                    // Include iADX (Main, +DI, -DI)
input bool                 InpUseATR                = true;                    // Include iATR (Volatility)
input bool                 InpUseBands              = true;                    // Include iBands (Diff Mid, Bandwidth)
input bool                 InpUseMACD               = true;                    // Include iMACD (Main, Signal)
input bool                 InpUseFastMA             = true;                    // Include Fast iMA Diff
input bool                 InpUseSlowMA             = true;                    // Include Slow iMA Diff
input bool                 InpUseRSI                = true;                    // Include iRSI (Momentum)
input bool                 InpUseStochastic         = true;                    // Include iStochastic (K, D)
input bool                 InpUseCandlestick        = true;                    // Include Candlestick (Type, Body, Shadows)
input bool                 InpUseTimestampWeek      = true;                    // Include Weekday (0f=Mon ... 4f=Fri)
input bool                 InpUseTimestampDay       = true;                    // Include Quarter Day (0f=00-06h ... 3f=18-24h)
input bool                 InpUseOpenMarkets        = true;                    // Include Open Markets Session Code (0f-7f)
input bool                 InpUseSpread             = true;                    // Include Current Spread in Points
input bool                 InpUseGarchFeatures      = true;                    // Include GARCH(1,1) Volatility Features

//--- 10. ONNX Model File Paths (Optional Override)
input group "=== ONNX Model Files ==="
input string               InpModelBuyPath          = "";                      // BUY Model Path (empty = auto)
input string               InpModelSellPath         = "";                      // SELL Model Path (empty = auto)

//--- 11. Indicator Parameters
input group "=== Indicator Parameters ==="
input int                  InpADXPeriod             = 14;                      // ADX Period
input int                  InpATRPeriod             = 14;                      // ATR Period
input int                  InpBandsPeriod           = 20;                      // Bollinger Bands Period
input int                  InpBandsShift            = 0;                       // Bollinger Bands Shift
input double               InpBandsDev              = 2.0;                     // Bollinger Bands Deviation
input ENUM_APPLIED_PRICE   InpBandsAppliedPrice     = PRICE_CLOSE;             // Bollinger Bands Applied Price
input int                  InpMACDFastPeriod        = 12;                      // MACD Fast EMA Period
input int                  InpMACDSlowPeriod        = 26;                      // MACD Slow EMA Period
input int                  InpMACDSignalPeriod      = 9;                       // MACD Signal SMA Period
input ENUM_APPLIED_PRICE   InpMACDAppliedPrice      = PRICE_CLOSE;             // MACD Applied Price
input int                  InpFastMAPeriod          = 20;                      // Fast MA Period
input int                  InpFastMAShift           = 0;                       // Fast MA Shift
input ENUM_MA_METHOD       InpFastMAMethod          = MODE_EMA;                // Fast MA Method
input ENUM_APPLIED_PRICE   InpFastMAAppliedPrice    = PRICE_CLOSE;             // Fast MA Applied Price
input int                  InpSlowMAPeriod          = 50;                      // Slow MA Period
input int                  InpSlowMAShift           = 0;                       // Slow MA Shift
input ENUM_MA_METHOD       InpSlowMAMethod          = MODE_EMA;                // Slow MA Method
input ENUM_APPLIED_PRICE   InpSlowMAAppliedPrice    = PRICE_CLOSE;             // Slow MA Applied Price
input int                  InpRSIPeriod             = 14;                      // RSI Period
input ENUM_APPLIED_PRICE   InpRSIAppliedPrice       = PRICE_CLOSE;             // RSI Applied Price
input int                  InpStochK                = 8;                       // Stochastic %K Period
input int                  InpStochD                = 3;                       // Stochastic %D Period
input int                  InpStochSlowing          = 3;                       // Stochastic Slowing
input ENUM_MA_METHOD       InpStochMethod           = MODE_SMA;                // Stochastic MA Method
input ENUM_STO_PRICE       InpStochPriceField       = STO_LOWHIGH;             // Stochastic Price Field

//--- 12. Execution & Telemetry Audit Settings
input group "=== Execution & Telemetry Audit Settings ==="
input bool                 InpIgnoreAudit           = false;                   // Ignore Audit Subsystem (Bypass SQLite .db creation and telemetry)

//+------------------------------------------------------------------+
//| STRUCTURES FOR HIGH-PERFORMANCE EXECUTION & CACHING              |
//+------------------------------------------------------------------+
struct SDaySchedule
{
   bool isEnabled;
   int  startSeconds;
   int  endSeconds;
};

struct SMacroEventCache
{
   datetime lastCheckBarTime;
   bool     hasCalendarEvent;
   string   calTitle;
   string   calDesc;
   string   calAction;
   int      calTrailingPoints;
   
   datetime lastNewsCheckTime;
   bool     hasNewsEvent;
   string   newsTitle;
   string   newsDesc;
   string   newsAction;
   int      newsTrailingPoints;
};

//+------------------------------------------------------------------+
//| GLOBAL STATE & OBJECTS                                           |
//+------------------------------------------------------------------+
CTrade              g_trade;
CGarchEngine        g_garch;
CFeatureExtractor   g_featureExtractor;
CConsecutiveManager g_consecutiveManager;
CExecutionAuditor   g_auditor;
SFeatureConfig      g_config;

long              g_hModelBuy   = INVALID_HANDLE;
long              g_hModelSell  = INVALID_HANDLE;
datetime          g_lastBarTime = 0;
int               g_featureCount = 0;
int               g_hMacroDB    = INVALID_HANDLE;

SDaySchedule      g_daySchedules[5]; // 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
SMacroEventCache  g_macroCache;

//+------------------------------------------------------------------+
//| Active Trade In-Memory Lifecycle Metadata Tracking               |
//+------------------------------------------------------------------+
struct SActiveTradeMetadata
{
   ulong    positionId;
   ulong    entryDealTicket;
   datetime openTime;
   string   orderType;
   double   volume;
   double   targetEntryPrice;
   double   actualEntryPrice;
   double   entrySlippagePoints;
   ulong    orderLatencyMs;
   double   initialTP;
   double   initialSL;
   double   maxFavorablePrice;
   double   maxAdversePrice;
};
SActiveTradeMetadata g_activeTrades[];

void RegisterActiveTrade(const ulong posId, const ulong entryDeal, const datetime openT, const string orderType,
                         const double vol, const double targetPrice, const double actualPrice,
                         const double slippagePts, const ulong latencyMs, const double tp, const double sl)
{
   int size = ArraySize(g_activeTrades);
   ArrayResize(g_activeTrades, size + 1);
   g_activeTrades[size].positionId          = posId;
   g_activeTrades[size].entryDealTicket     = entryDeal;
   g_activeTrades[size].openTime            = openT;
   g_activeTrades[size].orderType           = orderType;
   g_activeTrades[size].volume              = vol;
   g_activeTrades[size].targetEntryPrice    = targetPrice;
   g_activeTrades[size].actualEntryPrice    = actualPrice;
   g_activeTrades[size].entrySlippagePoints = slippagePts;
   g_activeTrades[size].orderLatencyMs      = latencyMs;
   g_activeTrades[size].initialTP           = tp;
   g_activeTrades[size].initialSL           = sl;
   g_activeTrades[size].maxFavorablePrice   = actualPrice;
   g_activeTrades[size].maxAdversePrice     = actualPrice;
}

int FindActiveTrade(const ulong posId)
{
   int size = ArraySize(g_activeTrades);
   for(int i = 0; i < size; i++)
   {
      if(g_activeTrades[i].positionId == posId)
         return i;
   }
   return -1;
}

void RemoveActiveTrade(const int index)
{
   int size = ArraySize(g_activeTrades);
   if(index < 0 || index >= size) return;
   for(int i = index; i < size - 1; i++)
   {
      g_activeTrades[i] = g_activeTrades[i + 1];
   }
   ArrayResize(g_activeTrades, size - 1);
}

void UpdateActiveTradesExcursion()
{
   int size = ArraySize(g_activeTrades);
   if(size == 0) return;
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(currentBid <= 0.0 || currentAsk <= 0.0) return;
   
   for(int i = 0; i < size; i++)
   {
      if(g_activeTrades[i].orderType == "BUY")
      {
         if(currentBid > g_activeTrades[i].maxFavorablePrice)
            g_activeTrades[i].maxFavorablePrice = currentBid;
         if(currentBid < g_activeTrades[i].maxAdversePrice)
            g_activeTrades[i].maxAdversePrice = currentBid;
      }
      else if(g_activeTrades[i].orderType == "SELL")
      {
         if(currentAsk < g_activeTrades[i].maxFavorablePrice)
            g_activeTrades[i].maxFavorablePrice = currentAsk;
         if(currentAsk > g_activeTrades[i].maxAdversePrice)
            g_activeTrades[i].maxAdversePrice = currentAsk;
      }
   }
}

double CalculateShannonEntropy(const double p)
{
   if(p <= 0.00001 || p >= 0.99999) return 0.0;
   double q = 1.0 - p;
   return -(p * MathLog(p) + q * MathLog(q)) / 0.6931471805599453; // ln(2) conversion
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
//| configured daily trading window in MT5 Server Time (O(1) lookup) |
//+------------------------------------------------------------------+
bool IsTradeScheduleAllowed(const datetime barTime)
{
   MqlDateTime dt;
   TimeToStruct(barTime, dt);
   
   // MT5 dt.day_of_week: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
   if(dt.day_of_week < 1 || dt.day_of_week > 5)
      return false; // Weekend
      
   int dayIdx = dt.day_of_week - 1; // Map 1..5 to 0..4
   if(!g_daySchedules[dayIdx].isEnabled)
      return false;
      
   // Daily bars and higher open at 00:00:00 MT5 Server Time.
   // If the day is enabled, daily bars are allowed.
   if(_Period >= PERIOD_D1)
      return true;
      
   int startSec = g_daySchedules[dayIdx].startSeconds;
   int endSec   = g_daySchedules[dayIdx].endSeconds;
   
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
//| ValidateInputParameters: Exhaustive pre-flight domain checks     |
//+------------------------------------------------------------------+
bool ValidateInputParameters(string &outErrorMsg)
{
   outErrorMsg = "";
   
   // 1. Probabilities & Trading Direction
   if(InpMinimalLevelAcceptedBuy < 0.0 || InpMinimalLevelAcceptedBuy > 1.0)
   {
      outErrorMsg = StringFormat("InpMinimalLevelAcceptedBuy (%.2f) must be in range [0.0, 1.0]", InpMinimalLevelAcceptedBuy);
      return false;
   }
   if(InpMinimalLevelAcceptedSell < 0.0 || InpMinimalLevelAcceptedSell > 1.0)
   {
      outErrorMsg = StringFormat("InpMinimalLevelAcceptedSell (%.2f) must be in range [0.0, 1.0]", InpMinimalLevelAcceptedSell);
      return false;
   }
   
   // 2. Broker Volume & Lot Sizing
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(minLot <= 0.0) minLot = 0.01;
   if(maxLot <= 0.0) maxLot = 100.0;
   
   if(InpLotSize < minLot || InpLotSize > maxLot)
   {
      outErrorMsg = StringFormat("InpLotSize (%.2f) is out of broker bounds [%.2f, %.2f] for %s", InpLotSize, minLot, maxLot, _Symbol);
      return false;
   }
   
   if(InpEnableDynamicLotSizing)
   {
      if(InpMaxLotSize < minLot || InpMaxLotSize > maxLot)
      {
         outErrorMsg = StringFormat("InpMaxLotSize (%.2f) is out of broker bounds [%.2f, %.2f] for %s", InpMaxLotSize, minLot, maxLot, _Symbol);
         return false;
      }
      if(InpMaxLotSize < InpLotSize)
      {
         PrintFormat("[LiveONNX-EA] [WARNING] InpMaxLotSize (%.2f) is less than InpLotSize (%.2f). Max lot will be bounded by InpMaxLotSize.",
                     InpMaxLotSize, InpLotSize);
      }
   }
   
   // 3. Weekly Schedule & Session Inversion Checks
   if(!InpTradeMonday && !InpTradeTuesday && !InpTradeWednesday && !InpTradeThursday && !InpTradeFriday)
   {
      outErrorMsg = "All trading weekdays are disabled (InpTradeMonday through InpTradeFriday are all false)";
      return false;
   }
   
   // Check Monday
   if(InpTradeMonday)
   {
      int startSec = TimeStringToSeconds(InpMondayStartTime);
      int endSec   = TimeStringToSeconds(InpMondayEndTime);
      if(endSec != 0 && startSec == endSec)
      {
         outErrorMsg = StringFormat("Monday start time (%s) cannot equal end time (%s)", InpMondayStartTime, InpMondayEndTime);
         return false;
      }
   }
   // Check Tuesday
   if(InpTradeTuesday)
   {
      int startSec = TimeStringToSeconds(InpTuesdayStartTime);
      int endSec   = TimeStringToSeconds(InpTuesdayEndTime);
      if(endSec != 0 && startSec == endSec)
      {
         outErrorMsg = StringFormat("Tuesday start time (%s) cannot equal end time (%s)", InpTuesdayStartTime, InpTuesdayEndTime);
         return false;
      }
   }
   // Check Wednesday
   if(InpTradeWednesday)
   {
      int startSec = TimeStringToSeconds(InpWednesdayStartTime);
      int endSec   = TimeStringToSeconds(InpWednesdayEndTime);
      if(endSec != 0 && startSec == endSec)
      {
         outErrorMsg = StringFormat("Wednesday start time (%s) cannot equal end time (%s)", InpWednesdayStartTime, InpWednesdayEndTime);
         return false;
      }
   }
   // Check Thursday
   if(InpTradeThursday)
   {
      int startSec = TimeStringToSeconds(InpThursdayStartTime);
      int endSec   = TimeStringToSeconds(InpThursdayEndTime);
      if(endSec != 0 && startSec == endSec)
      {
         outErrorMsg = StringFormat("Thursday start time (%s) cannot equal end time (%s)", InpThursdayStartTime, InpThursdayEndTime);
         return false;
      }
   }
   // Check Friday
   if(InpTradeFriday)
   {
      int startSec = TimeStringToSeconds(InpFridayStartTime);
      int endSec   = TimeStringToSeconds(InpFridayEndTime);
      if(endSec != 0 && startSec == endSec)
      {
         outErrorMsg = StringFormat("Friday start time (%s) cannot equal end time (%s)", InpFridayStartTime, InpFridayEndTime);
         return false;
      }
   }
   
   // 4. Dynamic GARCH Risk Parameters
   if(InpKTP <= 0.0)
   {
      outErrorMsg = StringFormat("InpKTP (%.2f) must be strictly positive (> 0.0)", InpKTP);
      return false;
   }
   if(InpKSL <= 0.0)
   {
      outErrorMsg = StringFormat("InpKSL (%.2f) must be strictly positive (> 0.0)", InpKSL);
      return false;
   }
   if(InpRiskGarchHorizon < 1)
   {
      outErrorMsg = StringFormat("InpRiskGarchHorizon (%d) must be >= 1", InpRiskGarchHorizon);
      return false;
   }
   if(InpGarchHorizon < 1)
   {
      outErrorMsg = StringFormat("InpGarchHorizon (%d) must be >= 1", InpGarchHorizon);
      return false;
   }
   if(InpPriceSize < 30)
   {
      outErrorMsg = StringFormat("InpPriceSize (%d) must be >= 30 for valid statistical variance estimation", InpPriceSize);
      return false;
   }
   if(InpGarchAlpha <= 0.0 || InpGarchBeta <= 0.0 || (InpGarchAlpha + InpGarchBeta) >= 1.0)
   {
      outErrorMsg = StringFormat("GARCH parameters violate stationarity: alpha=%.4f, beta=%.4f (alpha+beta must be < 1.0)",
                                 InpGarchAlpha, InpGarchBeta);
      return false;
   }
   
   // 5. Structural S&R Snapping Parameters
   if(InpEnableSRSnapping)
   {
      if(InpSRLookbackBars < 5)
      {
         outErrorMsg = StringFormat("InpSRLookbackBars (%d) must be >= 5 for fractal scanning", InpSRLookbackBars);
         return false;
      }
      if(InpSRPivotStrength < 1)
      {
         outErrorMsg = StringFormat("InpSRPivotStrength (%d) must be >= 1", InpSRPivotStrength);
         return false;
      }
      if(InpSROffsetPoints < 0)
      {
         outErrorMsg = StringFormat("InpSROffsetPoints (%d) cannot be negative", InpSROffsetPoints);
         return false;
      }
   }
   
   // 6. Risk & Margin Governance Parameters
   if(InpEnableRiskFilter)
   {
      if(InpMarginSafetyMultiplier < 1.0)
      {
         outErrorMsg = StringFormat("InpMarginSafetyMultiplier (%.2f) must be >= 1.0", InpMarginSafetyMultiplier);
         return false;
      }
      if(InpMaxRiskRewardRatio > 0.0 && InpMaxRiskRewardRatio < 0.1)
      {
         outErrorMsg = StringFormat("InpMaxRiskRewardRatio (%.2f) is too restrictive (must be >= 0.1 if enabled)", InpMaxRiskRewardRatio);
         return false;
      }
      if(InpMaxTradeRiskPct > 0.0 && InpMaxTradeRiskPct > 100.0)
      {
         outErrorMsg = StringFormat("InpMaxTradeRiskPct (%.2f) cannot exceed 100.0%% of equity", InpMaxTradeRiskPct);
         return false;
      }
   }
   
   // 7. Consecutive Signal Management Parameters
   if(InpHurdleProfitPct < 0.0 || InpHurdleProfitPct > 100.0)
   {
      outErrorMsg = StringFormat("InpHurdleProfitPct (%.2f) must be between 0.0 and 100.0", InpHurdleProfitPct);
      return false;
   }
   if(InpProfitLockPct < 0.0 || InpProfitLockPct > 100.0)
   {
      outErrorMsg = StringFormat("InpProfitLockPct (%.2f) must be between 0.0 and 100.0", InpProfitLockPct);
      return false;
   }
   if(InpMaxConsecutiveOrders < 0)
   {
      outErrorMsg = StringFormat("InpMaxConsecutiveOrders (%d) cannot be negative", InpMaxConsecutiveOrders);
      return false;
   }
   if(InpAntiChopMinDisplacement < 0)
   {
      outErrorMsg = StringFormat("InpAntiChopMinDisplacement (%d) cannot be negative", InpAntiChopMinDisplacement);
      return false;
   }
   if(InpSafetyOffsetPoints < 0)
   {
      outErrorMsg = StringFormat("InpSafetyOffsetPoints (%d) cannot be negative", InpSafetyOffsetPoints);
      return false;
   }
   if(InpEnableOpposingRegimeFilter)
   {
      if(InpOpposingStreakThreshold < 1)
      {
         outErrorMsg = StringFormat("InpOpposingStreakThreshold (%d) must be >= 1", InpOpposingStreakThreshold);
         return false;
      }
      if(InpOpposingTrailingPoints < 0)
      {
         outErrorMsg = StringFormat("InpOpposingTrailingPoints (%d) cannot be negative", InpOpposingTrailingPoints);
         return false;
      }
      if(InpOpposingRecalculateRatio <= 0.0 || InpOpposingRecalculateRatio >= 1.0)
      {
         outErrorMsg = StringFormat("InpOpposingRecalculateRatio (%.2f) must be between 0.0 and 1.0 exclusive", InpOpposingRecalculateRatio);
         return false;
      }
   }
   
   // 8. Technical Indicator Parameter Relationships
   if(InpMACDFastPeriod <= 0 || InpMACDSlowPeriod <= 0 || InpMACDFastPeriod >= InpMACDSlowPeriod)
   {
      outErrorMsg = StringFormat("MACD parameters invalid: Fast (%d) must be < Slow (%d) and > 0", InpMACDFastPeriod, InpMACDSlowPeriod);
      return false;
   }
   if(InpFastMAPeriod <= 0 || InpSlowMAPeriod <= 0 || InpFastMAPeriod >= InpSlowMAPeriod)
   {
      outErrorMsg = StringFormat("Moving Average parameters invalid: Fast (%d) must be < Slow (%d) and > 0", InpFastMAPeriod, InpSlowMAPeriod);
      return false;
   }
   if(InpADXPeriod <= 1 || InpATRPeriod <= 1 || InpBandsPeriod <= 1 || InpRSIPeriod <= 1 || InpStochK <= 1)
   {
      outErrorMsg = "Indicator smoothing periods (ADX, ATR, Bands, RSI, StochK) must all be > 1";
      return false;
   }
   
   // 8. Initial Free Margin Viability Check
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0) ask = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask > 0.0)
   {
      double minReqMargin = 0.0;
      if(OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, minLot, ask, minReqMargin) && minReqMargin > 0.0)
      {
         if(freeMargin > 0.0 && freeMargin < minReqMargin)
         {
            outErrorMsg = StringFormat("Account free margin (%.2f) is insufficient for minimum lot %.2f (required: %.2f)",
                                       freeMargin, minLot, minReqMargin);
            return false;
         }
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| IsNewBar: Detects when a new bar begins                          |
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
//| LoadModelWithFallback: Tries multiple standard ONNX model paths  |
//+------------------------------------------------------------------+
long LoadModelWithFallback(const string specifiedPath, const string direction)
{
   string tfName = EnumToString(_Period);
   StringReplace(tfName, "PERIOD_", "");
   
   string pathsToTry[];
   ArrayResize(pathsToTry, 6);
   int count = 0;
   
   if(StringLen(specifiedPath) > 0)
      pathsToTry[count++] = specifiedPath;
      
   pathsToTry[count++] = "Models/" + _Symbol + "_" + tfName + "_model_" + direction + ".onnx";
   pathsToTry[count++] = _Symbol + "_" + tfName + "_model_" + direction + ".onnx";
   pathsToTry[count++] = "Models/model_" + direction + ".onnx";
   pathsToTry[count++] = "model_" + direction + ".onnx";
   
   for(int i = 0; i < count; i++)
   {
      // 1. Try Local Terminal Folder (MQL5\Files)
      long handle = OnnxCreate(pathsToTry[i], ONNX_DEFAULT);
      if(handle != INVALID_HANDLE)
      {
         PrintFormat("[LiveONNX-EA] Loaded %s ONNX model from local Files: %s", direction, pathsToTry[i]);
         return handle;
      }
      
      // 2. Try Shared Common Folder (Common\Files)
      handle = OnnxCreate(pathsToTry[i], ONNX_COMMON_FOLDER);
      if(handle != INVALID_HANDLE)
      {
         PrintFormat("[LiveONNX-EA] Loaded %s ONNX model from Common Files: %s", direction, pathsToTry[i]);
         return handle;
      }
   }
   return INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Structural Swing High / Swing Low Fractal Detection Helpers       |
//+------------------------------------------------------------------+
bool IsSwingHigh(const MqlRates &rates[], const int index, const int strength, const int total)
{
   if(index - strength < 0 || index + strength >= total)
      return false;
   double h = rates[index].high;
   for(int j = 1; j <= strength; j++)
   {
      if(rates[index - j].high >= h || rates[index + j].high >= h)
         return false;
   }
   return true;
}

bool IsSwingLow(const MqlRates &rates[], const int index, const int strength, const int total)
{
   if(index - strength < 0 || index + strength >= total)
      return false;
   double l = rates[index].low;
   for(int j = 1; j <= strength; j++)
   {
      if(rates[index - j].low <= l || rates[index + j].low <= l)
         return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| ApplyStructuralSRSnapping: Refines base GARCH TP and SL levels   |
//| by snapping to real Support and Resistance zones within lookback |
//+------------------------------------------------------------------+
bool ApplyStructuralSRSnapping(const string symbol, const ENUM_TIMEFRAMES period,
                               const bool isBuy, const double ask, const double bid,
                               const int lookbackBars, const int offsetPoints,
                               const ENUM_SR_ZONE_SELECTION zoneSelection,
                               const int pivotStrength,
                               const double garchSL, const double garchTP,
                               const double minDistancePoints,
                               double &outSL, double &outTP)
{
   // Default output strictly preserves baseline dynamic GARCH levels
   outSL = garchSL;
   outTP = garchTP;
   
   int k = (pivotStrength >= 1) ? pivotStrength : 2;
   int lookback = (lookbackBars >= 5) ? lookbackBars : 12;
   int totalBars = lookback + k;
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, period, 1, totalBars, rates);
   if(copied < totalBars)
   {
      PrintFormat("[LiveONNX-EA] [WARNING] Failed to copy %d rates for S&R snapping (got %d). Retaining GARCH.", totalBars, copied);
      return false;
   }
   
   double point   = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits     = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double offset  = (double)MathMax(offsetPoints, 0) * point;
   double minDist = minDistancePoints * point;
   
   bool snappedAny = false;
   
   if(isBuy)
   {
      // --- BUY ORDER ---
      // 1. Take Profit: Scan for confirmed Swing High (Resistance) between entry (Ask) and GARCH TP
      double bestResistance = 0.0;
      for(int i = k; i < lookback; i++)
      {
         if(IsSwingHigh(rates, i, k, copied))
         {
            if(rates[i].high > ask && rates[i].high <= garchTP)
            {
               if(zoneSelection == SR_ZONE_CLOSEST)
               {
                  // First barrier: lowest ceiling above ask (closest to open price)
                  if(bestResistance == 0.0 || rates[i].high < bestResistance)
                     bestResistance = rates[i].high;
               }
               else // SR_ZONE_FURTHEST
               {
                  // Highest ceiling below garchTP (max structural reach)
                  if(rates[i].high > bestResistance)
                     bestResistance = rates[i].high;
               }
            }
         }
      }
      
      if(bestResistance > 0.0)
      {
         // Offset pulls TP closer to entry price (before the resistance) to guarantee execution
         double candidateTP = bestResistance - offset;
         if((candidateTP - ask) >= minDist)
         {
            outTP = NormalizeDouble(candidateTP, digits);
            snappedAny = true;
         }
      }
      
      // 2. Stop Loss: Scan for confirmed Swing Low (Support) below entry (Bid) within GARCH SL
      double bestSupport = 0.0;
      for(int i = k; i < lookback; i++)
      {
         if(IsSwingLow(rates, i, k, copied))
         {
            if(rates[i].low < bid && rates[i].low >= garchSL)
            {
               if(zoneSelection == SR_ZONE_CLOSEST)
               {
                  // Highest floor below bid (closest structural protection)
                  if(bestSupport == 0.0 || rates[i].low > bestSupport)
                     bestSupport = rates[i].low;
               }
               else // SR_ZONE_FURTHEST
               {
                  // Deepest floor protecting position (max room against sweeps)
                  if(bestSupport == 0.0 || rates[i].low < bestSupport)
                     bestSupport = rates[i].low;
               }
            }
         }
      }
      
      if(bestSupport > 0.0)
      {
         // Offset pushes SL further away from entry price (below support) to prevent sweeps
         double candidateSL = bestSupport - offset;
         if(candidateSL < garchSL) candidateSL = garchSL; // Clamp to GARCH risk envelope
         if((bid - candidateSL) >= minDist)
         {
            outSL = NormalizeDouble(candidateSL, digits);
            snappedAny = true;
         }
      }
   }
   else
   {
      // --- SELL ORDER ---
      // 1. Take Profit: Scan for confirmed Swing Low (Support) between GARCH TP and entry (Bid)
      double bestSupport = 0.0;
      for(int i = k; i < lookback; i++)
      {
         if(IsSwingLow(rates, i, k, copied))
         {
            if(rates[i].low < bid && rates[i].low >= garchTP)
            {
               if(zoneSelection == SR_ZONE_CLOSEST)
               {
                  // First barrier: highest floor below bid (closest to open price)
                  if(bestSupport == 0.0 || rates[i].low > bestSupport)
                     bestSupport = rates[i].low;
               }
               else // SR_ZONE_FURTHEST
               {
                  // Deepest floor above garchTP (max structural reach)
                  if(bestSupport == 0.0 || rates[i].low < bestSupport)
                     bestSupport = rates[i].low;
               }
            }
         }
      }
      
      if(bestSupport > 0.0)
      {
         // Offset pulls TP closer to entry price (above the support) to guarantee execution
         double candidateTP = bestSupport + offset;
         if((bid - candidateTP) >= minDist)
         {
            outTP = NormalizeDouble(candidateTP, digits);
            snappedAny = true;
         }
      }
      
      // 2. Stop Loss: Scan for confirmed Swing High (Resistance) above entry (Ask) within GARCH SL
      double bestResistance = 0.0;
      for(int i = k; i < lookback; i++)
      {
         if(IsSwingHigh(rates, i, k, copied))
         {
            if(rates[i].high > ask && rates[i].high <= garchSL)
            {
               if(zoneSelection == SR_ZONE_CLOSEST)
               {
                  // Lowest ceiling above ask (closest structural protection)
                  if(bestResistance == 0.0 || rates[i].high < bestResistance)
                     bestResistance = rates[i].high;
               }
               else // SR_ZONE_FURTHEST
               {
                  // Highest ceiling protecting position (max room against sweeps)
                  if(rates[i].high > bestResistance)
                     bestResistance = rates[i].high;
               }
            }
         }
      }
      
      if(bestResistance > 0.0)
      {
         // Offset pushes SL further away from entry price (above resistance) to prevent sweeps
         double candidateSL = bestResistance + offset;
         if(candidateSL > garchSL) candidateSL = garchSL; // Clamp to GARCH risk envelope
         if((candidateSL - ask) >= minDist)
         {
            outSL = NormalizeDouble(candidateSL, digits);
            snappedAny = true;
         }
      }
   }
   
   ArrayFree(rates);
   return snappedAny;
}

//+------------------------------------------------------------------+
//| CheckTradeViability: Evaluates 3 quantitative governance gates   |
//| (Margin/Leverage cushion, Asymmetric SL/TP, and Max Risk % loss) |
//+------------------------------------------------------------------+
bool CheckTradeViability(const ENUM_ORDER_TYPE orderType, const string symbol,
                         const double lot, const double openPrice,
                         const double slPrice, const double tpPrice,
                         string &outRejectReason, int &outRejectedGate)
{
   outRejectReason = "";
   outRejectedGate = 0;
   
   // 1. Gate 1: Margin & Leverage Cushion Check
   double reqMargin = 0.0;
   if(!OrderCalcMargin(orderType, symbol, lot, openPrice, reqMargin) || reqMargin <= 0.0)
   {
      outRejectReason = "Broker failed to calculate margin requirement for symbol (OrderCalcMargin failed)";
      outRejectedGate = 1;
      return false;
   }
   
   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double currMargin = AccountInfoDouble(ACCOUNT_MARGIN);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   
   if(freeMargin < reqMargin)
   {
      outRejectReason = StringFormat("Insufficient free margin (Required: %.2f, Free: %.2f)",
                                     reqMargin, freeMargin);
      outRejectedGate = 1;
      return false;
   }
   
   double totalMargin = currMargin + reqMargin;
   if(totalMargin > 0.0 && equity > 0.0)
   {
      double projectedMarginLevel = (equity / totalMargin) * 100.0;
      
      // Dynamically query official broker account margin call and stop out levels
      double brokerCall = AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
      double brokerSO   = AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
      
      // Reference call level: fallback to 2x StopOut or baseline 100% if undefined
      double referenceCall = (brokerCall > 0.0) ? brokerCall : ((brokerSO > 0.0) ? brokerSO * 2.0 : 100.0);
      double minSafetyLevel = referenceCall * MathMax(InpMarginSafetyMultiplier, 1.0);
      
      if(projectedMarginLevel < minSafetyLevel)
      {
         outRejectReason = StringFormat("Projected Margin Level %.2f%% below safety threshold %.2f%% (Broker Call: %.0f%% * %.2fx)",
                                        projectedMarginLevel, minSafetyLevel, referenceCall, InpMarginSafetyMultiplier);
         outRejectedGate = 1;
         return false;
      }
   }
   
   // 2. Gate 2: Asymmetric Risk-Reward Cap (SL vs TP)
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point > 0.0)
   {
      double slDistPoints = MathAbs(openPrice - slPrice) / point;
      double tpDistPoints = MathAbs(tpPrice - openPrice) / point;
      
      if(tpDistPoints > 0.0)
      {
         double asymmetryRatio = slDistPoints / tpDistPoints;
         if(InpMaxRiskRewardRatio > 0.0 && asymmetryRatio > InpMaxRiskRewardRatio)
         {
            outRejectReason = StringFormat("Asymmetry ratio %.2f (SL: %.0f pts, TP: %.0f pts) exceeds max ratio %.2f",
                                           asymmetryRatio, slDistPoints, tpDistPoints, InpMaxRiskRewardRatio);
            outRejectedGate = 2;
            return false;
         }
      }
   }
   
   // 3. Gate 3: Maximum Trade Loss Budget (% of Equity)
   if(InpMaxTradeRiskPct > 0.0)
   {
      double potentialLoss = 0.0;
      if(!OrderCalcProfit(orderType, symbol, lot, openPrice, slPrice, potentialLoss))
      {
         outRejectReason = "Broker failed to calculate trade profit/loss for symbol (OrderCalcProfit failed)";
         outRejectedGate = 3;
         return false;
      }
      
      double lossAbs = MathAbs(potentialLoss);
      double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
      if(equity > 0.0)
      {
         double lossPct = (lossAbs / equity) * 100.0;
         if(lossPct > InpMaxTradeRiskPct)
         {
            outRejectReason = StringFormat("Estimated loss %.2f (%.2f%% of Equity %.2f) exceeds max budget of %.2f%%",
                                           lossAbs, lossPct, equity, InpMaxTradeRiskPct);
            outRejectedGate = 3;
            return false;
         }
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| NormalizeLotSize: Quantizes lot to broker steps & bounds         |
//+------------------------------------------------------------------+
double NormalizeLotSize(const string symbol, double rawLot)
{
   double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(stepLot <= 0.0) stepLot = 0.01;
   
   if(rawLot < minLot)
      return 0.0;
   if(rawLot > maxLot)
      rawLot = maxLot;
   
   double steps = MathFloor((rawLot / stepLot) + 1e-7);
   double quantized = steps * stepLot;
   
   int digits = 2;
   if(stepLot == 0.1) digits = 1;
   else if(stepLot >= 1.0) digits = 0;
   
   return NormalizeDouble(quantized, digits);
}

//+------------------------------------------------------------------+
//| CalculateViableLotSize: Analytically downsizes lot from maxLot   |
//| until it fits both risk budget (% equity) and margin cushion.    |
//+------------------------------------------------------------------+
double CalculateViableLotSize(const ENUM_ORDER_TYPE orderType, const string symbol,
                              const double openPrice, const double slPrice,
                              const double maxStartingLot, string &outLogDetails)
{
   outLogDetails = "";
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   if(minLot <= 0.0) minLot = 0.01;
   
   double allowedLot = (maxStartingLot > 0.0) ? maxStartingLot : minLot;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity <= 0.0)
   {
      outLogDetails = "Account equity is zero or negative";
      return 0.0;
   }
   
   // 1. Risk Budget Constraint (% of Equity)
   if(InpMaxTradeRiskPct > 0.0)
   {
      double unitLoss = 0.0;
      if(OrderCalcProfit(orderType, symbol, 1.0, openPrice, slPrice, unitLoss))
      {
         double absUnitLoss = MathAbs(unitLoss);
         if(absUnitLoss > 0.0)
         {
            double maxAllowedLossEUR = equity * (InpMaxTradeRiskPct / 100.0);
            double maxLotByRisk = maxAllowedLossEUR / absUnitLoss;
            if(maxLotByRisk < allowedLot)
               allowedLot = maxLotByRisk;
         }
      }
   }
   
   // 2. Margin & Leverage Constraint
   double unitMargin = 0.0;
   if(OrderCalcMargin(orderType, symbol, 1.0, openPrice, unitMargin) && unitMargin > 0.0)
   {
      double currMargin = AccountInfoDouble(ACCOUNT_MARGIN);
      double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      
      double brokerCall = AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL);
      double brokerSO   = AccountInfoDouble(ACCOUNT_MARGIN_SO_SO);
      double referenceCall = (brokerCall > 0.0) ? brokerCall : ((brokerSO > 0.0) ? brokerSO * 2.0 : 100.0);
      double minSafetyLevel = referenceCall * MathMax(InpMarginSafetyMultiplier, 1.0);
      
      // Margin capacity: max total margin allowed before ML drops below minSafetyLevel
      double maxTotalMargin = (minSafetyLevel > 0.0) ? (equity * 100.0 / minSafetyLevel) : equity;
      double marginRoom = maxTotalMargin - currMargin;
      if(marginRoom < 0.0) marginRoom = 0.0;
      
      double usableMargin = MathMin(freeMargin, marginRoom);
      double maxLotByMargin = usableMargin / unitMargin;
      if(maxLotByMargin < allowedLot)
         allowedLot = maxLotByMargin;
   }
   
   // Quantize and check bounds
   double finalLot = NormalizeLotSize(symbol, allowedLot);
   if(finalLot < minLot)
   {
      outLogDetails = StringFormat("Even minimum lot %.2f exceeds limits (Allowed: %.4f, Min: %.2f)",
                                   minLot, allowedLot, minLot);
      return 0.0;
   }
   
   outLogDetails = StringFormat("Fitted lot %.2f (Starting max: %.2f, RiskPct: %.1f%%)",
                                finalLot, maxStartingLot, InpMaxTradeRiskPct);
   return finalLot;
}

//+------------------------------------------------------------------+
//| Initialize SQLite Macroeconomic Governance Database              |
//+------------------------------------------------------------------+
bool InitMacroDatabase()
{
   if(!InpEnableCalendarFilter && !InpEnableNewsFilter)
      return true;
      
   // Open or create SQLite database in MT5 Common Files directory using static MACRO_DATABASE_NAME
   g_hMacroDB = DatabaseOpen(MACRO_DATABASE_NAME, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE | DATABASE_OPEN_COMMON);
   if(g_hMacroDB == INVALID_HANDLE)
   {
      PrintFormat("[LiveONNX-EA] [WARNING] Could not open SQLite database '%s' in Common/Files (Error: %d). Macro governance disabled.",
                  MACRO_DATABASE_NAME, GetLastError());
      return false;
   }
   
   // High-concurrency WAL configuration to eliminate multi-chart lock contention
   DatabaseExecute(g_hMacroDB, "PRAGMA journal_mode = WAL;");
   DatabaseExecute(g_hMacroDB, "PRAGMA synchronous = NORMAL;");
   DatabaseExecute(g_hMacroDB, "PRAGMA busy_timeout = 5000;");
   
   // Create calendar_events table and index if not exists
   string createCalSQL = "CREATE TABLE IF NOT EXISTS calendar_events ("
                         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "symbol TEXT NOT NULL, "
                         "title TEXT NOT NULL, "
                         "description TEXT NOT NULL, "
                         "start_time TEXT NOT NULL, "
                         "end_time TEXT NOT NULL, "
                         "action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES', "
                         "trailing_points INTEGER NOT NULL DEFAULT 0);";
   if(!DatabaseExecute(g_hMacroDB, createCalSQL))
   {
      PrintFormat("[LiveONNX-EA] [WARNING] Failed to execute DDL for calendar_events (Error: %d)", GetLastError());
   }
   
   // Defensive schema migration: ensure trailing_points column exists in legacy tables
   DatabaseExecute(g_hMacroDB, "ALTER TABLE calendar_events ADD COLUMN trailing_points INTEGER NOT NULL DEFAULT 0;");
   
   string createIndexSQL = "CREATE INDEX IF NOT EXISTS idx_cal_lookup ON calendar_events (symbol, start_time, end_time);";
   DatabaseExecute(g_hMacroDB, createIndexSQL);
   
   // Create news_events table if not exists
   string createNewsSQL = "CREATE TABLE IF NOT EXISTS news_events ("
                          "symbol TEXT PRIMARY KEY, "
                          "title TEXT NOT NULL, "
                          "description TEXT NOT NULL, "
                          "action TEXT NOT NULL DEFAULT 'BLOCK_ENTRIES', "
                          "trailing_points INTEGER NOT NULL DEFAULT 0);";
   if(!DatabaseExecute(g_hMacroDB, createNewsSQL))
   {
      PrintFormat("[LiveONNX-EA] [WARNING] Failed to execute DDL for news_events (Error: %d)", GetLastError());
   }
   
   // Defensive schema migration: ensure trailing_points column exists in legacy news table
   DatabaseExecute(g_hMacroDB, "ALTER TABLE news_events ADD COLUMN trailing_points INTEGER NOT NULL DEFAULT 0;");
   
   PrintFormat("[LiveONNX-EA] SQLite Macro Governance Engine connected to '%s' (Common/Files). Calendar: %s, News: %s.",
               MACRO_DATABASE_NAME, InpEnableCalendarFilter ? "ON" : "OFF", InpEnableNewsFilter ? "ON" : "OFF");
   return true;
}

//+------------------------------------------------------------------+
//| Close SQLite Macroeconomic Governance Database                   |
//+------------------------------------------------------------------+
void CloseMacroDatabase()
{
   if(g_hMacroDB != INVALID_HANDLE)
   {
      DatabaseClose(g_hMacroDB);
      g_hMacroDB = INVALID_HANDLE;
      Print("[LiveONNX-EA] SQLite Macro Governance Database connection closed.");
   }
}

//+------------------------------------------------------------------+
//| CheckMacroNews: Queries active breaking news blacklist in SQLite |
//| (Live only; skipped during Strategy Tester backtesting)          |
//| Cached with a 15-second TTL to eliminate redundant IPC I/O       |
//+------------------------------------------------------------------+
bool CheckMacroNews(const string symbol, string &outTitle, string &outDesc, string &outAction, int &outTrailingPoints)
{
   outTitle = "";
   outDesc = "";
   outAction = "";
   outTrailingPoints = 0;
   
   if(!InpEnableNewsFilter || g_hMacroDB == INVALID_HANDLE)
      return false;
      
   // Bypassed during backtesting per architecture specifications
   if(MQLInfoInteger(MQL_TESTER))
      return false;
      
   datetime now = TimeCurrent();
   // Check 15-second cache
   if(g_macroCache.lastNewsCheckTime > 0 && (now - g_macroCache.lastNewsCheckTime) < 15)
   {
      if(g_macroCache.hasNewsEvent)
      {
         outTitle = g_macroCache.newsTitle;
         outDesc = g_macroCache.newsDesc;
         outAction = g_macroCache.newsAction;
         outTrailingPoints = g_macroCache.newsTrailingPoints;
         return true;
      }
      return false;
   }
   
   g_macroCache.lastNewsCheckTime = now;
   g_macroCache.hasNewsEvent = false;
   
   string baseCurr  = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 0, 3) : "";
   string quoteCurr = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 3, 3) : "";
   string cleanPair = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 0, 6) : symbol;
   
   string query = StringFormat("SELECT title, description, action, trailing_points FROM news_events "
                               "WHERE symbol='%s' OR symbol='%s' OR symbol='%s' OR symbol='%s' OR symbol='GLOBAL' LIMIT 1;",
                               symbol, cleanPair, baseCurr, quoteCurr);
   int hQuery = DatabasePrepare(g_hMacroDB, query);
   if(hQuery == INVALID_HANDLE)
      return false;
      
   bool found = false;
   if(DatabaseRead(hQuery))
   {
      DatabaseColumnText(hQuery, 0, outTitle);
      DatabaseColumnText(hQuery, 1, outDesc);
      DatabaseColumnText(hQuery, 2, outAction);
      DatabaseColumnInteger(hQuery, 3, outTrailingPoints);
      found = true;
      
      g_macroCache.hasNewsEvent = true;
      g_macroCache.newsTitle = outTitle;
      g_macroCache.newsDesc = outDesc;
      g_macroCache.newsAction = outAction;
      g_macroCache.newsTrailingPoints = outTrailingPoints;
   }
   DatabaseFinalize(hQuery);
   return found;
}

//+------------------------------------------------------------------+
//| CheckMacroCalendar: Queries active scheduled events in SQLite    |
//| (Active in both Live Trading and Strategy Tester backtests)       |
//| Cached per bar timestamp to eliminate intra-bar SQLite queries   |
//+------------------------------------------------------------------+
bool CheckMacroCalendar(const string symbol, const datetime barTime,
                        string &outTitle, string &outDesc, string &outAction, int &outTrailingPoints)
{
   outTitle = "";
   outDesc = "";
   outAction = "";
   outTrailingPoints = 0;
   
   if(!InpEnableCalendarFilter || g_hMacroDB == INVALID_HANDLE)
      return false;
      
   // Check bar-timestamp cache
   if(g_macroCache.lastCheckBarTime == barTime)
   {
      if(g_macroCache.hasCalendarEvent)
      {
         outTitle = g_macroCache.calTitle;
         outDesc = g_macroCache.calDesc;
         outAction = g_macroCache.calAction;
         outTrailingPoints = g_macroCache.calTrailingPoints;
         return true;
      }
      return false;
   }
   
   g_macroCache.lastCheckBarTime = barTime;
   g_macroCache.hasCalendarEvent = false;
   
   // Format MT5 Server Time (EET/EEST) to match database event timestamps
   string timeStr = TimeToString(barTime, TIME_DATE | TIME_SECONDS);
   StringReplace(timeStr, ".", "-");
   
   string baseCurr  = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 0, 3) : "";
   string quoteCurr = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 3, 3) : "";
   string cleanPair = (StringLen(symbol) >= 6) ? StringSubstr(symbol, 0, 6) : symbol;
   
   string query = StringFormat("SELECT title, description, action, trailing_points FROM calendar_events "
                               "WHERE (symbol='%s' OR symbol='%s' OR symbol='%s' OR symbol='%s' OR symbol='GLOBAL') "
                               "AND '%s' >= start_time AND '%s' <= end_time "
                               "LIMIT 1;", symbol, cleanPair, baseCurr, quoteCurr, timeStr, timeStr);
                               
   int hQuery = DatabasePrepare(g_hMacroDB, query);
   if(hQuery == INVALID_HANDLE)
      return false;
      
   bool found = false;
   if(DatabaseRead(hQuery))
   {
      DatabaseColumnText(hQuery, 0, outTitle);
      DatabaseColumnText(hQuery, 1, outDesc);
      DatabaseColumnText(hQuery, 2, outAction);
      DatabaseColumnInteger(hQuery, 3, outTrailingPoints);
      found = true;
      
      g_macroCache.hasCalendarEvent = true;
      g_macroCache.calTitle = outTitle;
      g_macroCache.calDesc = outDesc;
      g_macroCache.calAction = outAction;
      g_macroCache.calTrailingPoints = outTrailingPoints;
   }
   DatabaseFinalize(hQuery);
   return found;
}

//+------------------------------------------------------------------+
//| ApplyMacroAction: Executes position protection actions on open   |
//| orders (TRAILING_STOP, BREAKEVEN, CLOSE_ALL)                     |
//+------------------------------------------------------------------+
void ApplyMacroAction(const string symbol, const string action, const int trailingPoints = 0)
{
   if(action == "BLOCK_ENTRIES" || action == "ADVISORY_ONLY" || action == "NONE" || action == "")
      return;
      
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      if(point <= 0.0) point = _Point;
      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      long stopLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      long spread    = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
      double minStopDist = (double)(stopLevel + spread + 5) * point;
      
      if(action == "CLOSE_ALL")
      {
         if(!g_trade.PositionClose(ticket))
         {
            PrintFormat("[LiveONNX-EA] [ERROR] Failed to close position #%I64u for %s (Retcode: %u, Desc: %s).",
                        ticket, symbol, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
         }
         else
         {
            PrintFormat("[LiveONNX-EA] [MACRO ACTION: CLOSE_ALL] Closed position #%I64u for %s ahead of macro event.",
                        ticket, symbol);
         }
      }
      else if(action == "BREAKEVEN")
      {
         // Move SL to openPrice if position is currently in profit
         if(posType == POSITION_TYPE_BUY)
         {
            double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
            if(bid > openPrice && (currentSL < openPrice || currentSL == 0.0))
            {
               if((bid - openPrice) >= minStopDist)
               {
                  if(!g_trade.PositionModify(ticket, NormalizeDouble(openPrice, digits), currentTP))
                  {
                     PrintFormat("[LiveONNX-EA] [WARNING] Failed to move BUY #%I64u SL to breakeven (Retcode: %u, Desc: %s). Closing position immediately for safety.",
                                 ticket, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
                     g_trade.PositionClose(ticket);
                  }
                  else
                  {
                     PrintFormat("[LiveONNX-EA] [MACRO ACTION: BREAKEVEN] Moved BUY #%I64u SL to entry %.5f.", ticket, openPrice);
                  }
               }
               else
               {
                  // Distance to openPrice violates broker stop level, fallback to closing position immediately
                  PrintFormat("[LiveONNX-EA] [WARNING] Breakeven distance (%.1f pts) below min stop distance (%.1f pts) for BUY #%I64u. Closing position immediately for safety.",
                              (bid - openPrice) / point, minStopDist / point, ticket);
                  g_trade.PositionClose(ticket);
               }
            }
         }
         else if(posType == POSITION_TYPE_SELL)
         {
            double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
            if(ask < openPrice && (currentSL > openPrice || currentSL == 0.0))
            {
               if((openPrice - ask) >= minStopDist)
               {
                  if(!g_trade.PositionModify(ticket, NormalizeDouble(openPrice, digits), currentTP))
                  {
                     PrintFormat("[LiveONNX-EA] [WARNING] Failed to move SELL #%I64u SL to breakeven (Retcode: %u, Desc: %s). Closing position immediately for safety.",
                                 ticket, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
                     g_trade.PositionClose(ticket);
                  }
                  else
                  {
                     PrintFormat("[LiveONNX-EA] [MACRO ACTION: BREAKEVEN] Moved SELL #%I64u SL to entry %.5f.", ticket, openPrice);
                  }
               }
               else
               {
                  PrintFormat("[LiveONNX-EA] [WARNING] Breakeven distance (%.1f pts) below min stop distance (%.1f pts) for SELL #%I64u. Closing position immediately for safety.",
                              (openPrice - ask) / point, minStopDist / point, ticket);
                  g_trade.PositionClose(ticket);
               }
            }
         }
      }
      else if(action == "TRAILING_STOP")
      {
         if(trailingPoints <= 0)
         {
            // User Rule: If trailing_points is 0 or unset, execute immediate position closure
            PrintFormat("[LiveONNX-EA] [MACRO ACTION: TRAILING_STOP -> CLOSE] Trailing points is %d (0/unset). Closing position #%I64u immediately for safety.",
                        trailingPoints, ticket);
            if(!g_trade.PositionClose(ticket))
            {
               PrintFormat("[LiveONNX-EA] [ERROR] Failed to emergency close position #%I64u on unset trailing points (Retcode: %u, Desc: %s).",
                           ticket, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
            }
         }
         else
         {
            double trailingDist = (double)trailingPoints * point;
            
            if(posType == POSITION_TYPE_BUY)
            {
               double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
               if(bid - openPrice > trailingDist)
               {
                  double newSL = NormalizeDouble(bid - trailingDist, digits);
                  if(newSL > currentSL && (bid - newSL) >= minStopDist)
                  {
                     if(!g_trade.PositionModify(ticket, newSL, currentTP))
                     {
                        PrintFormat("[LiveONNX-EA] [WARNING] Failed to trail BUY #%I64u SL to %.5f (Retcode: %u, Desc: %s). Closing position immediately for safety.",
                                    ticket, newSL, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
                        g_trade.PositionClose(ticket);
                     }
                     else
                     {
                        PrintFormat("[LiveONNX-EA] [MACRO ACTION: TRAILING_STOP] Trailed BUY #%I64u SL to %.5f.", ticket, newSL);
                     }
                  }
               }
            }
            else if(posType == POSITION_TYPE_SELL)
            {
               double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
               if(openPrice - ask > trailingDist)
               {
                  double newSL = NormalizeDouble(ask + trailingDist, digits);
                  if((newSL < currentSL || currentSL == 0.0) && (newSL - ask) >= minStopDist)
                  {
                     if(!g_trade.PositionModify(ticket, newSL, currentTP))
                     {
                        PrintFormat("[LiveONNX-EA] [WARNING] Failed to trail SELL #%I64u SL to %.5f (Retcode: %u, Desc: %s). Closing position immediately for safety.",
                                    ticket, newSL, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
                        g_trade.PositionClose(ticket);
                     }
                     else
                     {
                        PrintFormat("[LiveONNX-EA] [MACRO ACTION: TRAILING_STOP] Trailed SELL #%I64u SL to %.5f.", ticket, newSL);
                     }
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   PrintFormat("[LiveONNX-EA] Initializing Live ONNX Trading EA on %s (%s)...", _Symbol, EnumToString(_Period));
   
   // 1. Exhaustive Parameter Validation (Early Exit with INIT_PARAMETERS_INCORRECT)
   string validationError = "";
   if(!ValidateInputParameters(validationError))
   {
      PrintFormat("[LiveONNX-EA] [ERROR] Parameter validation failed: %s", validationError);
      return INIT_PARAMETERS_INCORRECT;
   }
   
   // 2. Pre-calculate Daily Schedule Arrays (O(1) lookups during execution)
   g_daySchedules[0].isEnabled    = InpTradeMonday;
   g_daySchedules[0].startSeconds = TimeStringToSeconds(InpMondayStartTime);
   g_daySchedules[0].endSeconds   = TimeStringToSeconds(InpMondayEndTime);
   
   g_daySchedules[1].isEnabled    = InpTradeTuesday;
   g_daySchedules[1].startSeconds = TimeStringToSeconds(InpTuesdayStartTime);
   g_daySchedules[1].endSeconds   = TimeStringToSeconds(InpTuesdayEndTime);
   
   g_daySchedules[2].isEnabled    = InpTradeWednesday;
   g_daySchedules[2].startSeconds = TimeStringToSeconds(InpWednesdayStartTime);
   g_daySchedules[2].endSeconds   = TimeStringToSeconds(InpWednesdayEndTime);
   
   g_daySchedules[3].isEnabled    = InpTradeThursday;
   g_daySchedules[3].startSeconds = TimeStringToSeconds(InpThursdayStartTime);
   g_daySchedules[3].endSeconds   = TimeStringToSeconds(InpThursdayEndTime);
   
   g_daySchedules[4].isEnabled    = InpTradeFriday;
   g_daySchedules[4].startSeconds = TimeStringToSeconds(InpFridayStartTime);
   g_daySchedules[4].endSeconds   = TimeStringToSeconds(InpFridayEndTime);
   
   // 3. Reset in-memory Macroeconomic Event Cache
   g_macroCache.lastCheckBarTime   = 0;
   g_macroCache.hasCalendarEvent   = false;
   g_macroCache.calTitle           = "";
   g_macroCache.calDesc            = "";
   g_macroCache.calAction          = "";
   g_macroCache.calTrailingPoints  = 0;
   
   g_macroCache.lastNewsCheckTime  = 0;
   g_macroCache.hasNewsEvent       = false;
   g_macroCache.newsTitle          = "";
   g_macroCache.newsDesc           = "";
   g_macroCache.newsAction         = "";
   g_macroCache.newsTrailingPoints = 0;
   
   string tfName = EnumToString(_Period);
   StringReplace(tfName, "PERIOD_", "");
   
   // 4. Setup Feature Configuration from Native Inputs
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
   
   // 2. Initialize GARCH Dynamic Risk Engine (Execution Sizing)
   g_garch.SetParameters(g_config.priceSize, InpRiskGarchHorizon, InpGarchAlpha, InpGarchBeta);
   
   // 3. Initialize Feature Extractor
   if(!g_featureExtractor.Init(_Symbol, _Period, g_config))
   {
      Print("[LiveONNX-EA] Failed to initialize Feature Extractor!");
      return INIT_FAILED;
   }
   
   g_featureCount = g_featureExtractor.GetTotalVectorSize();
   
   // 4. Load ONNX BUY Model if direction allows
   if(InpTradeDirection == DIRECTION_BOTH || InpTradeDirection == DIRECTION_ONLY_BUY)
   {
      g_hModelBuy = LoadModelWithFallback(InpModelBuyPath, "buy");
      if(g_hModelBuy == INVALID_HANDLE)
      {
         PrintFormat("[LiveONNX-EA] ERROR: Could not find BUY ONNX model for %s %s. Expected 'Models/%s_%s_model_buy.onnx'. Error: %d",
                     _Symbol, tfName, _Symbol, tfName, GetLastError());
         return INIT_FAILED;
      }
   }
   
   // 5. Load ONNX SELL Model if direction allows
   if(InpTradeDirection == DIRECTION_BOTH || InpTradeDirection == DIRECTION_ONLY_SELL)
   {
      g_hModelSell = LoadModelWithFallback(InpModelSellPath, "sell");
      if(g_hModelSell == INVALID_HANDLE)
      {
         PrintFormat("[LiveONNX-EA] ERROR: Could not find SELL ONNX model for %s %s. Expected 'Models/%s_%s_model_sell.onnx'. Error: %d",
                     _Symbol, tfName, _Symbol, tfName, GetLastError());
         if(g_hModelBuy != INVALID_HANDLE)
         {
            OnnxRelease(g_hModelBuy);
            g_hModelBuy = INVALID_HANDLE;
         }
         return INIT_FAILED;
      }
   }
   
   // 6. Explicitly define 1D tensor shapes for sub-millisecond execution
   const ulong inputShape[]  = {1, (ulong)g_featureCount};
   const ulong outputShape[] = {1, 2};
   
   if(g_hModelBuy != INVALID_HANDLE)
   {
      if(!OnnxSetInputShape(g_hModelBuy, 0, inputShape))
      {
         PrintFormat("[LiveONNX-EA] ERROR: BUY model input tensor shape mismatch! Extracted features: %d. Error: %d",
                     g_featureCount, GetLastError());
         return INIT_FAILED;
      }
      if(!OnnxSetOutputShape(g_hModelBuy, 0, outputShape))
      {
         PrintFormat("[LiveONNX-EA] Failed to set BUY model output shape! Error: %d", GetLastError());
         return INIT_FAILED;
      }
   }
   
   if(g_hModelSell != INVALID_HANDLE)
   {
      if(!OnnxSetInputShape(g_hModelSell, 0, inputShape))
      {
         PrintFormat("[LiveONNX-EA] ERROR: SELL model input tensor shape mismatch! Extracted features: %d. Error: %d",
                     g_featureCount, GetLastError());
         return INIT_FAILED;
      }
      if(!OnnxSetOutputShape(g_hModelSell, 0, outputShape))
      {
         PrintFormat("[LiveONNX-EA] Failed to set SELL model output shape! Error: %d", GetLastError());
         return INIT_FAILED;
      }
   }
   
   // 7. Setup CTrade Parameters with Adaptive Filling Mode
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(GetOptimalFillingType(_Symbol));
   
   // 8. Initialize SQLite Macroeconomic Governance Engine
   InitMacroDatabase();
   
   // 9. Initialize Consecutive Signal & Position Manager
   SConsecutiveConfig consecConfig;
   consecConfig.mode                          = InpConsecutiveMode;
   consecConfig.maxConsecutiveOrders          = InpMaxConsecutiveOrders;
   consecConfig.hurdleProfitPct               = InpHurdleProfitPct;
   consecConfig.profitLockPct                 = InpProfitLockPct;
   consecConfig.antiChopMinDisplacementPoints = InpAntiChopMinDisplacement;
   consecConfig.safetyOffsetPoints            = InpSafetyOffsetPoints;
   consecConfig.enableSwapAmortization        = InpEnableSwapAmortization;
   consecConfig.consecutiveSlotFilter         = InpConsecutiveSlotFilter;
   consecConfig.ignoreConflictingSignals      = InpIgnoreConflictingSignals;
   consecConfig.enableOpposingRegimeFilter    = InpEnableOpposingRegimeFilter;
   consecConfig.opposingStreakThreshold       = InpOpposingStreakThreshold;
   consecConfig.opposingAction                = InpOpposingAction;
   consecConfig.opposingTrailingPoints        = InpOpposingTrailingPoints;
   consecConfig.opposingRecalculateRatio      = InpOpposingRecalculateRatio;
   g_consecutiveManager.Init(_Symbol, InpMagicNumber, consecConfig);
   
   PrintFormat("[LiveONNX-EA] Live Inference Engine Ready. Tensor Dimensions: %d, Direction: %d, Threshold BUY: %.4f, Threshold SELL: %.4f, ConsecutiveMode: %d (MaxOrders: %d, Hurdle: %.1f%%, Lock: %.1f%%, AntiChop: %d pts, SwapAmort: %s, SlotFilter: %s), S&R Snapping: %s (Lookback: %d bars, Pivot K: %d, Offset: %d pts, Mode: %s), Risk Filter: %s (Max SL/TP: %.2f, Max Risk: %.1f%%, Margin Multiplier: %.2fx, DynLot: %s MaxLot: %.2f), Calendar Filter: %s, News Filter: %s, Dynamic Risk GARCH (Horizon: %d, kTP: %.2f, kSL: %.2f), Schedule: Mon=%d Tue=%d Wed=%d Thu=%d Fri=%d, Audit: %s",
               g_featureCount, (int)InpTradeDirection, InpMinimalLevelAcceptedBuy, InpMinimalLevelAcceptedSell,
               (int)InpConsecutiveMode, InpMaxConsecutiveOrders, InpHurdleProfitPct, InpProfitLockPct, InpAntiChopMinDisplacement,
               InpEnableSwapAmortization ? "ON" : "OFF", InpConsecutiveSlotFilter ? "ON" : "OFF",
               InpEnableSRSnapping ? "ON" : "OFF", InpSRLookbackBars, InpSRPivotStrength, InpSROffsetPoints, EnumToString(InpSRZoneSelection),
               InpEnableRiskFilter ? "ON" : "OFF", InpMaxRiskRewardRatio, InpMaxTradeRiskPct, InpMarginSafetyMultiplier,
               InpEnableDynamicLotSizing ? "ON" : "OFF", InpMaxLotSize,
               InpEnableCalendarFilter ? "ON" : "OFF", InpEnableNewsFilter ? "ON" : "OFF",
               InpRiskGarchHorizon, InpKTP, InpKSL,
               (int)InpTradeMonday, (int)InpTradeTuesday, (int)InpTradeWednesday, (int)InpTradeThursday, (int)InpTradeFriday,
               InpIgnoreAudit ? "IGNORED" : "ACTIVE");

   // 10. Initialize SQLite Execution & Telemetry Audit Logging Engine (Optional via InpIgnoreAudit)
   if(!InpIgnoreAudit)
   {
      g_auditor.Init(_Symbol, _Period);
   }
   else
   {
      Print("[LiveONNX-EA] [INFO] Audit logging bypassed by user parameter (InpIgnoreAudit=true). SQLite DB creation suppressed.");
   }

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   PrintFormat("[LiveONNX-EA] Deinitializing (Reason: %d)...", reason);
   
   // Close SQLite Macroeconomic Governance Database
   CloseMacroDatabase();
   
   // Close SQLite Prediction Audit Logging Database
   g_auditor.Close();
   
   // Release ONNX runtime handles
   if(g_hModelBuy != INVALID_HANDLE)
   {
      OnnxRelease(g_hModelBuy);
      g_hModelBuy = INVALID_HANDLE;
   }
   if(g_hModelSell != INVALID_HANDLE)
   {
      OnnxRelease(g_hModelSell);
      g_hModelSell = INVALID_HANDLE;
   }
   
   g_featureExtractor.ReleaseHandles();
   ArrayFree(g_activeTrades);
   Print("[LiveONNX-EA] Deinitialization complete.");
}

//+------------------------------------------------------------------+
//| Expert tick function: Evaluates ONNX inference on new bar        |
//+------------------------------------------------------------------+
void OnTick()
{
   // Continuous excursion tracking across all active positions
   UpdateActiveTradesExcursion();

   // Execute inference only on new bar open
   if(!IsNewBar()) return;
   
   datetime barTime = iTime(_Symbol, _Period, 0);
   double ask        = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid        = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   long   spread     = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   
   SCandleTelemetryRecord auditRec;
   auditRec.Reset();
   auditRec.barTime              = barTime;
   auditRec.ask                  = ask;
   auditRec.bid                  = bid;
   auditRec.spread               = spread;
   auditRec.thresholdBuy         = InpMinimalLevelAcceptedBuy;
   auditRec.thresholdSell        = InpMinimalLevelAcceptedSell;
   auditRec.consecutiveMode      = (int)InpConsecutiveMode;
   auditRec.activePositionsCount = PositionsTotal();
   auditRec.floatingProfit       = AccountInfoDouble(ACCOUNT_PROFIT);
   auditRec.accountEquity        = AccountInfoDouble(ACCOUNT_EQUITY);
   auditRec.accountBalance       = AccountInfoDouble(ACCOUNT_BALANCE);
   auditRec.accountMarginLevel   = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   auditRec.accountFreeMargin    = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   
   // 1. Build identical flattened feature vector for base timestamp
   vectorf inputVector;
   bool featOk = g_featureExtractor.ExtractFlattenedVector(0, inputVector);
   float probBuy  = 0.0f;
   float probSell = 0.0f;
   ulong latencyUs = 0;
   
   if(featOk)
   {
      ulong startUs = GetMicrosecondCount();
      if(g_hModelBuy != INVALID_HANDLE)
      {
         vectorf outBuy(2);
         if(OnnxRun(g_hModelBuy, ONNX_NO_CONVERSION, inputVector, outBuy))
         {
            probBuy = outBuy[1];
         }
         else
         {
            PrintFormat("[LiveONNX-EA] OnnxRun BUY failed! Error: %d", GetLastError());
            g_auditor.LogEvent(AUDIT_SEV_ERROR, "ONNX_MODEL", GetLastError(), "OnnxRun BUY failed");
         }
      }
      
      if(g_hModelSell != INVALID_HANDLE)
      {
         vectorf outSell(2);
         if(OnnxRun(g_hModelSell, ONNX_NO_CONVERSION, inputVector, outSell))
         {
            probSell = outSell[1];
         }
         else
         {
            PrintFormat("[LiveONNX-EA] OnnxRun SELL failed! Error: %d", GetLastError());
            g_auditor.LogEvent(AUDIT_SEV_ERROR, "ONNX_MODEL", GetLastError(), "OnnxRun SELL failed");
         }
      }
      latencyUs = GetMicrosecondCount() - startUs;
   }
   else
   {
      Print("[LiveONNX-EA] Failed to extract feature vector!");
      g_auditor.LogEvent(AUDIT_SEV_ERROR, "FEATURE_EXTRACTOR", GetLastError(), "Failed to extract feature vector");
   }
   
   auditRec.probBuy            = probBuy;
   auditRec.probSell           = probSell;
   auditRec.rawBuySignal       = (probBuy >= (float)InpMinimalLevelAcceptedBuy);
   auditRec.rawSellSignal      = (probSell >= (float)InpMinimalLevelAcceptedSell);
   auditRec.inferenceLatencyUs = latencyUs;
   auditRec.convictionDelta    = MathAbs((double)probBuy - (double)probSell);
   auditRec.probEntropy        = (CalculateShannonEntropy((double)probBuy) + CalculateShannonEntropy((double)probSell)) / 2.0;
   auditRec.conflictingSignals = (auditRec.rawBuySignal && auditRec.rawSellSignal);
   
   PrintFormat("[LiveONNX-EA] Inference => Prob BUY: %.4f (Thresh: %.4f), Prob SELL: %.4f (Thresh: %.4f) [Latency: %I64u us, Entropy: %.4f, Conviction: %.4f]",
               probBuy, InpMinimalLevelAcceptedBuy, probSell, InpMinimalLevelAcceptedSell, latencyUs, auditRec.probEntropy, auditRec.convictionDelta);

   // Check daily schedule filter (MT5 Server Time)
   bool scheduleAllowed = IsTradeScheduleAllowed(barTime);
   auditRec.scheduleAllowed = scheduleAllowed;
   if(!scheduleAllowed)
   {
      auditRec.executionAction = "BLOCKED_SCHEDULE";
      g_auditor.RecordCandleTelemetry(auditRec);
      return;
   }
   
   // Check Macroeconomic News Blacklist (Live Only, bypassed in Strategy Tester)
   string newsTitle = "", newsDesc = "", newsAction = "";
   int newsTrailingPoints = 0;
   if(CheckMacroNews(_Symbol, newsTitle, newsDesc, newsAction, newsTrailingPoints))
   {
      ApplyMacroAction(_Symbol, newsAction, newsTrailingPoints);
      auditRec.macroNewsBlocked = (newsAction != "ADVISORY_ONLY");
      auditRec.macroAction      = newsAction;
      if(newsAction != "ADVISORY_ONLY")
      {
         PrintFormat("[LiveONNX-EA] [GLOBAL NEWS BLACKLIST] Order blocked for %s! Headline: '%s' (Action: %s) | Reason: %s. Skipping bar.",
                     _Symbol, newsTitle, newsAction, newsDesc);
         g_auditor.LogEvent(AUDIT_SEV_INFO, "MACRO_NEWS", 0, "Trade blocked by news: " + newsTitle, newsAction);
         auditRec.executionAction = "BLOCKED_NEWS";
         g_auditor.RecordCandleTelemetry(auditRec);
         return;
      }
      else
      {
         PrintFormat("[LiveONNX-EA] [GLOBAL NEWS ADVISORY] Active advisory for %s: '%s' | Note: %s. Non-blocking.",
                     _Symbol, newsTitle, newsDesc);
      }
   }
   
   // Check Scheduled Macroeconomic Calendar Events (Live & Strategy Tester)
   string calTitle = "", calDesc = "", calAction = "";
   int calTrailingPoints = 0;
   if(CheckMacroCalendar(_Symbol, barTime, calTitle, calDesc, calAction, calTrailingPoints))
   {
      ApplyMacroAction(_Symbol, calAction, calTrailingPoints);
      auditRec.macroCalendarBlocked = (calAction != "ADVISORY_ONLY");
      if(auditRec.macroAction == "NONE" || auditRec.macroAction == "")
         auditRec.macroAction = calAction;
      if(calAction != "ADVISORY_ONLY")
      {
         PrintFormat("[LiveONNX-EA] [MACRO CALENDAR BLOCK] Order blocked for %s: Event '%s' is active! (Action: %s) | Reason: %s. Skipping bar.",
                     _Symbol, calTitle, calAction, calDesc);
         g_auditor.LogEvent(AUDIT_SEV_INFO, "MACRO_CALENDAR", 0, "Trade blocked by calendar: " + calTitle, calAction);
         auditRec.executionAction = "BLOCKED_CALENDAR";
         g_auditor.RecordCandleTelemetry(auditRec);
         return;
      }
      else
      {
         PrintFormat("[LiveONNX-EA] [MACRO CALENDAR ADVISORY] Event '%s' active for %s: %s. Non-blocking.",
                     calTitle, _Symbol, calDesc);
      }
   }
   
   if(!featOk)
   {
      auditRec.executionAction = "ERROR_FEATURE_EXTRACTION";
      g_auditor.RecordCandleTelemetry(auditRec);
      return;
   }
   
   // 3. Determine Dynamic GARCH TP/SL Points & Volatility Metrics
   double gOmega = 0.0, gVolRatio = 1.0, gVolTrend = 1.0, gSigmaCond = 0.0, gSigmaAgg = 0.0;
   if(g_garch.ComputeGarchMetrics(_Symbol, _Period, 0, gOmega, gVolRatio, gVolTrend, gSigmaCond, gSigmaAgg))
   {
      auditRec.garchSigmaCond = gSigmaCond;
      auditRec.garchVolRatio  = gVolRatio;
   }
   
   double tpPoints = 0.0, slPoints = 0.0, sigmaAgg = 0.0;
   if(!g_garch.CalculateDynamicRisk(_Symbol, _Period, InpKTP, InpKSL, tpPoints, slPoints, sigmaAgg))
   {
      Print("[LiveONNX-EA] GARCH dynamic risk calculation failed!");
      auditRec.executionAction = "ERROR_GARCH_CALCULATION";
      g_auditor.LogEvent(AUDIT_SEV_ERROR, "GARCH_ENGINE", 0, "GARCH dynamic risk calculation failed");
      g_auditor.RecordCandleTelemetry(auditRec);
      return;
   }
   
   auditRec.garchSigmaAgg = sigmaAgg;
   auditRec.garchTpPoints = tpPoints;
   auditRec.garchSlPoints = slPoints;
   
   double point      = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0) point = _Point;
   int digits        = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   long stopsLevel   = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   
   double minStopPoints = (double)(stopsLevel + spread + 5);
   double slDist     = MathMax(slPoints * point, minStopPoints * point);
   double tpDist     = MathMax(tpPoints * point, minStopPoints * point);
   
   // 4. Raw Signal Generation & Conflicting Signals Suppression
   bool buySignalRaw  = auditRec.rawBuySignal;
   bool sellSignalRaw = auditRec.rawSellSignal;

   if(InpIgnoreConflictingSignals && buySignalRaw && sellSignalRaw)
   {
      PrintFormat("[LiveONNX-EA] [CONFLICTING SIGNALS] Both BUY (%.4f) and SELL (%.4f) met thresholds on same bar. Suppressing new trade entries.",
                  probBuy, probSell);
      g_auditor.LogEvent(AUDIT_SEV_INFO, "SIGNAL_CONFLICT", 0, "Both BUY and SELL met thresholds; suppressed new entries", StringFormat("Buy:%.4f, Sell:%.4f", probBuy, probSell));
      buySignalRaw  = false;
      sellSignalRaw = false;
   }

   bool allowBuy  = (InpTradeDirection == DIRECTION_BOTH || InpTradeDirection == DIRECTION_ONLY_BUY);
   bool allowSell = (InpTradeDirection == DIRECTION_BOTH || InpTradeDirection == DIRECTION_ONLY_SELL);

   bool buyCondition  = allowBuy && buySignalRaw;
   bool sellCondition = allowSell && sellSignalRaw;

   // 5. Pre-calculate Candidate Levels & Dynamic Lot Sizes for Signals / Opposing Actions
   double buySL = 0.0, buyTP = 0.0, buyLot = InpLotSize;
   bool buySROk = false, buyViable = false;
   if(buySignalRaw)
   {
      buySL = NormalizeDouble(bid - slDist, digits);
      buyTP = NormalizeDouble(ask + tpDist, digits);
      if(InpEnableSRSnapping)
      {
         buySROk = ApplyStructuralSRSnapping(_Symbol, _Period, true, ask, bid,
                                             InpSRLookbackBars, InpSROffsetPoints,
                                             InpSRZoneSelection, InpSRPivotStrength,
                                             buySL, buyTP, minStopPoints, buySL, buyTP);
      }
      buyViable = true;
      if(InpEnableRiskFilter && InpEnableDynamicLotSizing)
      {
         string sizingLog = "";
         buyLot = CalculateViableLotSize(ORDER_TYPE_BUY, _Symbol, ask, buySL, InpMaxLotSize, sizingLog);
         if(buyLot <= 0.0)
         {
            PrintFormat("[LiveONNX-EA] [RISK FILTER] Order BUY rejected for %s: %s. Skipping BUY.", _Symbol, sizingLog);
            g_auditor.LogEvent(AUDIT_SEV_WARNING, "DYNAMIC_LOT", 0, "BUY lot sizing failed: " + sizingLog);
            buyViable = false;
         }
         else if(buyLot < InpMaxLotSize)
         {
            PrintFormat("[LiveONNX-EA] [DYNAMIC LOT] Adapted BUY volume from %.2f to %.2f (%s)", InpMaxLotSize, buyLot, sizingLog);
         }
      }
      if(buyViable && InpEnableRiskFilter)
      {
         string rejectReason = "";
         int rejectedGate = 0;
         if(!CheckTradeViability(ORDER_TYPE_BUY, _Symbol, buyLot, ask, buySL, buyTP, rejectReason, rejectedGate))
         {
            PrintFormat("[LiveONNX-EA] [RISK FILTER] Order BUY rejected for %s (Gate %d): %s. Skipping BUY.", _Symbol, rejectedGate, rejectReason);
            g_auditor.LogEvent(AUDIT_SEV_WARNING, "RISK_GATE", rejectedGate, "BUY order rejected: " + rejectReason);
            buyViable = false;
            auditRec.rejectedGateId = rejectedGate;
            auditRec.executionAction = StringFormat("BUY_REJECTED_GATE_%d", rejectedGate);
         }
      }
      auditRec.finalSlPrice = buySL;
      auditRec.finalTpPrice = buyTP;
      auditRec.srSnapped = buySROk;
      auditRec.srZoneType = buySROk ? "SWING_PIVOT" : "NONE";
      auditRec.dynamicLot = buyLot;
      auditRec.riskFilterPassed = buyViable;
   }

   double sellSL = 0.0, sellTP = 0.0, sellLot = InpLotSize;
   bool sellSROk = false, sellViable = false;
   if(sellSignalRaw)
   {
      sellSL = NormalizeDouble(ask + slDist, digits);
      sellTP = NormalizeDouble(bid - tpDist, digits);
      if(InpEnableSRSnapping)
      {
         sellSROk = ApplyStructuralSRSnapping(_Symbol, _Period, false, ask, bid,
                                              InpSRLookbackBars, InpSROffsetPoints,
                                              InpSRZoneSelection, InpSRPivotStrength,
                                              sellSL, sellTP, minStopPoints, sellSL, sellTP);
      }
      sellViable = true;
      if(InpEnableRiskFilter && InpEnableDynamicLotSizing)
      {
         string sizingLog = "";
         sellLot = CalculateViableLotSize(ORDER_TYPE_SELL, _Symbol, bid, sellSL, InpMaxLotSize, sizingLog);
         if(sellLot <= 0.0)
         {
            PrintFormat("[LiveONNX-EA] [RISK FILTER] Order SELL rejected for %s: %s. Skipping SELL.", _Symbol, sizingLog);
            g_auditor.LogEvent(AUDIT_SEV_WARNING, "DYNAMIC_LOT", 0, "SELL lot sizing failed: " + sizingLog);
            sellViable = false;
         }
         else if(sellLot < InpMaxLotSize)
         {
            PrintFormat("[LiveONNX-EA] [DYNAMIC LOT] Adapted SELL volume from %.2f to %.2f (%s)", InpMaxLotSize, sellLot, sizingLog);
         }
      }
      if(sellViable && InpEnableRiskFilter)
      {
         string rejectReason = "";
         int rejectedGate = 0;
         if(!CheckTradeViability(ORDER_TYPE_SELL, _Symbol, sellLot, bid, sellSL, sellTP, rejectReason, rejectedGate))
         {
            PrintFormat("[LiveONNX-EA] [RISK FILTER] Order SELL rejected for %s (Gate %d): %s. Skipping SELL.", _Symbol, rejectedGate, rejectReason);
            g_auditor.LogEvent(AUDIT_SEV_WARNING, "RISK_GATE", rejectedGate, "SELL order rejected: " + rejectReason);
            sellViable = false;
            auditRec.rejectedGateId = rejectedGate;
            auditRec.executionAction = StringFormat("SELL_REJECTED_GATE_%d", rejectedGate);
         }
      }
      auditRec.finalSlPrice = sellSL;
      auditRec.finalTpPrice = sellTP;
      auditRec.srSnapped = sellSROk;
      auditRec.srZoneType = sellSROk ? "SWING_PIVOT" : "NONE";
      auditRec.dynamicLot = sellLot;
      auditRec.riskFilterPassed = sellViable;
   }

   // 6. Opposing Regime Defense Filter (Check active positions facing adverse predictions)
   if(InpEnableOpposingRegimeFilter)
   {
      // Defend BUY positions against adverse SELL signals
      bool buyDefended = g_consecutiveManager.CheckAndProcessOpposingRegime(
         g_trade, POSITION_TYPE_BUY, sellSignalRaw, ask, bid, digits, stopsLevel, spread,
         sellSL, sellTP, sellLot, probSell, sellSROk
      );
      if(buyDefended && InpOpposingAction == OPPOSING_ACTION_STOP_AND_REVERSE)
      {
         sellCondition = false; // Reversed order already executed
         auditRec.executionAction = "OPPOSING_REVERSE_BUY_TO_SELL";
         auditRec.consecutiveAction = "STOP_AND_REVERSE";
         g_auditor.LogEvent(AUDIT_SEV_INFO, "OPPOSING_DEFENSE", (int)InpOpposingAction, "Opposing defense reversed BUY to SELL");
      }

      // Defend SELL positions against adverse BUY signals
      bool sellDefended = g_consecutiveManager.CheckAndProcessOpposingRegime(
         g_trade, POSITION_TYPE_SELL, buySignalRaw, ask, bid, digits, stopsLevel, spread,
         buySL, buyTP, buyLot, probBuy, buySROk
      );
      if(sellDefended && InpOpposingAction == OPPOSING_ACTION_STOP_AND_REVERSE)
      {
         buyCondition = false; // Reversed order already executed
         auditRec.executionAction = "OPPOSING_REVERSE_SELL_TO_BUY";
         auditRec.consecutiveAction = "STOP_AND_REVERSE";
         g_auditor.LogEvent(AUDIT_SEV_INFO, "OPPOSING_DEFENSE", (int)InpOpposingAction, "Opposing defense reversed SELL to BUY");
      }
   }

   // 7. BUY Execution: Execution at Ask, closes at Bid -> SL below Bid, TP above Ask
   if(buyCondition && buyViable)
   {
      ulong tStartBuy = GetMicrosecondCount();
      bool buyExecuted = g_consecutiveManager.ExecuteBuy(g_trade, _Period, ask, bid, buySL, buyTP, buyLot, probBuy, buySROk, digits, stopsLevel, spread);
      ulong buyLatencyMs = (GetMicrosecondCount() - tStartBuy) / 1000;
      
      if(!buyExecuted)
      {
         uint retcode = g_trade.ResultRetcode();
         auditRec.executionAction = "BUY_FAILED";
         auditRec.executionRetcode = retcode;
         ENUM_AUDIT_SEVERITY sev = (retcode == TRADE_RETCODE_MARKET_CLOSED ||
                                    retcode == TRADE_RETCODE_OFFQUOTES ||
                                    retcode == TRADE_RETCODE_PRICE_OFF ||
                                    retcode == TRADE_RETCODE_TRADE_DISABLED ||
                                    retcode == TRADE_RETCODE_INVALID_STOPS) ? AUDIT_SEV_WARNING : AUDIT_SEV_ERROR;
         g_auditor.LogEvent(sev, "BROKER_EXEC", (int)retcode, "BUY Order rejected by broker", g_trade.ResultRetcodeDescription());
         
         if(sev == AUDIT_SEV_WARNING)
         {
            PrintFormat("[LiveONNX-EA] [WARNING] Order %s rejected for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s). Skipping bar.",
                        "BUY", _Symbol, digits, ask, digits, bid, digits, buySL, digits, buyTP,
                        (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription());
         }
         else if(retcode > 0)
         {
            PrintFormat("[LiveONNX-EA] [ERROR] Order %s failed for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s, Deal: %I64u, Order: %I64u, LastError: %d)",
                        "BUY", _Symbol, digits, ask, digits, bid, digits, buySL, digits, buyTP,
                        (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription(), g_trade.ResultDeal(), g_trade.ResultOrder(), GetLastError());
         }
      }
      else
      {
         auditRec.executionAction = "BUY_EXECUTED";
         auditRec.executionRetcode = g_trade.ResultRetcode();
         ulong dealTicket = g_trade.ResultDeal();
         ulong orderTicket = g_trade.ResultOrder();
         auditRec.executionTicket = (dealTicket > 0) ? dealTicket : orderTicket;
         
         double fillPrice = ask;
         if(dealTicket > 0 && HistoryDealSelect(dealTicket))
         {
            double dPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
            if(dPrice > 0.0) fillPrice = dPrice;
         }
         if(point <= 0.0) point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         if(point <= 0.0) point = _Point;
         double slippagePts = (point > 0.0) ? (fillPrice - ask) / point : 0.0;
         ulong posId = 0;
         if(dealTicket > 0 && HistoryDealSelect(dealTicket))
            posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
         if(posId == 0) posId = auditRec.executionTicket;
         
         RegisterActiveTrade(posId, dealTicket, TimeCurrent(), "BUY", buyLot, ask, fillPrice, slippagePts, buyLatencyMs, buyTP, buySL);
         
         PrintFormat("[LiveONNX-EA] Executed BUY condition (Mode: %d, Lot: %.2f, Prob: %.4f, TP: %.*f, SL: %.*f, Fill: %.*f, Slippage: %.1f pts, Latency: %I64u ms, Snapping: %s)",
                     (int)InpConsecutiveMode, buyLot, probBuy, digits, buyTP, digits, buySL, digits, fillPrice, slippagePts, buyLatencyMs, buySROk ? "Real_SR" : "Classic_GARCH");
      }
   }

   // 8. SELL Execution: Execution at Bid, closes at Ask -> SL above Ask, TP below Bid
   if(sellCondition && sellViable)
   {
      ulong tStartSell = GetMicrosecondCount();
      bool sellExecuted = g_consecutiveManager.ExecuteSell(g_trade, _Period, ask, bid, sellSL, sellTP, sellLot, probSell, sellSROk, digits, stopsLevel, spread);
      ulong sellLatencyMs = (GetMicrosecondCount() - tStartSell) / 1000;
      
      if(!sellExecuted)
      {
         uint retcode = g_trade.ResultRetcode();
         auditRec.executionAction = "SELL_FAILED";
         auditRec.executionRetcode = retcode;
         ENUM_AUDIT_SEVERITY sev = (retcode == TRADE_RETCODE_MARKET_CLOSED ||
                                    retcode == TRADE_RETCODE_OFFQUOTES ||
                                    retcode == TRADE_RETCODE_PRICE_OFF ||
                                    retcode == TRADE_RETCODE_TRADE_DISABLED ||
                                    retcode == TRADE_RETCODE_INVALID_STOPS) ? AUDIT_SEV_WARNING : AUDIT_SEV_ERROR;
         g_auditor.LogEvent(sev, "BROKER_EXEC", (int)retcode, "SELL Order rejected by broker", g_trade.ResultRetcodeDescription());
         
         if(sev == AUDIT_SEV_WARNING)
         {
            PrintFormat("[LiveONNX-EA] [WARNING] Order %s rejected for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s). Skipping bar.",
                        "SELL", _Symbol, digits, ask, digits, bid, digits, sellSL, digits, sellTP,
                        (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription());
         }
         else if(retcode > 0)
         {
            PrintFormat("[LiveONNX-EA] [ERROR] Order %s failed for %s (Ask: %.*f, Bid: %.*f, SL: %.*f, TP: %.*f, StopsLevel: %d, Spread: %d, Retcode: %u, Desc: %s, Deal: %I64u, Order: %I64u, LastError: %d)",
                        "SELL", _Symbol, digits, ask, digits, bid, digits, sellSL, digits, sellTP,
                        (int)stopsLevel, (int)spread, retcode, g_trade.ResultRetcodeDescription(), g_trade.ResultDeal(), g_trade.ResultOrder(), GetLastError());
         }
      }
      else
      {
         auditRec.executionAction = "SELL_EXECUTED";
         auditRec.executionRetcode = g_trade.ResultRetcode();
         ulong dealTicket = g_trade.ResultDeal();
         ulong orderTicket = g_trade.ResultOrder();
         auditRec.executionTicket = (dealTicket > 0) ? dealTicket : orderTicket;
         
         double fillPrice = bid;
         if(dealTicket > 0 && HistoryDealSelect(dealTicket))
         {
            double dPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
            if(dPrice > 0.0) fillPrice = dPrice;
         }
         if(point <= 0.0) point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         if(point <= 0.0) point = _Point;
         double slippagePts = (point > 0.0) ? (bid - fillPrice) / point : 0.0;
         ulong posId = 0;
         if(dealTicket > 0 && HistoryDealSelect(dealTicket))
            posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
         if(posId == 0) posId = auditRec.executionTicket;
         
         RegisterActiveTrade(posId, dealTicket, TimeCurrent(), "SELL", sellLot, bid, fillPrice, slippagePts, sellLatencyMs, sellTP, sellSL);
         
         PrintFormat("[LiveONNX-EA] Executed SELL condition (Mode: %d, Lot: %.2f, Prob: %.4f, TP: %.*f, SL: %.*f, Fill: %.*f, Slippage: %.1f pts, Latency: %I64u ms, Snapping: %s)",
                     (int)InpConsecutiveMode, sellLot, probSell, digits, sellTP, digits, sellSL, digits, fillPrice, slippagePts, sellLatencyMs, sellSROk ? "Real_SR" : "Classic_GARCH");
      }
   }

   // Fallback action state if no trade or error occurred
   if(auditRec.executionAction == "NONE")
   {
      auditRec.executionAction = "NO_SIGNAL";
   }

   // Persist full candle inference & decision snapshot to SQLite
   g_auditor.RecordCandleTelemetry(auditRec);
}

//+------------------------------------------------------------------+
//| Trade Transaction handler: Intercepts position exits             |
//| and records full trade lifecycle attribution to SQLite           |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong dealTicket = trans.deal;
      if(dealTicket > 0 && HistoryDealSelect(dealTicket))
      {
         ENUM_DEAL_ENTRY dealEntry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
         if(dealEntry == DEAL_ENTRY_OUT || dealEntry == DEAL_ENTRY_OUT_BY)
         {
            ulong posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
            if(posId == 0)
               posId = trans.position;
            if(posId == 0)
               posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_ORDER);
            if(posId == 0)
               posId = dealTicket;

            ENUM_DEAL_REASON dealReason = (ENUM_DEAL_REASON)HistoryDealGetInteger(dealTicket, DEAL_REASON);
            double closePrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
            datetime closeTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
            double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
            double swap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
            double commission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
            double netProfit = profit + swap + commission;
            double dealVolume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
            
            // Check for partial close (position remains open with residual volume)
            bool isPartialClose = (posId > 0 && PositionSelectByTicket(posId));
            
            string exitReasonStr = "OTHER";
            if(dealReason == DEAL_REASON_TP) exitReasonStr = "TP";
            else if(dealReason == DEAL_REASON_SL) exitReasonStr = "SL";
            else if(dealReason == DEAL_REASON_EXPERT) exitReasonStr = "EXPERT_OR_DEFENSE";
            else if(dealReason == DEAL_REASON_SO) exitReasonStr = "STOP_OUT";
            else if(dealReason == DEAL_REASON_CLIENT) exitReasonStr = "MANUAL";
            
            if(isPartialClose)
               exitReasonStr = exitReasonStr + "_PARTIAL";
            
            STradeLifecycleRecord rec;
            rec.Reset();
            rec.positionId = posId;
            rec.exitDealTicket = dealTicket;
            rec.symbol = _Symbol;
            rec.timeframe = StringSubstr(EnumToString(_Period), 7);
            rec.closeTime = closeTime;
            rec.actualClosePrice = closePrice;
            rec.exitReason = exitReasonStr;
            rec.grossProfit = profit;
            rec.swapCharges = swap;
            rec.commissionCharges = commission;
            rec.netLiquidProfit = netProfit;
            rec.volume = dealVolume;
            
            int idx = FindActiveTrade(posId);
            if(idx >= 0)
            {
               rec.entryDealTicket     = g_activeTrades[idx].entryDealTicket;
               rec.openTime            = g_activeTrades[idx].openTime;
               rec.holdingDurationSec  = (long)(closeTime - g_activeTrades[idx].openTime);
               rec.holdingBars         = (int)(rec.holdingDurationSec / MathMax(PeriodSeconds(_Period), 60));
               rec.orderType           = g_activeTrades[idx].orderType;
               rec.targetEntryPrice    = g_activeTrades[idx].targetEntryPrice;
               rec.actualEntryPrice    = g_activeTrades[idx].actualEntryPrice;
               rec.entrySlippagePoints = g_activeTrades[idx].entrySlippagePoints;
               rec.orderLatencyMs      = g_activeTrades[idx].orderLatencyMs;
               
               double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
               if(point <= 0.0) point = _Point;
               if(point > 0.0)
               {
                  if(rec.orderType == "BUY")
                  {
                     rec.maxFavorablePoints = MathMax(0.0, (g_activeTrades[idx].maxFavorablePrice - rec.actualEntryPrice) / point);
                     rec.maxAdversePoints   = MathMax(0.0, (rec.actualEntryPrice - g_activeTrades[idx].maxAdversePrice) / point);
                  }
                  else
                  {
                     rec.maxFavorablePoints = MathMax(0.0, (rec.actualEntryPrice - g_activeTrades[idx].maxFavorablePrice) / point);
                     rec.maxAdversePoints   = MathMax(0.0, (g_activeTrades[idx].maxAdversePrice - rec.actualEntryPrice) / point);
                  }
               }
               
               if(isPartialClose)
               {
                  g_activeTrades[idx].volume = MathMax(0.0, g_activeTrades[idx].volume - dealVolume);
               }
               else
               {
                  RemoveActiveTrade(idx);
               }
            }
            else
            {
               // Fallback: recover entry deal parameters directly from terminal deal history
               if(posId > 0 && HistorySelectByPosition(posId))
               {
                  int dealsCount = HistoryDealsTotal();
                  for(int d = 0; d < dealsCount; d++)
                  {
                     ulong dt = HistoryDealGetTicket(d);
                     if(dt > 0 && (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dt, DEAL_ENTRY) == DEAL_ENTRY_IN)
                     {
                        rec.entryDealTicket    = dt;
                        rec.openTime           = (datetime)HistoryDealGetInteger(dt, DEAL_TIME);
                        rec.actualEntryPrice   = HistoryDealGetDouble(dt, DEAL_PRICE);
                        rec.targetEntryPrice   = rec.actualEntryPrice;
                        rec.orderType          = (HistoryDealGetInteger(dt, DEAL_TYPE) == DEAL_TYPE_BUY) ? "BUY" : "SELL";
                        rec.holdingDurationSec = (long)(closeTime - rec.openTime);
                        rec.holdingBars        = (int)(rec.holdingDurationSec / MathMax(PeriodSeconds(_Period), 60));
                        break;
                     }
                  }
               }
               
               if(rec.openTime == 0)
               {
                  rec.openTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
                  rec.orderType = (HistoryDealGetInteger(dealTicket, DEAL_TYPE) == DEAL_TYPE_BUY) ? "SELL" : "BUY";
                  rec.holdingDurationSec = 0;
                  rec.holdingBars = 0;
               }
            }
            
            g_auditor.RecordTradeExit(rec);
            PrintFormat("[LiveONNX-EA] [TRADE CLOSED] Position #%I64u closed (%s) | Net Profit: %.2f (Gross: %.2f, Swap: %.2f, Comm: %.2f) | Duration: %d bars (%d sec)",
                        posId, exitReasonStr, netProfit, profit, swap, commission, rec.holdingBars, (int)rec.holdingDurationSec);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Expert Tester function (Optimization Custom Fitness Criterion)   |
//+------------------------------------------------------------------+
double OnTester()
{
   double netProfit    = TesterStatistics(STAT_PROFIT);
   double maxEquityDD  = TesterStatistics(STAT_EQUITY_DD);
   double profitFactor = TesterStatistics(STAT_PROFIT_FACTOR);
   long   dealsTotal   = (long)TesterStatistics(STAT_DEALS);
   
   // If no trades were opened (e.g. impossible probability threshold) or test ended in net loss
   if(dealsTotal <= 1 || netProfit <= 0.0 || maxEquityDD <= 0.0)
   {
      return 0.0;
   }
   
   // Recovery ratio (Net Profit / Max Equity Drawdown) weighted by Profit Factor
   double recoveryRatio = netProfit / maxEquityDD;
   double clampedPF     = MathMin(profitFactor, 5.0);
   double customScore   = recoveryRatio * clampedPF;
   
   return customScore;
}


