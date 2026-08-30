//+------------------------------------------------------------------+
//|                                             ExecutionAuditor.mqh |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "2.00"

#ifndef EXECUTION_AUDITOR_MQH
#define EXECUTION_AUDITOR_MQH

//+------------------------------------------------------------------+
//| ENUM_AUDIT_SEVERITY                                              |
//| Diagnostic severity classification for system event incidents    |
//+------------------------------------------------------------------+
enum ENUM_AUDIT_SEVERITY
{
   AUDIT_SEV_INFO     = 0, // Informational operational milestone
   AUDIT_SEV_WARNING  = 1, // Non-fatal operational anomaly (e.g. offquotes, transient spread)
   AUDIT_SEV_ERROR    = 2, // Unexpected execution or calculation error (e.g. invalid stops)
   AUDIT_SEV_CRITICAL = 3  // Fatal subsystem failure (e.g. ONNX handle failure, DB corruption)
};

//+------------------------------------------------------------------+
//| STRUCT SCandleTelemetryRecord                                    |
//| Pillar 1: Candle-by-candle snapshot of features, probabilities,  |
//| entropy, volatility, filters, gates, and trade dispatch states   |
//+------------------------------------------------------------------+
struct SCandleTelemetryRecord
{
   datetime barTime;
   double   ask;
   double   bid;
   long     spread;
   float    probBuy;
   float    probSell;
   double   thresholdBuy;
   double   thresholdSell;
   double   convictionDelta;     // |probBuy - probSell|
   double   probEntropy;         // Shannon Entropy: -(p*log2(p) + (1-p)*log2(1-p))
   bool     conflictingSignals;  // True if both buy & sell meet threshold
   bool     rawBuySignal;
   bool     rawSellSignal;
   bool     scheduleAllowed;
   bool     macroCalendarBlocked;
   bool     macroNewsBlocked;
   string   macroAction;
   double   garchSigmaCond;
   double   garchSigmaAgg;
   double   garchVolRatio;       // sigma_cond / sqrt(s^2)
   double   garchTpPoints;
   double   garchSlPoints;
   bool     srSnapped;
   double   finalTpPrice;
   double   finalSlPrice;
   string   srZoneType;          // "SWING_HIGH", "SWING_LOW", "NONE"
   bool     riskFilterPassed;
   int      rejectedGateId;      // 0=None, 1=Margin, 2=Asymmetry, 3=EquityRisk
   double   accountEquity;
   double   accountBalance;
   double   accountMarginLevel;  // Equity / Margin * 100.0
   double   accountFreeMargin;
   double   dynamicLot;
   int      consecutiveMode;
   string   consecutiveAction;
   int      activePositionsCount;
   double   floatingProfit;
   string   executionAction;
   uint     executionRetcode;
   ulong    executionTicket;
   ulong    inferenceLatencyUs;

   void Reset()
   {
      barTime              = 0;
      ask                  = 0.0;
      bid                  = 0.0;
      spread               = 0;
      probBuy              = 0.0f;
      probSell             = 0.0f;
      thresholdBuy         = 0.0;
      thresholdSell        = 0.0;
      convictionDelta      = 0.0;
      probEntropy          = 0.0;
      conflictingSignals   = false;
      rawBuySignal         = false;
      rawSellSignal        = false;
      scheduleAllowed      = true;
      macroCalendarBlocked = false;
      macroNewsBlocked     = false;
      macroAction          = "NONE";
      garchSigmaCond       = 0.0;
      garchSigmaAgg        = 0.0;
      garchVolRatio        = 1.0;
      garchTpPoints        = 0.0;
      garchSlPoints        = 0.0;
      srSnapped            = false;
      finalTpPrice         = 0.0;
      finalSlPrice         = 0.0;
      srZoneType           = "NONE";
      riskFilterPassed     = true;
      rejectedGateId       = 0;
      accountEquity        = 0.0;
      accountBalance       = 0.0;
      accountMarginLevel   = 0.0;
      accountFreeMargin    = 0.0;
      dynamicLot           = 0.0;
      consecutiveMode      = 0;
      consecutiveAction    = "NONE";
      activePositionsCount = 0;
      floatingProfit       = 0.0;
      executionAction      = "NONE";
      executionRetcode     = 0;
      executionTicket      = 0;
      inferenceLatencyUs   = 0;
   }
};

//+------------------------------------------------------------------+
//| STRUCT STradeLifecycleRecord                                     |
//| Pillar 3: Complete lifecycle of a closed position (entry to exit)|
//+------------------------------------------------------------------+
struct STradeLifecycleRecord
{
   ulong    positionId;
   ulong    entryDealTicket;
   ulong    exitDealTicket;
   string   symbol;
   string   timeframe;
   string   orderType;           // "BUY" or "SELL"
   double   volume;
   datetime openTime;
   datetime closeTime;
   long     holdingDurationSec;
   int      holdingBars;
   double   targetEntryPrice;    // Quote Ask/Bid at inference time
   double   actualEntryPrice;    // Execution fill price
   double   entrySlippagePoints; // (actual - target) in broker points
   ulong    orderLatencyMs;      // Roundtrip execution latency
   double   actualClosePrice;    // Close execution price
   string   exitReason;          // "TP", "SL", "TRAILING_STOP", "OPPOSING_DEFENSE", "MACRO_EMERGENCY", "MANUAL"
   double   grossProfit;         // DEAL_PROFIT
   double   swapCharges;         // DEAL_SWAP
   double   commissionCharges;   // DEAL_COMMISSION
   double   netLiquidProfit;     // Gross + Swap + Commission
   double   maxFavorablePoints;  // MFE in points
   double   maxAdversePoints;    // MAE in points

   void Reset()
   {
      positionId          = 0;
      entryDealTicket     = 0;
      exitDealTicket      = 0;
      symbol              = "";
      timeframe           = "";
      orderType           = "NONE";
      volume              = 0.0;
      openTime            = 0;
      closeTime           = 0;
      holdingDurationSec  = 0;
      holdingBars         = 0;
      targetEntryPrice    = 0.0;
      actualEntryPrice    = 0.0;
      entrySlippagePoints = 0.0;
      orderLatencyMs      = 0;
      actualClosePrice    = 0.0;
      exitReason          = "NONE";
      grossProfit         = 0.0;
      swapCharges         = 0.0;
      commissionCharges   = 0.0;
      netLiquidProfit     = 0.0;
      maxFavorablePoints  = 0.0;
      maxAdversePoints    = 0.0;
   }
};

//+------------------------------------------------------------------+
//| CLASS CExecutionAuditor                                          |
//| Institutional 3-Pillar SQLite Telemetry & Audit Engine           |
//+------------------------------------------------------------------+
class CExecutionAuditor
{
private:
   int      m_hDB;
   string   m_dbPath;
   string   m_symbol;
   string   m_timeframe;
   bool     m_initialized;

   string FormatDateTime(const datetime dt)
   {
      MqlDateTime mdt;
      TimeToStruct(dt, mdt);
      return StringFormat("%04d-%02d-%02d %02d:%02d:%02d",
                          mdt.year, mdt.mon, mdt.day, mdt.hour, mdt.min, mdt.sec);
   }

   string SeverityToString(const ENUM_AUDIT_SEVERITY sev)
   {
      switch(sev)
      {
         case AUDIT_SEV_INFO:     return "INFO";
         case AUDIT_SEV_WARNING:  return "WARNING";
         case AUDIT_SEV_ERROR:    return "ERROR";
         case AUDIT_SEV_CRITICAL: return "CRITICAL";
         default:                 return "UNKNOWN";
      }
   }

   string SanitizeSQL(const string rawText)
   {
      string output = rawText;
      StringReplace(output, "'", "''");
      return output;
   }

public:
   CExecutionAuditor() : m_hDB(INVALID_HANDLE), m_dbPath(""), m_symbol(""), m_timeframe(""), m_initialized(false)
   {
   }

   ~CExecutionAuditor()
   {
      Close();
   }

   //+---------------------------------------------------------------+
   //| Initialize SQLite Audit Database in Common/Files/AuditLogs/   |
   //+---------------------------------------------------------------+
   bool Init(const string symbol, const ENUM_TIMEFRAMES period)
   {
      Close();
      m_symbol    = symbol;
      m_timeframe = StringSubstr(EnumToString(period), 7);
      
      const string auditFolder = "AuditLogs";
      FolderCreate(auditFolder, FILE_COMMON);
      
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      m_dbPath = StringFormat("%s\\%s_%s_%04d%02d%02d_%02d%02d%02d.db",
                              auditFolder, m_symbol, m_timeframe,
                              now.year, now.mon, now.day, now.hour, now.min, now.sec);
                              
      // If the database already exists (e.g. repeated test runs in Strategy Tester starting at same timestamp),
      // create a backup copy with .bkp and reset the active file so it starts completely empty (zerado).
      if(FileIsExist(m_dbPath, FILE_COMMON))
      {
         string bkpPath = m_dbPath + ".bkp";
         if(FileCopy(m_dbPath, FILE_COMMON, bkpPath, FILE_COMMON | FILE_REWRITE))
         {
            PrintFormat("[CExecutionAuditor] [INFO] Existing audit DB found at '%s'. Backed up to '%s'.",
                        m_dbPath, bkpPath);
         }
         else
         {
            PrintFormat("[CExecutionAuditor] [WARNING] Failed to backup existing audit DB to '%s'. Error: %d",
                        bkpPath, GetLastError());
         }
         
         // Delete old database and associated SQLite WAL/SHM files to ensure brand new empty DB
         FileDelete(m_dbPath, FILE_COMMON);
         FileDelete(m_dbPath + "-wal", FILE_COMMON);
         FileDelete(m_dbPath + "-shm", FILE_COMMON);
      }
                               
      m_hDB = DatabaseOpen(m_dbPath, DATABASE_OPEN_READWRITE | DATABASE_OPEN_CREATE | DATABASE_OPEN_COMMON);
      if(m_hDB == INVALID_HANDLE)
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to create audit SQLite database at '%s' (Common/Files). Error: %d",
                     m_dbPath, GetLastError());
         m_initialized = false;
         return false;
      }
      
      // High-concurrency WAL configuration
      DatabaseExecute(m_hDB, "PRAGMA journal_mode = WAL;");
      DatabaseExecute(m_hDB, "PRAGMA synchronous = NORMAL;");
      DatabaseExecute(m_hDB, "PRAGMA busy_timeout = 5000;");
      
      // --- TABLE 1: candle_telemetry (Continuous snapshot on every bar) ---
      string ddlCandle = 
         "CREATE TABLE IF NOT EXISTS candle_telemetry ("
         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
         "created_at TEXT NOT NULL, "
         "bar_time TEXT NOT NULL, "
         "symbol TEXT NOT NULL, "
         "timeframe TEXT NOT NULL, "
         "ask REAL NOT NULL, "
         "bid REAL NOT NULL, "
         "spread_points INTEGER NOT NULL, "
         "prob_buy REAL NOT NULL, "
         "prob_sell REAL NOT NULL, "
         "threshold_buy REAL NOT NULL, "
         "threshold_sell REAL NOT NULL, "
         "conviction_delta REAL NOT NULL, "
         "prob_entropy REAL NOT NULL, "
         "conflicting_signals INTEGER NOT NULL, "
         "raw_buy_signal INTEGER NOT NULL, "
         "raw_sell_signal INTEGER NOT NULL, "
         "schedule_allowed INTEGER NOT NULL, "
         "macro_calendar_blocked INTEGER NOT NULL, "
         "macro_news_blocked INTEGER NOT NULL, "
         "macro_action TEXT NOT NULL, "
         "garch_sigma_cond REAL NOT NULL, "
         "garch_sigma_agg REAL NOT NULL, "
         "garch_vol_ratio REAL NOT NULL, "
         "garch_tp_points REAL NOT NULL, "
         "garch_sl_points REAL NOT NULL, "
         "sr_snapped INTEGER NOT NULL, "
         "final_tp_price REAL NOT NULL, "
         "final_sl_price REAL NOT NULL, "
         "sr_zone_type TEXT NOT NULL, "
         "risk_filter_passed INTEGER NOT NULL, "
         "rejected_gate_id INTEGER NOT NULL, "
         "account_equity REAL NOT NULL, "
         "account_balance REAL NOT NULL, "
         "account_margin_level REAL NOT NULL, "
         "account_free_margin REAL NOT NULL, "
         "dynamic_lot REAL NOT NULL, "
         "consecutive_mode INTEGER NOT NULL, "
         "consecutive_action TEXT NOT NULL, "
         "active_positions_count INTEGER NOT NULL, "
         "floating_profit REAL NOT NULL, "
         "execution_action TEXT NOT NULL, "
         "execution_retcode INTEGER NOT NULL, "
         "execution_ticket INTEGER NOT NULL, "
         "inference_latency_us INTEGER NOT NULL);";

      if(!DatabaseExecute(m_hDB, ddlCandle))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to execute DDL for candle_telemetry table. Error: %d", GetLastError());
         Close();
         return false;
      }
      
      // Backward compatibility view for legacy queries
      DatabaseExecute(m_hDB, "CREATE VIEW IF NOT EXISTS prediction_audit_logs AS SELECT * FROM candle_telemetry;");

      // --- TABLE 2: system_events_log (Incidents, warnings, retcodes, and alerts) ---
      string ddlEvents = 
         "CREATE TABLE IF NOT EXISTS system_events_log ("
         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
         "created_at TEXT NOT NULL, "
         "bar_time TEXT NOT NULL, "
         "severity TEXT NOT NULL, "
         "subsystem TEXT NOT NULL, "
         "error_code INTEGER NOT NULL, "
         "event_message TEXT NOT NULL, "
         "context_data TEXT NOT NULL);";

      if(!DatabaseExecute(m_hDB, ddlEvents))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to execute DDL for system_events_log table. Error: %d", GetLastError());
         Close();
         return false;
      }

      // --- TABLE 3: trade_lifecycle_log (Detailed attribution for every closed trade) ---
      string ddlTrades = 
         "CREATE TABLE IF NOT EXISTS trade_lifecycle_log ("
         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
         "created_at TEXT NOT NULL, "
         "position_id INTEGER NOT NULL, "
         "entry_deal_ticket INTEGER NOT NULL, "
         "exit_deal_ticket INTEGER NOT NULL, "
         "symbol TEXT NOT NULL, "
         "timeframe TEXT NOT NULL, "
         "order_type TEXT NOT NULL, "
         "volume REAL NOT NULL, "
         "open_time TEXT NOT NULL, "
         "close_time TEXT NOT NULL, "
         "holding_duration_seconds INTEGER NOT NULL, "
         "holding_bars INTEGER NOT NULL, "
         "target_entry_price REAL NOT NULL, "
         "actual_entry_price REAL NOT NULL, "
         "entry_slippage_points REAL NOT NULL, "
         "order_latency_ms INTEGER NOT NULL, "
         "actual_close_price REAL NOT NULL, "
         "exit_reason TEXT NOT NULL, "
         "gross_profit REAL NOT NULL, "
         "swap_charges REAL NOT NULL, "
         "commission_charges REAL NOT NULL, "
         "net_liquid_profit REAL NOT NULL, "
         "max_favorable_points REAL NOT NULL, "
         "max_adverse_points REAL NOT NULL);";

      if(!DatabaseExecute(m_hDB, ddlTrades))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to execute DDL for trade_lifecycle_log table. Error: %d", GetLastError());
         Close();
         return false;
      }

      // High-performance indexing for rapid offline slicing
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_telemetry_bar_time ON candle_telemetry (bar_time);");
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_telemetry_action ON candle_telemetry (execution_action);");
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events_log (severity);");
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_events_subsystem ON system_events_log (subsystem);");
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_trade_position ON trade_lifecycle_log (position_id);");
      DatabaseExecute(m_hDB, "CREATE INDEX IF NOT EXISTS idx_trade_exit_reason ON trade_lifecycle_log (exit_reason);");
      
      m_initialized = true;
      PrintFormat("[CExecutionAuditor] Initialized 3-Pillar SQLite Audit Database successfully at '%s' (Common/Files).", m_dbPath);
      return true;
   }

   //+---------------------------------------------------------------+
   //| Pillar 1: Record Candle Telemetry Snapshot                    |
   //+---------------------------------------------------------------+
   bool RecordCandleTelemetry(const SCandleTelemetryRecord &rec)
   {
      if(!m_initialized || m_hDB == INVALID_HANDLE)
         return false;
         
      string createdAtStr = FormatDateTime(TimeCurrent());
      string barTimeStr   = FormatDateTime(rec.barTime);
      
      string sql = StringFormat(
         "INSERT INTO candle_telemetry ("
         "created_at, bar_time, symbol, timeframe, ask, bid, spread_points, "
         "prob_buy, prob_sell, threshold_buy, threshold_sell, conviction_delta, prob_entropy, conflicting_signals, "
         "raw_buy_signal, raw_sell_signal, schedule_allowed, macro_calendar_blocked, macro_news_blocked, macro_action, "
         "garch_sigma_cond, garch_sigma_agg, garch_vol_ratio, garch_tp_points, garch_sl_points, "
         "sr_snapped, final_tp_price, final_sl_price, sr_zone_type, risk_filter_passed, rejected_gate_id, "
         "account_equity, account_balance, account_margin_level, account_free_margin, dynamic_lot, "
         "consecutive_mode, consecutive_action, active_positions_count, floating_profit, "
         "execution_action, execution_retcode, execution_ticket, inference_latency_us"
         ") VALUES ("
         "'%s', '%s', '%s', '%s', %.5f, %.5f, %d, "
         "%.6f, %.6f, %.4f, %.4f, %.6f, %.6f, %d, "
         "%d, %d, %d, %d, %d, '%s', "
         "%.8f, %.8f, %.4f, %.2f, %.2f, "
         "%d, %.5f, %.5f, '%s', %d, %d, "
         "%.2f, %.2f, %.2f, %.2f, %.2f, "
         "%d, '%s', %d, %.2f, "
         "'%s', %u, %I64u, %I64u);",
         createdAtStr, barTimeStr, m_symbol, m_timeframe, rec.ask, rec.bid, (int)rec.spread,
         rec.probBuy, rec.probSell, rec.thresholdBuy, rec.thresholdSell, rec.convictionDelta, rec.probEntropy, rec.conflictingSignals ? 1 : 0,
         rec.rawBuySignal ? 1 : 0, rec.rawSellSignal ? 1 : 0, rec.scheduleAllowed ? 1 : 0, rec.macroCalendarBlocked ? 1 : 0, rec.macroNewsBlocked ? 1 : 0, rec.macroAction,
         rec.garchSigmaCond, rec.garchSigmaAgg, rec.garchVolRatio, rec.garchTpPoints, rec.garchSlPoints,
         rec.srSnapped ? 1 : 0, rec.finalTpPrice, rec.finalSlPrice, rec.srZoneType, rec.riskFilterPassed ? 1 : 0, rec.rejectedGateId,
         rec.accountEquity, rec.accountBalance, rec.accountMarginLevel, rec.accountFreeMargin, rec.dynamicLot,
         rec.consecutiveMode, rec.consecutiveAction, rec.activePositionsCount, rec.floatingProfit,
         rec.executionAction, rec.executionRetcode, rec.executionTicket, rec.inferenceLatencyUs
      );
      
      if(!DatabaseExecute(m_hDB, sql))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to insert candle telemetry for bar %s. Error: %d",
                     barTimeStr, GetLastError());
         return false;
      }
      return true;
   }

   //+---------------------------------------------------------------+
   //| Pillar 2: Log Asynchronous System Incidents, Warnings & Errors|
   //+---------------------------------------------------------------+
   bool LogEvent(const ENUM_AUDIT_SEVERITY severity,
                 const string subsystem,
                 const int errorCode,
                 const string message,
                 const string contextData = "")
   {
      if(!m_initialized || m_hDB == INVALID_HANDLE)
         return false;

      string createdAtStr = FormatDateTime(TimeCurrent());
      datetime currentBar = iTime(m_symbol, PERIOD_CURRENT, 0);
      string barTimeStr   = (currentBar > 0) ? FormatDateTime(currentBar) : createdAtStr;

      string safeMsg = SanitizeSQL(message);
      string safeCtx = SanitizeSQL(contextData);

      string sql = StringFormat(
         "INSERT INTO system_events_log (created_at, bar_time, severity, subsystem, error_code, event_message, context_data) "
         "VALUES ('%s', '%s', '%s', '%s', %d, '%s', '%s');",
         createdAtStr, barTimeStr, SeverityToString(severity), subsystem, errorCode, safeMsg, safeCtx
      );

      if(!DatabaseExecute(m_hDB, sql))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to insert event log. Error: %d", GetLastError());
         return false;
      }
      return true;
   }

   //+---------------------------------------------------------------+
   //| Pillar 3: Record Trade Closure & Financial Attribution        |
   //+---------------------------------------------------------------+
   bool RecordTradeExit(const STradeLifecycleRecord &rec)
   {
      if(!m_initialized || m_hDB == INVALID_HANDLE)
         return false;

      string createdAtStr = FormatDateTime(TimeCurrent());
      string openTimeStr  = FormatDateTime(rec.openTime);
      string closeTimeStr = FormatDateTime(rec.closeTime);

      string sql = StringFormat(
         "INSERT INTO trade_lifecycle_log ("
         "created_at, position_id, entry_deal_ticket, exit_deal_ticket, symbol, timeframe, order_type, volume, "
         "open_time, close_time, holding_duration_seconds, holding_bars, target_entry_price, actual_entry_price, "
         "entry_slippage_points, order_latency_ms, actual_close_price, exit_reason, gross_profit, swap_charges, "
         "commission_charges, net_liquid_profit, max_favorable_points, max_adverse_points"
         ") VALUES ("
         "'%s', %I64u, %I64u, %I64u, '%s', '%s', '%s', %.2f, "
         "'%s', '%s', %I64d, %d, %.5f, %.5f, "
         "%.2f, %I64u, %.5f, '%s', %.2f, %.2f, "
         "%.2f, %.2f, %.2f, %.2f);",
         createdAtStr, rec.positionId, rec.entryDealTicket, rec.exitDealTicket, m_symbol, m_timeframe, rec.orderType, rec.volume,
         openTimeStr, closeTimeStr, rec.holdingDurationSec, rec.holdingBars, rec.targetEntryPrice, rec.actualEntryPrice,
         rec.entrySlippagePoints, rec.orderLatencyMs, rec.actualClosePrice, rec.exitReason, rec.grossProfit, rec.swapCharges,
         rec.commissionCharges, rec.netLiquidProfit, rec.maxFavorablePoints, rec.maxAdversePoints
      );

      if(!DatabaseExecute(m_hDB, sql))
      {
         PrintFormat("[CExecutionAuditor] [ERROR] Failed to insert trade lifecycle record for ticket #%I64u. Error: %d",
                     rec.positionId, GetLastError());
         return false;
      }
      return true;
   }

   //+---------------------------------------------------------------+
   //| Close SQLite audit database connection                        |
   //+---------------------------------------------------------------+
   void Close()
   {
      if(m_hDB != INVALID_HANDLE)
      {
         DatabaseClose(m_hDB);
         m_hDB = INVALID_HANDLE;
         m_initialized = false;
         PrintFormat("[CExecutionAuditor] Closed audit database connection at '%s'.", m_dbPath);
      }
   }
   
   string GetDbPath() const { return m_dbPath; }
   bool   IsInitialized() const { return m_initialized; }
};

//+------------------------------------------------------------------+
//| Backward Compatibility Macro Aliases                             |
//+------------------------------------------------------------------+
#define CPredictionAuditor CExecutionAuditor
#define SPredictionAuditRecord SCandleTelemetryRecord

#endif // EXECUTION_AUDITOR_MQH
