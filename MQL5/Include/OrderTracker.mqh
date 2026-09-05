//+------------------------------------------------------------------+
//|                                                 OrderTracker.mqh |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| ORDER TRACKING & DATASET EXPORT ARCHITECTURE:                    |
//|                                                                  |
//| 1. In-Memory Ticket Mapping (Bypassing MT5 Comment Limits):      |
//|    MT5 limits order comments to 31 characters, which is far too  |
//|    small to encode feature vectors. OrderTracker maps active     |
//|    position tickets directly to high-dimensional float feature   |
//|    vectors in RAM during Strategy Tester execution.              |
//|                                                                  |
//| 2. Binary Outcome Labeling (OnTradeTransaction):                 |
//|    - Position closed by TP (Take Profit)  => Label: 1.0f (OPEN)  |
//|    - Position closed by SL (Stop Loss)    => Label: 0.0f (NOT_OPEN)
//|                                                                  |
//| 3. Unresolved Position Handling (OnDeinit):                      |
//|    For positions still open when Strategy Tester concludes,      |
//|    Triple Barrier vertical horizon assigns Label 0.0f (NOT_OPEN).|
//|                                                                  |
//| 4. Chronological Ordering & Dataset Export:                      |
//|    All samples are sorted chronologically by base timestamp using|
//|    an optimized index-based QuickSort algorithm, avoiding heap   |
//|    allocations, and exported to CSV without timestamp column.    |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
#include "FeatureExtractor.mqh"

//+------------------------------------------------------------------+
//| Data Structures for RAM State Storage                            |
//+------------------------------------------------------------------+
struct STrackedPosition
{
   ulong                ticket;         // Position ticket (DEAL_POSITION_ID)
   ENUM_POSITION_TYPE   posType;        // POSITION_TYPE_BUY or POSITION_TYPE_SELL
   datetime             baseTimestamp;  // Timestamp of the bar when position was registered
   double               openPrice;      // Entry price
   double               tpPrice;        // Take Profit target price
   double               slPrice;        // Stop Loss target price
   float                features[];     // Flattened feature vector
   int                  featureCount;   // Feature vector size
   bool                 isActive;       // Active tracking flag
};

//+------------------------------------------------------------------+
//| SLabeledSample: Completed historical sample with binary label    |
//+------------------------------------------------------------------+
struct SLabeledSample
{
   datetime             baseTimestamp;  // Base bar timestamp (for chronological sorting)
   ENUM_POSITION_TYPE   posType;        // BUY or SELL dataset partition
   float                label;          // 1.0f = OPEN (TP hit), 0.0f = NOT_OPEN (SL hit)
   float                features[];     // Flattened feature vector
   int                  featureCount;   // Feature count
};

//+------------------------------------------------------------------+
//| COrderTracker: In-Memory Ticket Mapper & Golden Rule Labeler     |
//+------------------------------------------------------------------+
class COrderTracker
{
private:
   string            m_symbol;
   ENUM_TIMEFRAMES   m_period;
   string            m_tfName;
   
   STrackedPosition  m_activePositions[];
   int               m_activeCount;
   
   SLabeledSample    m_recordedSamples[];
   int               m_sampleCount;
   int               m_sortIndices[];
   
public:
   //+---------------------------------------------------------------+
   //| Constructor                                                   |
   //+---------------------------------------------------------------+
   COrderTracker()
   {
      m_activeCount = 0;
      m_sampleCount = 0;
   }
   
   //+---------------------------------------------------------------+
   //| Destructor: Frees dynamically allocated memory arrays         |
   //+---------------------------------------------------------------+
   ~COrderTracker()
   {
      ClearAll();
   }
   
   //+---------------------------------------------------------------+
   //| Init: Configures symbol and timeframe for export naming       |
   //+---------------------------------------------------------------+
   void Init(const string symbol, ENUM_TIMEFRAMES period)
   {
      m_symbol = symbol;
      m_period = period;
      m_tfName = EnumToString(period);
      StringReplace(m_tfName, "PERIOD_", "");
      ClearAll();
   }
   
   //+---------------------------------------------------------------+
   //| ClearAll: Resets internal tracking collections                |
   //+---------------------------------------------------------------+
   void ClearAll()
   {
      ArrayFree(m_activePositions);
      ArrayFree(m_recordedSamples);
      ArrayFree(m_sortIndices);
      m_activeCount = 0;
      m_sampleCount = 0;
   }
   
   //+---------------------------------------------------------------+
   //| RegisterPosition: Stores new active position in memory        |
   //+---------------------------------------------------------------+
   bool RegisterPosition(ulong ticket, ENUM_POSITION_TYPE posType, datetime baseTimestamp,
                         double openPrice, double tpPrice, double slPrice,
                         const vectorf &features)
   {
      if(ticket == 0) return false;
      
      int size = ArraySize(m_activePositions);
      if(m_activeCount >= size)
      {
         ArrayResize(m_activePositions, size + 512);
      }
      
      int idx = m_activeCount++;
      m_activePositions[idx].ticket        = ticket;
      m_activePositions[idx].posType       = posType;
      m_activePositions[idx].baseTimestamp = baseTimestamp;
      m_activePositions[idx].openPrice     = openPrice;
      m_activePositions[idx].tpPrice       = tpPrice;
      m_activePositions[idx].slPrice       = slPrice;
      m_activePositions[idx].isActive      = true;
      m_activePositions[idx].featureCount  = (int)features.Size();
      
      ArrayResize(m_activePositions[idx].features, m_activePositions[idx].featureCount);
      for(int i = 0; i < m_activePositions[idx].featureCount; i++)
      {
         m_activePositions[idx].features[i] = features[i];
      }
      
      return true;
   }
   
   //+---------------------------------------------------------------+
   //| ProcessTransaction: Detects position closure and assigns label|
   //+---------------------------------------------------------------+
   void ProcessTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &req, const MqlTradeResult &res)
   {
      if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
      
      ulong dealTicket = trans.deal;
      if(dealTicket == 0) return;
      
      if(!HistoryDealSelect(dealTicket)) return;
      
      ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY) return;
      
      ulong positionId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      int posIdx = FindActivePosition(positionId);
      if(posIdx < 0) return;
      
      // Determine closure trigger: TP vs SL
      ENUM_DEAL_REASON dealReason = (ENUM_DEAL_REASON)HistoryDealGetInteger(dealTicket, DEAL_REASON);
      double closePrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
      double tpPrice    = m_activePositions[posIdx].tpPrice;
      double slPrice    = m_activePositions[posIdx].slPrice;
      ENUM_POSITION_TYPE pType = m_activePositions[posIdx].posType;
      
      // Golden Rule: Calculate net liquid profit (Profit + Swap + Commission) across all deals for positionId
      double totalProfit = 0.0;
      double totalSwap = 0.0;
      double totalCommission = 0.0;
      if(HistorySelectByPosition(positionId))
      {
         int dealTotal = HistoryDealsTotal();
         for(int d = 0; d < dealTotal; d++)
         {
            ulong dTicket = HistoryDealGetTicket(d);
            if(dTicket > 0)
            {
               totalProfit += HistoryDealGetDouble(dTicket, DEAL_PROFIT);
               totalSwap += HistoryDealGetDouble(dTicket, DEAL_SWAP);
               totalCommission += HistoryDealGetDouble(dTicket, DEAL_COMMISSION);
            }
         }
      }
      else
      {
         totalProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
         totalSwap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
         totalCommission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
      }
      double netLiquidProfit = totalProfit + totalSwap + totalCommission;
      
      float label = 0.0f; // Default: NOT_OPEN
      
      if(netLiquidProfit <= 0.0)
      {
         // Strictly NOT_OPEN if net financial outcome was negative or zero
         label = 0.0f;
      }
      else if(dealReason == DEAL_REASON_TP)
      {
         label = 1.0f; // OPEN
      }
      else if(dealReason == DEAL_REASON_SL)
      {
         label = 0.0f; // NOT_OPEN
      }
      else
      {
         // Proximity fallback check
         double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
         if(point <= 0.0) point = 0.00001;

         if(pType == POSITION_TYPE_BUY)
         {
            if(tpPrice > 0.0 && closePrice >= tpPrice - 2.0 * point && netLiquidProfit > 0.0)
               label = 1.0f;
            else
               label = 0.0f;
         }
         else if(pType == POSITION_TYPE_SELL)
         {
            if(tpPrice > 0.0 && closePrice <= tpPrice + 2.0 * point && netLiquidProfit > 0.0)
               label = 1.0f;
            else
               label = 0.0f;
         }
      }
      
      // Save labeled sample
      AddSample(m_activePositions[posIdx].baseTimestamp,
                m_activePositions[posIdx].posType,
                label,
                m_activePositions[posIdx].features,
                m_activePositions[posIdx].featureCount);
                
      // Mark inactive
      m_activePositions[posIdx].isActive = false;
   }
   
   //+---------------------------------------------------------------+
   //| CheckTimeouts: Closes active positions exceeding horizon bars |
   //+---------------------------------------------------------------+
   void CheckTimeouts(int maxBars, CTrade &trade)
   {
      if(maxBars <= 0) return;
      
      for(int i = 0; i < m_activeCount; i++)
      {
         if(!m_activePositions[i].isActive) continue;
         
         ulong ticket = m_activePositions[i].ticket;
         int shift = iBarShift(m_symbol, m_period, m_activePositions[i].baseTimestamp, true);
         if(shift >= maxBars)
         {
            if(PositionSelectByTicket(ticket))
            {
               trade.PositionClose(ticket);
            }
            else
            {
               m_activePositions[i].isActive = false;
            }
         }
      }
   }
   
   //+---------------------------------------------------------------+
   //| ProcessUnresolvedPositions: Applies deinitialization logic    |
   //| Positions still open at test end are labeled NOT_OPEN (0.0f). |
   //+---------------------------------------------------------------+
   void ProcessUnresolvedPositions()
   {
      for(int i = 0; i < m_activeCount; i++)
      {
         if(!m_activePositions[i].isActive) continue;
         
         // In fixed-horizon modeling, any position remaining unresolved at test end is NOT_OPEN (0.0f)
         AddSample(m_activePositions[i].baseTimestamp,
                   m_activePositions[i].posType,
                   0.0f,
                   m_activePositions[i].features,
                   m_activePositions[i].featureCount);
                   
         m_activePositions[i].isActive = false;
      }
   }
   
   //+---------------------------------------------------------------+
   //| SortChronologically: Sorts samples oldest to newest by time   |
   //| using an index array to eliminate struct/heap reallocation    |
   //+---------------------------------------------------------------+
   void SortChronologically()
   {
      if(m_sampleCount <= 1) return;
      ArrayResize(m_sortIndices, m_sampleCount);
      for(int i = 0; i < m_sampleCount; i++)
      {
         m_sortIndices[i] = i;
      }
      QuickSortIndices(0, m_sampleCount - 1);
   }
   
   //+---------------------------------------------------------------+
   //| ExportDatasets: Writes BUY CSV and SELL CSV directly          |
   //+---------------------------------------------------------------+
   bool ExportDatasets(const CFeatureExtractor &featureExt)
   {
      // 1. Process unresolved positions
      ProcessUnresolvedPositions();
      
      // 2. Sort chronologically
      SortChronologically();
      
      // 3. Prepare CSV file paths: <Symbol>_<TF>_buy.csv and <Symbol>_<TF>_sell.csv
      string buyCsvName  = m_symbol + "_" + m_tfName + "_buy.csv";
      string sellCsvName = m_symbol + "_" + m_tfName + "_sell.csv";
      
      string csvHeader = featureExt.GetCSVHeader();
      
      int buyCount = 0, sellCount = 0;
      int buyPositive = 0, sellPositive = 0;
      
      // Open CSV handles (using FILE_COMMON to survive tester sandbox deletion, fallback to local)
      int hBuy = FileOpen(buyCsvName, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(hBuy == INVALID_HANDLE)
         hBuy = FileOpen(buyCsvName, FILE_WRITE | FILE_TXT | FILE_ANSI);
         
      int hSell = FileOpen(sellCsvName, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
      if(hSell == INVALID_HANDLE)
         hSell = FileOpen(sellCsvName, FILE_WRITE | FILE_TXT | FILE_ANSI);
         
      if(hBuy != INVALID_HANDLE)
         FileWriteString(hBuy, csvHeader + "\n");
         
      if(hSell != INVALID_HANDLE)
         FileWriteString(hSell, csvHeader + "\n");
         
      // Export rows in sorted order (stripping timestamp column)
      for(int k = 0; k < m_sampleCount; k++)
      {
         int i = (ArraySize(m_sortIndices) == m_sampleCount) ? m_sortIndices[k] : k;
         string row = FormatSampleRow(m_recordedSamples[i]);
         if(m_recordedSamples[i].posType == POSITION_TYPE_BUY)
         {
            buyCount++;
            if(m_recordedSamples[i].label > 0.5f) buyPositive++;
            if(hBuy != INVALID_HANDLE)
               FileWriteString(hBuy, row + "\n");
         }
         else if(m_recordedSamples[i].posType == POSITION_TYPE_SELL)
         {
            sellCount++;
            if(m_recordedSamples[i].label > 0.5f) sellPositive++;
            if(hSell != INVALID_HANDLE)
               FileWriteString(hSell, row + "\n");
         }
      }
      
      if(hBuy != INVALID_HANDLE)  FileClose(hBuy);
      if(hSell != INVALID_HANDLE) FileClose(hSell);
      
      PrintFormat("[OrderTracker] Export complete: %s (%d rows, %d positive), %s (%d rows, %d positive)",
                  buyCsvName, buyCount, buyPositive, sellCsvName, sellCount, sellPositive);
      return true;
   }
   
   bool ExportDatasets(const CFeatureExtractor &featureExt, const SFeatureConfig &config)
   {
      return ExportDatasets(featureExt);
   }
   
   int GetSampleCount() const { return m_sampleCount; }
   
private:
   int FindActivePosition(ulong ticket)
   {
      for(int i = 0; i < m_activeCount; i++)
      {
         if(m_activePositions[i].isActive && m_activePositions[i].ticket == ticket)
            return i;
      }
      return -1;
   }
   
   void AddSample(datetime baseTs, ENUM_POSITION_TYPE posType, float label, const float &features[], int count)
   {
      int size = ArraySize(m_recordedSamples);
      if(m_sampleCount >= size)
      {
         ArrayResize(m_recordedSamples, size + 1024);
      }
      
      int idx = m_sampleCount++;
      m_recordedSamples[idx].baseTimestamp = baseTs;
      m_recordedSamples[idx].posType       = posType;
      m_recordedSamples[idx].label         = label;
      m_recordedSamples[idx].featureCount  = count;
      
      ArrayResize(m_recordedSamples[idx].features, count);
      for(int i = 0; i < count; i++)
      {
         m_recordedSamples[idx].features[i] = features[i];
      }
   }
   
   string FormatSampleRow(const SLabeledSample &sample)
   {
      string row = "";
      for(int i = 0; i < sample.featureCount; i++)
      {
         if(i > 0) row += ",";
         row += StringFormat("%.6f", sample.features[i]);
      }
      row += "," + StringFormat("%.1f", sample.label);
      return row;
   }
   
   //--- Optimized index-based QuickSort for chronological ordering by baseTimestamp
   void QuickSortIndices(int left, int right)
   {
      if(left >= right) return;
      
      int i = left, j = right;
      datetime pivot = m_recordedSamples[m_sortIndices[(left + right) / 2]].baseTimestamp;
      
      while(i <= j)
      {
         while(m_recordedSamples[m_sortIndices[i]].baseTimestamp < pivot) i++;
         while(m_recordedSamples[m_sortIndices[j]].baseTimestamp > pivot) j--;
         
         if(i <= j)
         {
            int tmp = m_sortIndices[i];
            m_sortIndices[i] = m_sortIndices[j];
            m_sortIndices[j] = tmp;
            i++;
            j--;
         }
      }
      
      if(left < j)  QuickSortIndices(left, j);
      if(i < right) QuickSortIndices(i, right);
   }
};

