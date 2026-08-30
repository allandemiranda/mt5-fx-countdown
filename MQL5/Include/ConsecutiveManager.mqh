//+------------------------------------------------------------------+
//|                                           ConsecutiveManager.mqh |
//|                                  Copyright 2026, Quant ML Engine |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Quant ML Engine"
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| CONSECUTIVE POSITION & SIGNAL MANAGEMENT ARCHITECTURE:           |
//|                                                                  |
//| 1. High Cohesion & Low Coupling:                                 |
//|    Decouples raw signal inference and GARCH/S&R calculation from |
//|    order continuation policies, stop ratcheting, and basket sync.|
//|                                                                  |
//| 2. Scientifically Validated Execution Modes:                     |
//|    - LEGACY_INDEPENDENT: Baseline multi-ticket execution.        |
//|    - SINGLE_HURDLE_RATCHET: Single order with profit hurdle      |
//|      ratchet (% TP reached before locking profit in SL).         |
//|    - SINGLE_CHAIN_LINK: Single order with previous bar close SL  |
//|      guarded by Anti-Chop / Min Displacement Filter.             |
//|    - UNIFIED_BASKET: Scale-in multi-lot basket with unified      |
//|      target and stop synchronization up to user order budget.    |
//|    - PYRAMIDING_STEP_LOCK: Incremental order scaling permitted   |
//|      only when preceding order has secured breakeven/profit.     |
//|                                                                  |
//| 3. Transversal Financial Governance (Swap & Microstructure):     |
//|    - Dynamically converts accrued overnight swap and commissions |
//|      into exact price points, guaranteeing Net Liquid Profit     |
//|      greater than or equal to 0.0f upon breakeven stop-out.      |
//|    - Enforces broker stops level and spread cushions.            |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| CONSECUTIVE SIGNAL EXECUTION MODES                               |
//+------------------------------------------------------------------+
enum ENUM_CONSECUTIVE_SIGNAL_MODE
{
   CONSECUTIVE_MODE_LEGACY_INDEPENDENT    = 0, // Legacy: Independent orders per consecutive signal
   CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET = 1, // Single Position: Hurdle Profit Ratchet (% of TP reached before moving SL)
   CONSECUTIVE_MODE_SINGLE_CHAIN_LINK     = 2, // Single Position: Anchor SL to Previous Bar Close (with Anti-Chop filter)
   CONSECUTIVE_MODE_UNIFIED_BASKET        = 3, // Unified Basket: Scale-in lots up to limit & synchronize all TP/SL
   CONSECUTIVE_MODE_PYRAMIDING_STEP_LOCK  = 4  // Pyramiding Step-Lock: Open new order only if prior order is protected in profit
};

//+------------------------------------------------------------------+
//| OPPOSING REGIME DEFENSIVE ACTIONS                                |
//+------------------------------------------------------------------+
enum ENUM_OPPOSING_DEFENSIVE_ACTION
{
   OPPOSING_ACTION_CLOSE_IF_PROFIT       = 0, // Close immediately if net liquid profit > 0
   OPPOSING_ACTION_CLOSE_IMMEDIATE       = 1, // Close immediately regardless of PnL (thesis invalidation)
   OPPOSING_ACTION_TRAILING_DEFENSIVE    = 2, // Trailing Take-Profit (tight defensive trailing stop)
   OPPOSING_ACTION_BREAKEVEN_NET         = 3, // Move SL to Net-Breakeven (covering swap/commission)
   OPPOSING_ACTION_RECALCULATE_DEFENSIVE = 4, // Recalculate defensive barriers (pull TP closer & tighten SL)
   OPPOSING_ACTION_STOP_AND_REVERSE      = 5  // Stop and Reverse: Liquidate position & open opposing order
};

//+------------------------------------------------------------------+
//| CONFIGURATION STRUCT FOR CONSECUTIVE POSITION MANAGEMENT         |
//+------------------------------------------------------------------+
struct SConsecutiveConfig
{
   ENUM_CONSECUTIVE_SIGNAL_MODE   mode;
   int                            maxConsecutiveOrders;          // Max open orders in same direction (0 = Unlimited)
   double                         hurdleProfitPct;               // % of original TP distance required before ratcheting SL (e.g. 50.0%)
   double                         profitLockPct;                 // % of accumulated profit locked into SL once hurdle met (e.g. 50.0%)
   int                            antiChopMinDisplacementPoints; // Min points displacement required before moving SL in chain-link
   int                            safetyOffsetPoints;            // Safety buffer offset beyond breakeven / anchor (points)
   bool                           enableSwapAmortization;        // Add accrued swap & commission points to protection SL
   bool                           consecutiveSlotFilter;         // Require new slot amplitude (TP-SL) >= current slot
   bool                           ignoreConflictingSignals;      // Ignore same-candle conflicting signals
   bool                           enableOpposingRegimeFilter;    // Enable opposing force regime detection
   int                            opposingStreakThreshold;       // N consecutive opposing predictions to trigger action
   ENUM_OPPOSING_DEFENSIVE_ACTION opposingAction;                // Defensive action to execute on opposing streak
   int                            opposingTrailingPoints;        // Trailing points for OPPOSING_ACTION_TRAILING_DEFENSIVE
   double                         opposingRecalculateRatio;      // Target ratio (0.1-0.9) for OPPOSING_ACTION_RECALCULATE_DEFENSIVE
};

//+------------------------------------------------------------------+
//| CLASS CConsecutiveManager                                        |
//| Encapsulates trade continuation, profit locks, basket sync,      |
//| and swap amortization logic.                                     |
//+------------------------------------------------------------------+
class CConsecutiveManager
{
private:
   string             m_symbol;
   ulong              m_magicNumber;
   SConsecutiveConfig m_config;
   int                m_buyOpposingStreak;
   int                m_sellOpposingStreak;

public:
   //+---------------------------------------------------------------+
   //| Constructor & Destructor                                      |
   //+---------------------------------------------------------------+
   CConsecutiveManager()
   {
      m_symbol             = _Symbol;
      m_magicNumber        = 0;
      m_buyOpposingStreak  = 0;
      m_sellOpposingStreak = 0;
      m_config.mode        = CONSECUTIVE_MODE_LEGACY_INDEPENDENT;
      m_config.maxConsecutiveOrders          = 3;
      m_config.hurdleProfitPct               = 50.0;
      m_config.profitLockPct                 = 50.0;
      m_config.antiChopMinDisplacementPoints = 150;
      m_config.safetyOffsetPoints            = 20;
      m_config.enableSwapAmortization        = true;
      m_config.consecutiveSlotFilter         = false;
      m_config.ignoreConflictingSignals      = true;
      m_config.enableOpposingRegimeFilter    = false;
      m_config.opposingStreakThreshold       = 2;
      m_config.opposingAction                = OPPOSING_ACTION_CLOSE_IF_PROFIT;
      m_config.opposingTrailingPoints        = 50;
      m_config.opposingRecalculateRatio      = 0.5;
   }

   ~CConsecutiveManager()
   {
   }

   //+---------------------------------------------------------------+
   //| Configuration and Initialization                              |
   //+---------------------------------------------------------------+
   void Init(const string symbol, const ulong magicNumber, const SConsecutiveConfig &config)
   {
      m_symbol             = symbol;
      m_magicNumber        = magicNumber;
      m_config             = config;
      m_buyOpposingStreak  = 0;
      m_sellOpposingStreak = 0;
   }

   void SetConfig(const SConsecutiveConfig &config)
   {
      m_config = config;
   }

   SConsecutiveConfig GetConfig() const
   {
      return m_config;
   }

   int  GetBuyOpposingStreak() const { return m_buyOpposingStreak; }
   int  GetSellOpposingStreak() const { return m_sellOpposingStreak; }
   void ResetOpposingStreaks() { m_buyOpposingStreak = 0; m_sellOpposingStreak = 0; }

   //+---------------------------------------------------------------+
   //| CountActivePositions: Inspects currently open orders for this |
   //| symbol, magic number, and direction.                          |
   //+---------------------------------------------------------------+
   int CountActivePositions(const ENUM_POSITION_TYPE posType,
                            ulong &outFirstTicket,
                            ulong &outLastTicket,
                            double &outFirstOpenPrice,
                            double &outLastOpenPrice,
                            double &outFirstSL,
                            double &outFirstTP,
                            double &outTotalVolume,
                            double &outTotalAccruedSwap)
   {
      outFirstTicket      = 0;
      outLastTicket       = 0;
      outFirstOpenPrice   = 0.0;
      outLastOpenPrice    = 0.0;
      outFirstSL          = 0.0;
      outFirstTP          = 0.0;
      outTotalVolume      = 0.0;
      outTotalAccruedSwap = 0.0;

      int count = 0;
      int total = PositionsTotal();
      datetime earliestTime = 0;
      datetime latestTime   = 0;

      for(int i = 0; i < total; i++)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
         if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
         if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

         datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);
         double vol       = PositionGetDouble(POSITION_VOLUME);
         double swap      = PositionGetDouble(POSITION_SWAP);
         double openP     = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl        = PositionGetDouble(POSITION_SL);
         double tp        = PositionGetDouble(POSITION_TP);

         outTotalVolume      += vol;
         outTotalAccruedSwap += swap;

         if(count == 0 || posTime < earliestTime)
         {
            earliestTime      = posTime;
            outFirstTicket    = ticket;
            outFirstOpenPrice = openP;
            outFirstSL        = sl;
            outFirstTP        = tp;
         }

         if(count == 0 || posTime >= latestTime)
         {
            latestTime       = posTime;
            outLastTicket    = ticket;
            outLastOpenPrice = openP;
         }

         count++;
      }

      return count;
   }

   //+---------------------------------------------------------------+
   //| CalculateSwapAmortizationPoints: Computes points required to  |
   //| neutralize accrued swap and guarantee Net Liquid Profit >= 0. |
   //+---------------------------------------------------------------+
   double CalculateSwapAmortizationPoints(const double volume, const double accruedSwap) const
   {
      if(!m_config.enableSwapAmortization || accruedSwap >= 0.0 || volume <= 0.0)
         return 0.0;

      double tickValue = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);
      double point     = SymbolInfoDouble(m_symbol, SYMBOL_POINT);

      if(tickSize <= 0.0 || point <= 0.0 || tickValue <= 0.0)
         return 0.0;

      double pointValuePerLot = (tickValue / tickSize) * point;
      if(pointValuePerLot <= 0.0)
         return 0.0;

      double totalPointValue = volume * pointValuePerLot;
      if(totalPointValue <= 0.0)
         return 0.0;

      double swapPoints = MathAbs(accruedSwap) / totalPointValue;
      return swapPoints;
   }

   //+---------------------------------------------------------------+
   //| ExecuteBuy: Governs BUY execution under consecutive signals.  |
   //+---------------------------------------------------------------+
   bool ExecuteBuy(CTrade &trade,
                   const ENUM_TIMEFRAMES period,
                   const double ask,
                   const double bid,
                   const double candidateSL,
                   const double candidateTP,
                   const double lotSize,
                   const float probBuy,
                   const bool srOk,
                   const int digits,
                   const long stopsLevel,
                   const long spread)
   {
      double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      if(point <= 0.0)
      {
         PrintFormat("[ConsecutiveManager] [ERROR] Invalid point size (%.5f) for symbol %s", point, m_symbol);
         return false;
      }
      double minStopDist = (double)(stopsLevel + spread + 5) * point;

      ulong firstTicket = 0, lastTicket = 0;
      double firstOpenPrice = 0.0, lastOpenPrice = 0.0, firstSL = 0.0, firstTP = 0.0;
      double totalVolume = 0.0, totalAccruedSwap = 0.0;

      int activeCount = CountActivePositions(POSITION_TYPE_BUY, firstTicket, lastTicket,
                                             firstOpenPrice, lastOpenPrice, firstSL, firstTP,
                                             totalVolume, totalAccruedSwap);

      // Case 1: No open positions, or Legacy Independent Mode
      if(activeCount == 0 || m_config.mode == CONSECUTIVE_MODE_LEGACY_INDEPENDENT)
      {
         trade.SetExpertMagicNumber(m_magicNumber);
         return trade.Buy(lotSize, m_symbol, ask, candidateSL, candidateTP, StringFormat("ONNX_BUY_%.2f", probBuy));
      }

      // Slot Filter Check (if enabled): candidate slot vs current slot
      if(m_config.consecutiveSlotFilter)
      {
         double currentSlot   = MathAbs(firstTP - firstSL);
         double candidateSlot = MathAbs(candidateTP - candidateSL);
         if(candidateSlot < currentSlot)
         {
            PrintFormat("[ConsecutiveManager] [SLOT FILTER] Candidate BUY slot (%.1f pts) < current slot (%.1f pts). Retaining current TP/SL.",
                        candidateSlot / point, currentSlot / point);
            return true;
         }
      }

      double swapPoints = CalculateSwapAmortizationPoints(totalVolume, totalAccruedSwap);

      // Case 2: Mode 1 - Single Position Hurdle Ratchet
      if(m_config.mode == CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET)
      {
         double favorablePoints = (bid - firstOpenPrice) / point;
         double initialTargetPoints = (firstTP > firstOpenPrice) ? (firstTP - firstOpenPrice) / point : 0.0;
         double hurdlePoints = (initialTargetPoints > 0.0) ? (initialTargetPoints * (m_config.hurdleProfitPct / 100.0)) : 0.0;

         double newSL = firstSL;
         double newTP = (candidateTP > firstTP) ? candidateTP : firstTP;

         if(hurdlePoints > 0.0 && favorablePoints >= hurdlePoints)
         {
            double lockedPoints = favorablePoints * (m_config.profitLockPct / 100.0);
            double protectionPoints = MathMax(lockedPoints, swapPoints + (double)m_config.safetyOffsetPoints);
            double candidateProtectedSL = NormalizeDouble(firstOpenPrice + (protectionPoints * point), digits);

            if(candidateProtectedSL > newSL && (bid - candidateProtectedSL) >= minStopDist)
            {
               newSL = candidateProtectedSL;
            }
         }

         if(newSL != firstSL || newTP != firstTP)
         {
            if(!trade.PositionModify(firstTicket, NormalizeDouble(newSL, digits), NormalizeDouble(newTP, digits)))
            {
               PrintFormat("[ConsecutiveManager] [WARNING] Failed to modify BUY #%I64u (SL: %.5f, TP: %.5f, Retcode: %u, Desc: %s)",
                           firstTicket, newSL, newTP, trade.ResultRetcode(), trade.ResultRetcodeDescription());
               return false;
            }
            PrintFormat("[ConsecutiveManager] [HURDLE RATCHET] Updated BUY #%I64u (SL: %.5f -> %.5f, TP: %.5f -> %.5f, Favorable: %.1f pts, SwapPts: %.1f)",
                        firstTicket, firstSL, newSL, firstTP, newTP, favorablePoints, swapPoints);
         }
         return true;
      }

      // Case 3: Mode 2 - Single Position Chain-Link (Previous Bar Close with Anti-Chop)
      if(m_config.mode == CONSECUTIVE_MODE_SINGLE_CHAIN_LINK)
      {
         double prevClose = iClose(m_symbol, period, 1);
         double displacementPoints = (prevClose - firstOpenPrice) / point;

         double newSL = firstSL;
         double newTP = (candidateTP > firstTP) ? candidateTP : firstTP;

         if(displacementPoints < (double)m_config.antiChopMinDisplacementPoints)
         {
            PrintFormat("[ConsecutiveManager] [CHAIN-LINK ANTI-CHOP] BUY #%I64u displacement (%.1f pts) < min (%.1f pts). Retaining GARCH SL to absorb chop.",
                        firstTicket, displacementPoints, (double)m_config.antiChopMinDisplacementPoints);
         }
         else
         {
            double protectedFloor = firstOpenPrice + ((swapPoints + (double)m_config.safetyOffsetPoints) * point);
            double candidateAnchorSL = NormalizeDouble(prevClose - ((double)m_config.safetyOffsetPoints * point), digits);
            candidateAnchorSL = MathMax(candidateAnchorSL, protectedFloor);

            if(candidateAnchorSL > newSL && (bid - candidateAnchorSL) >= minStopDist)
            {
               newSL = candidateAnchorSL;
            }
         }

         if(newSL != firstSL || newTP != firstTP)
         {
            if(!trade.PositionModify(firstTicket, NormalizeDouble(newSL, digits), NormalizeDouble(newTP, digits)))
            {
               PrintFormat("[ConsecutiveManager] [WARNING] Failed to modify BUY #%I64u (SL: %.5f, TP: %.5f, Retcode: %u, Desc: %s)",
                           firstTicket, newSL, newTP, trade.ResultRetcode(), trade.ResultRetcodeDescription());
               return false;
            }
            PrintFormat("[ConsecutiveManager] [CHAIN-LINK] Updated BUY #%I64u (SL: %.5f -> %.5f, TP: %.5f -> %.5f, Displacement: %.1f pts, SwapPts: %.1f)",
                        firstTicket, firstSL, newSL, firstTP, newTP, displacementPoints, swapPoints);
         }
         return true;
      }

      // Case 4: Mode 3 - Unified Basket (Scale-In lots & synchronize all stops)
      if(m_config.mode == CONSECUTIVE_MODE_UNIFIED_BASKET)
      {
         if(m_config.maxConsecutiveOrders > 0 && activeCount >= m_config.maxConsecutiveOrders)
         {
            PrintFormat("[ConsecutiveManager] [BASKET LIMIT] Max orders reached (%d/%d) for BUY. Synchronizing stops without new order.",
                        activeCount, m_config.maxConsecutiveOrders);
            SynchronizeBasket(trade, POSITION_TYPE_BUY, candidateSL, candidateTP, digits, minStopDist);
            return true;
         }

         trade.SetExpertMagicNumber(m_magicNumber);
         if(trade.Buy(lotSize, m_symbol, ask, candidateSL, candidateTP, StringFormat("ONNX_BUY_BASKET_%.2f", probBuy)))
         {
            PrintFormat("[ConsecutiveManager] [BASKET ENTRY] Opened BUY slot #%d at %.5f (TP: %.5f, SL: %.5f)",
                        activeCount + 1, ask, candidateTP, candidateSL);
            SynchronizeBasket(trade, POSITION_TYPE_BUY, candidateSL, candidateTP, digits, minStopDist);
            return true;
         }
         return false;
      }

      // Case 5: Mode 4 - Pyramiding with Step-Lock (open new order only if prior is protected)
      if(m_config.mode == CONSECUTIVE_MODE_PYRAMIDING_STEP_LOCK)
      {
         if(m_config.maxConsecutiveOrders > 0 && activeCount >= m_config.maxConsecutiveOrders)
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID LIMIT] Max orders reached (%d/%d) for BUY. Skipping.",
                        activeCount, m_config.maxConsecutiveOrders);
            return true;
         }

         double lastSL = 0.0;
         int total = PositionsTotal();
         for(int i = 0; i < total; i++)
         {
            ulong t = PositionGetTicket(i);
            if(t == lastTicket)
            {
               lastSL = PositionGetDouble(POSITION_SL);
               break;
            }
         }

         double breakevenPrice = lastOpenPrice + ((swapPoints + (double)m_config.safetyOffsetPoints) * point);
         bool isProtected = (lastSL >= NormalizeDouble(breakevenPrice, digits));

         if(!isProtected)
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID STEP-LOCK] Prior BUY #%I64u not yet protected (SL: %.5f, Required: %.5f). Skipping new slot to constrain risk.",
                        lastTicket, lastSL, breakevenPrice);
            return true;
         }

         trade.SetExpertMagicNumber(m_magicNumber);
         if(trade.Buy(lotSize, m_symbol, ask, candidateSL, candidateTP, StringFormat("ONNX_BUY_PYRAMID_%.2f", probBuy)))
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID ENTRY] Opened protected BUY slot #%d at %.5f (TP: %.5f, SL: %.5f)",
                        activeCount + 1, ask, candidateTP, candidateSL);
            return true;
         }
         return false;
      }

      return false;
   }

   //+---------------------------------------------------------------+
   //| ExecuteSell: Governs SELL execution under consecutive signals.|
   //+---------------------------------------------------------------+
   bool ExecuteSell(CTrade &trade,
                    const ENUM_TIMEFRAMES period,
                    const double ask,
                    const double bid,
                    const double candidateSL,
                    const double candidateTP,
                    const double lotSize,
                    const float probSell,
                    const bool srOk,
                    const int digits,
                    const long stopsLevel,
                    const long spread)
   {
      double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      if(point <= 0.0)
      {
         PrintFormat("[ConsecutiveManager] [ERROR] Invalid point size (%.5f) for symbol %s", point, m_symbol);
         return false;
      }
      double minStopDist = (double)(stopsLevel + spread + 5) * point;

      ulong firstTicket = 0, lastTicket = 0;
      double firstOpenPrice = 0.0, lastOpenPrice = 0.0, firstSL = 0.0, firstTP = 0.0;
      double totalVolume = 0.0, totalAccruedSwap = 0.0;

      int activeCount = CountActivePositions(POSITION_TYPE_SELL, firstTicket, lastTicket,
                                             firstOpenPrice, lastOpenPrice, firstSL, firstTP,
                                             totalVolume, totalAccruedSwap);

      // Case 1: No open positions, or Legacy Independent Mode
      if(activeCount == 0 || m_config.mode == CONSECUTIVE_MODE_LEGACY_INDEPENDENT)
      {
         trade.SetExpertMagicNumber(m_magicNumber);
         return trade.Sell(lotSize, m_symbol, bid, candidateSL, candidateTP, StringFormat("ONNX_SELL_%.2f", probSell));
      }

      // Slot Filter Check (if enabled): candidate slot vs current slot
      if(m_config.consecutiveSlotFilter)
      {
         double currentSlot   = MathAbs(firstTP - firstSL);
         double candidateSlot = MathAbs(candidateTP - candidateSL);
         if(candidateSlot < currentSlot)
         {
            PrintFormat("[ConsecutiveManager] [SLOT FILTER] Candidate SELL slot (%.1f pts) < current slot (%.1f pts). Retaining current TP/SL.",
                        candidateSlot / point, currentSlot / point);
            return true;
         }
      }

      double swapPoints = CalculateSwapAmortizationPoints(totalVolume, totalAccruedSwap);

      // Case 2: Mode 1 - Single Position Hurdle Ratchet
      if(m_config.mode == CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET)
      {
         double favorablePoints = (firstOpenPrice - ask) / point;
         double initialTargetPoints = (firstTP > 0.0 && firstOpenPrice > firstTP) ? (firstOpenPrice - firstTP) / point : 0.0;
         double hurdlePoints = (initialTargetPoints > 0.0) ? (initialTargetPoints * (m_config.hurdleProfitPct / 100.0)) : 0.0;

         double newSL = firstSL;
         double newTP = (firstTP == 0.0 || (candidateTP > 0.0 && candidateTP < firstTP)) ? candidateTP : firstTP;

         if(hurdlePoints > 0.0 && favorablePoints >= hurdlePoints)
         {
            double lockedPoints = favorablePoints * (m_config.profitLockPct / 100.0);
            double protectionPoints = MathMax(lockedPoints, swapPoints + (double)m_config.safetyOffsetPoints);
            double candidateProtectedSL = NormalizeDouble(firstOpenPrice - (protectionPoints * point), digits);

            if((newSL == 0.0 || candidateProtectedSL < newSL) && (candidateProtectedSL - ask) >= minStopDist)
            {
               newSL = candidateProtectedSL;
            }
         }

         if(newSL != firstSL || newTP != firstTP)
         {
            if(!trade.PositionModify(firstTicket, NormalizeDouble(newSL, digits), NormalizeDouble(newTP, digits)))
            {
               PrintFormat("[ConsecutiveManager] [WARNING] Failed to modify SELL #%I64u (SL: %.5f, TP: %.5f, Retcode: %u, Desc: %s)",
                           firstTicket, newSL, newTP, trade.ResultRetcode(), trade.ResultRetcodeDescription());
               return false;
            }
            PrintFormat("[ConsecutiveManager] [HURDLE RATCHET] Updated SELL #%I64u (SL: %.5f -> %.5f, TP: %.5f -> %.5f, Favorable: %.1f pts, SwapPts: %.1f)",
                        firstTicket, firstSL, newSL, firstTP, newTP, favorablePoints, swapPoints);
         }
         return true;
      }

      // Case 3: Mode 2 - Single Position Chain-Link (Previous Bar Close with Anti-Chop)
      if(m_config.mode == CONSECUTIVE_MODE_SINGLE_CHAIN_LINK)
      {
         double prevClose = iClose(m_symbol, period, 1);
         double displacementPoints = (firstOpenPrice - prevClose) / point;

         double newSL = firstSL;
         double newTP = (firstTP == 0.0 || (candidateTP > 0.0 && candidateTP < firstTP)) ? candidateTP : firstTP;

         if(displacementPoints < (double)m_config.antiChopMinDisplacementPoints)
         {
            PrintFormat("[ConsecutiveManager] [CHAIN-LINK ANTI-CHOP] SELL #%I64u displacement (%.1f pts) < min (%.1f pts). Retaining GARCH SL to absorb chop.",
                        firstTicket, displacementPoints, (double)m_config.antiChopMinDisplacementPoints);
         }
         else
         {
            double protectedFloor = firstOpenPrice - ((swapPoints + (double)m_config.safetyOffsetPoints) * point);
            double candidateAnchorSL = NormalizeDouble(prevClose + ((double)m_config.safetyOffsetPoints * point), digits);
            candidateAnchorSL = MathMin(candidateAnchorSL, protectedFloor);

            if((newSL == 0.0 || candidateAnchorSL < newSL) && (candidateAnchorSL - ask) >= minStopDist)
            {
               newSL = candidateAnchorSL;
            }
         }

         if(newSL != firstSL || newTP != firstTP)
         {
            if(!trade.PositionModify(firstTicket, NormalizeDouble(newSL, digits), NormalizeDouble(newTP, digits)))
            {
               PrintFormat("[ConsecutiveManager] [WARNING] Failed to modify SELL #%I64u (SL: %.5f, TP: %.5f, Retcode: %u, Desc: %s)",
                           firstTicket, newSL, newTP, trade.ResultRetcode(), trade.ResultRetcodeDescription());
               return false;
            }
            PrintFormat("[ConsecutiveManager] [CHAIN-LINK] Updated SELL #%I64u (SL: %.5f -> %.5f, TP: %.5f -> %.5f, Displacement: %.1f pts, SwapPts: %.1f)",
                        firstTicket, firstSL, newSL, firstTP, newTP, displacementPoints, swapPoints);
         }
         return true;
      }

      // Case 4: Mode 3 - Unified Basket (Scale-In lots & synchronize all stops)
      if(m_config.mode == CONSECUTIVE_MODE_UNIFIED_BASKET)
      {
         if(m_config.maxConsecutiveOrders > 0 && activeCount >= m_config.maxConsecutiveOrders)
         {
            PrintFormat("[ConsecutiveManager] [BASKET LIMIT] Max orders reached (%d/%d) for SELL. Synchronizing stops without new order.",
                        activeCount, m_config.maxConsecutiveOrders);
            SynchronizeBasket(trade, POSITION_TYPE_SELL, candidateSL, candidateTP, digits, minStopDist);
            return true;
         }

         trade.SetExpertMagicNumber(m_magicNumber);
         if(trade.Sell(lotSize, m_symbol, bid, candidateSL, candidateTP, StringFormat("ONNX_SELL_BASKET_%.2f", probSell)))
         {
            PrintFormat("[ConsecutiveManager] [BASKET ENTRY] Opened SELL slot #%d at %.5f (TP: %.5f, SL: %.5f)",
                        activeCount + 1, bid, candidateTP, candidateSL);
            SynchronizeBasket(trade, POSITION_TYPE_SELL, candidateSL, candidateTP, digits, minStopDist);
            return true;
         }
         return false;
      }

      // Case 5: Mode 4 - Pyramiding with Step-Lock (open new order only if prior is protected)
      if(m_config.mode == CONSECUTIVE_MODE_PYRAMIDING_STEP_LOCK)
      {
         if(m_config.maxConsecutiveOrders > 0 && activeCount >= m_config.maxConsecutiveOrders)
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID LIMIT] Max orders reached (%d/%d) for SELL. Skipping.",
                        activeCount, m_config.maxConsecutiveOrders);
            return true;
         }

         double lastSL = 0.0;
         int total = PositionsTotal();
         for(int i = 0; i < total; i++)
         {
            ulong t = PositionGetTicket(i);
            if(t == lastTicket)
            {
               lastSL = PositionGetDouble(POSITION_SL);
               break;
            }
         }

         double breakevenPrice = lastOpenPrice - ((swapPoints + (double)m_config.safetyOffsetPoints) * point);
         bool isProtected = (lastSL > 0.0 && lastSL <= NormalizeDouble(breakevenPrice, digits));

         if(!isProtected)
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID STEP-LOCK] Prior SELL #%I64u not yet protected (SL: %.5f, Required: %.5f). Skipping new slot to constrain risk.",
                        lastTicket, lastSL, breakevenPrice);
            return true;
         }

         trade.SetExpertMagicNumber(m_magicNumber);
         if(trade.Sell(lotSize, m_symbol, bid, candidateSL, candidateTP, StringFormat("ONNX_SELL_PYRAMID_%.2f", probSell)))
         {
            PrintFormat("[ConsecutiveManager] [PYRAMID ENTRY] Opened protected SELL slot #%d at %.5f (TP: %.5f, SL: %.5f)",
                        activeCount + 1, bid, candidateTP, candidateSL);
            return true;
         }
         return false;
      }

      return false;
   }

private:
   int GetStreakForType(const ENUM_POSITION_TYPE posType) const
   {
      return (posType == POSITION_TYPE_BUY) ? m_buyOpposingStreak : m_sellOpposingStreak;
   }

   void SetStreakForType(const ENUM_POSITION_TYPE posType, const int val)
   {
      if(posType == POSITION_TYPE_BUY) m_buyOpposingStreak = val;
      else                             m_sellOpposingStreak = val;
   }

   void IncrementStreakForType(const ENUM_POSITION_TYPE posType)
   {
      if(posType == POSITION_TYPE_BUY) m_buyOpposingStreak++;
      else                             m_sellOpposingStreak++;
   }

   //+---------------------------------------------------------------+
   //| SynchronizeBasket: Synchronizes all open orders in the basket |
   //| to the latest candidate SL and TP levels.                     |
   //+---------------------------------------------------------------+
   bool SynchronizeBasket(CTrade &trade,
                          const ENUM_POSITION_TYPE posType,
                          const double newSL,
                          const double newTP,
                          const int digits,
                          const double minStopDist)
   {
      int total = PositionsTotal();
      for(int i = total - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
         if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
         if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

         double curSL = PositionGetDouble(POSITION_SL);
         double curTP = PositionGetDouble(POSITION_TP);

         double targetSL = curSL;
         if(posType == POSITION_TYPE_BUY)
         {
            double bid = SymbolInfoDouble(m_symbol, SYMBOL_BID);
            if(newSL > curSL && (bid - newSL) >= minStopDist)
               targetSL = newSL;
         }
         else
         {
            double ask = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
            if((newSL < curSL || curSL == 0.0) && (newSL - ask) >= minStopDist)
               targetSL = newSL;
         }

         double targetTP = newTP;

         if(targetSL != curSL || targetTP != curTP)
         {
            trade.PositionModify(ticket, NormalizeDouble(targetSL, digits), NormalizeDouble(targetTP, digits));
         }
      }
      return true;
   }

public:
   //+---------------------------------------------------------------+
   //| CheckAndProcessOpposingRegime: Monitors opposing signal       |
   //| streaks against open positions and executes defensive action. |
   //+---------------------------------------------------------------+
   bool CheckAndProcessOpposingRegime(CTrade &trade,
                                      const ENUM_POSITION_TYPE posType,
                                      const bool opposingSignalActive,
                                      const double ask,
                                      const double bid,
                                      const int digits,
                                      const long stopsLevel,
                                      const long spread,
                                      const double candidateSL,
                                      const double candidateTP,
                                      const double lotSize,
                                      const float probOpposing,
                                      const bool srOk)
   {
      if(!m_config.enableOpposingRegimeFilter)
         return false;

      ulong firstTicket = 0, lastTicket = 0;
      double firstOpenPrice = 0.0, lastOpenPrice = 0.0, firstSL = 0.0, firstTP = 0.0;
      double totalVolume = 0.0, totalAccruedSwap = 0.0;

      int activeCount = CountActivePositions(posType, firstTicket, lastTicket,
                                             firstOpenPrice, lastOpenPrice, firstSL, firstTP,
                                             totalVolume, totalAccruedSwap);

      int currentStreak = GetStreakForType(posType);

      if(activeCount == 0)
      {
         SetStreakForType(posType, 0);
         return false;
      }

      if(opposingSignalActive)
      {
         IncrementStreakForType(posType);
         currentStreak = GetStreakForType(posType);
         PrintFormat("[ConsecutiveManager] [OPPOSING REGIME] Position %s has opposing signal streak: %d/%d (probOpposing: %.4f)",
                     (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"), currentStreak, m_config.opposingStreakThreshold, probOpposing);
      }
      else
      {
         if(currentStreak > 0)
         {
            PrintFormat("[ConsecutiveManager] [OPPOSING REGIME] Opposing signal subsided for %s. Resetting streak from %d to 0.",
                        (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"), currentStreak);
         }
         SetStreakForType(posType, 0);
         return false;
      }

      if(currentStreak < m_config.opposingStreakThreshold)
         return false;

      // Streak threshold met! Execute configured defensive action
      PrintFormat("[ConsecutiveManager] [OPPOSING REGIME TRIGGER] Streak threshold %d reached for %s positions! Executing action: %d",
                  currentStreak, (posType == POSITION_TYPE_BUY ? "BUY" : "SELL"), (int)m_config.opposingAction);

      double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      double minStopDist = (double)(stopsLevel + spread + 5) * point;

      // Action 0: CLOSE_IF_PROFIT
      if(m_config.opposingAction == OPPOSING_ACTION_CLOSE_IF_PROFIT)
      {
         int total = PositionsTotal();
         bool anyClosed = false;
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            double profit = PositionGetDouble(POSITION_PROFIT);
            double swap   = PositionGetDouble(POSITION_SWAP);
            double netProfit = profit + swap;

            if(netProfit > 0.0)
            {
               PrintFormat("[ConsecutiveManager] [ACTION: CLOSE_IF_PROFIT] Closing position #%I64u in profit (Net: %.2f) ahead of opposing regime.",
                           ticket, netProfit);
               trade.PositionClose(ticket);
               anyClosed = true;
            }
            else
            {
               PrintFormat("[ConsecutiveManager] [ACTION: CLOSE_IF_PROFIT] Position #%I64u not in net profit (Net: %.2f). Retaining stops.",
                           ticket, netProfit);
            }
         }
         if(anyClosed) SetStreakForType(posType, 0);
         return anyClosed;
      }

      // Action 1: CLOSE_IMMEDIATE
      if(m_config.opposingAction == OPPOSING_ACTION_CLOSE_IMMEDIATE)
      {
         int total = PositionsTotal();
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            PrintFormat("[ConsecutiveManager] [ACTION: CLOSE_IMMEDIATE] Liquidating position #%I64u due to statistical thesis invalidation.",
                        ticket);
            trade.PositionClose(ticket);
         }
         SetStreakForType(posType, 0);
         return true;
      }

      // Action 2: TRAILING_DEFENSIVE (Trailing Take-Profit)
      if(m_config.opposingAction == OPPOSING_ACTION_TRAILING_DEFENSIVE)
      {
         double trailDist = (double)m_config.opposingTrailingPoints * point;
         int total = PositionsTotal();
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            double curSL = PositionGetDouble(POSITION_SL);
            double curTP = PositionGetDouble(POSITION_TP);

            if(posType == POSITION_TYPE_BUY)
            {
               double newSL = NormalizeDouble(bid - trailDist, digits);
               if(newSL > curSL && (bid - newSL) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, curTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: TRAILING_DEFENSIVE] Trailed BUY #%I64u SL to %.5f.", ticket, newSL);
               }
            }
            else
            {
               double newSL = NormalizeDouble(ask + trailDist, digits);
               if((newSL < curSL || curSL == 0.0) && (newSL - ask) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, curTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: TRAILING_DEFENSIVE] Trailed SELL #%I64u SL to %.5f.", ticket, newSL);
               }
            }
         }
         return true;
      }

      // Action 3: BREAKEVEN_NET
      if(m_config.opposingAction == OPPOSING_ACTION_BREAKEVEN_NET)
      {
         double swapPoints = CalculateSwapAmortizationPoints(totalVolume, totalAccruedSwap);
         int total = PositionsTotal();
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            double openP = PositionGetDouble(POSITION_PRICE_OPEN);
            double curSL = PositionGetDouble(POSITION_SL);
            double curTP = PositionGetDouble(POSITION_TP);

            if(posType == POSITION_TYPE_BUY)
            {
               double newSL = NormalizeDouble(openP + ((swapPoints + (double)m_config.safetyOffsetPoints) * point), digits);
               if(bid > newSL && newSL > curSL && (bid - newSL) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, curTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: BREAKEVEN_NET] Protected BUY #%I64u SL to %.5f (SwapPts: %.1f).", ticket, newSL, swapPoints);
               }
            }
            else
            {
               double newSL = NormalizeDouble(openP - ((swapPoints + (double)m_config.safetyOffsetPoints) * point), digits);
               if(ask < newSL && (newSL < curSL || curSL == 0.0) && (newSL - ask) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, curTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: BREAKEVEN_NET] Protected SELL #%I64u SL to %.5f (SwapPts: %.1f).", ticket, newSL, swapPoints);
               }
            }
         }
         return true;
      }

      // Action 4: RECALCULATE_DEFENSIVE
      if(m_config.opposingAction == OPPOSING_ACTION_RECALCULATE_DEFENSIVE)
      {
         double swapPoints = CalculateSwapAmortizationPoints(totalVolume, totalAccruedSwap);
         int total = PositionsTotal();
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            double openP = PositionGetDouble(POSITION_PRICE_OPEN);
            double curSL = PositionGetDouble(POSITION_SL);
            double curTP = PositionGetDouble(POSITION_TP);

            double ratio = (m_config.opposingRecalculateRatio > 0.0 && m_config.opposingRecalculateRatio < 1.0)
                           ? m_config.opposingRecalculateRatio : 0.5;

            if(posType == POSITION_TYPE_BUY)
            {
               double targetDist = (curTP > openP) ? (curTP - openP) * ratio : 0.0;
               double newTP = (targetDist > 0.0) ? NormalizeDouble(openP + targetDist, digits) : curTP;
               double candidateProtSL = NormalizeDouble(openP + (swapPoints * point), digits);
               double newSL = MathMax(curSL, candidateProtSL);
               if((bid - newSL) >= minStopDist && (newTP - bid) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, newTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: RECALCULATE_DEFENSIVE] Tightened BUY #%I64u (SL: %.5f, TP: %.5f).", ticket, newSL, newTP);
               }
            }
            else
            {
               double targetDist = (curTP > 0.0 && curTP < openP) ? (openP - curTP) * ratio : 0.0;
               double newTP = (targetDist > 0.0) ? NormalizeDouble(openP - targetDist, digits) : curTP;
               double candidateProtSL = NormalizeDouble(openP - (swapPoints * point), digits);
               double newSL = (curSL == 0.0) ? candidateProtSL : MathMin(curSL, candidateProtSL);
               if((newSL - ask) >= minStopDist && (ask - newTP) >= minStopDist)
               {
                  trade.PositionModify(ticket, newSL, newTP);
                  PrintFormat("[ConsecutiveManager] [ACTION: RECALCULATE_DEFENSIVE] Tightened SELL #%I64u (SL: %.5f, TP: %.5f).", ticket, newSL, newTP);
               }
            }
         }
         return true;
      }

      // Action 5: STOP_AND_REVERSE
      if(m_config.opposingAction == OPPOSING_ACTION_STOP_AND_REVERSE)
      {
         // 1. Close all existing positions of posType
         int total = PositionsTotal();
         for(int i = total - 1; i >= 0; i--)
         {
            ulong ticket = PositionGetTicket(i);
            if(ticket == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)m_magicNumber) continue;
            if(PositionGetInteger(POSITION_TYPE) != (long)posType) continue;

            trade.PositionClose(ticket);
         }

         // 2. Open reverse position
         trade.SetExpertMagicNumber(m_magicNumber);
         if(posType == POSITION_TYPE_BUY)
         {
            trade.Sell(lotSize, m_symbol, bid, candidateSL, candidateTP, StringFormat("ONNX_REV_SELL_%.2f", probOpposing));
            PrintFormat("[ConsecutiveManager] [ACTION: STOP_AND_REVERSE] Closed BUY positions and reversed to SELL at %.5f (Lot: %.2f).", bid, lotSize);
         }
         else
         {
            trade.Buy(lotSize, m_symbol, ask, candidateSL, candidateTP, StringFormat("ONNX_REV_BUY_%.2f", probOpposing));
            PrintFormat("[ConsecutiveManager] [ACTION: STOP_AND_REVERSE] Closed SELL positions and reversed to BUY at %.5f (Lot: %.2f).", ask, lotSize);
         }

         SetStreakForType(posType, 0);
         return true;
      }

      return false;
   }
};
//+------------------------------------------------------------------+
