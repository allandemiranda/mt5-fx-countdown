//+------------------------------------------------------------------+
//|                                     TestConsecutiveManager.mqh   |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|        Unit Tests for CConsecutiveManager (MQL5 Architecture)    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property strict

#include <Tests/MqlTestFramework.mqh>
#include <ConsecutiveManager.mqh>

//+------------------------------------------------------------------+
//| CTestConsecutiveManager: Verification suite for CConsecutiveMgr  |
//+------------------------------------------------------------------+
class CTestConsecutiveManager
{
public:
   static void RunAll(CMqlTestFramework &tf)
   {
      tf.SetSuiteName("TestConsecutiveManager");

      Test_DefaultConstructor(tf);
      Test_SetConfig_GetConfig(tf);
      Test_CalculateSwapAmortization_PositiveOrZeroSwap_ReturnsZero(tf);
      Test_CalculateSwapAmortization_Disabled_ReturnsZero(tf);
      Test_CalculateSwapAmortization_ZeroVolume_ReturnsZero(tf);
      Test_CountActivePositions_EmptyBook_ReturnsZero(tf);
      Test_ModeEnumeration_Invariants(tf);
      Test_OpposingActionEnumeration_Invariants(tf);
      Test_OpposingRegime_DisabledFilter_ReturnsFalse(tf);
   }

private:
   // Scenario 1: Default constructor initializes institutional baseline defaults
   static void Test_DefaultConstructor(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      SConsecutiveConfig cfg = mgr.GetConfig();

      tf.AssertEqualInt((int)cfg.mode, (int)CONSECUTIVE_MODE_LEGACY_INDEPENDENT, "DefaultConstructor_Mode");
      tf.AssertEqualInt(cfg.maxConsecutiveOrders, 3, "DefaultConstructor_MaxOrders");
      tf.AssertEqualDouble(cfg.hurdleProfitPct, 50.0, 1e-6, "DefaultConstructor_HurdleProfitPct");
      tf.AssertEqualDouble(cfg.profitLockPct, 50.0, 1e-6, "DefaultConstructor_ProfitLockPct");
      tf.AssertEqualInt(cfg.antiChopMinDisplacementPoints, 150, "DefaultConstructor_AntiChopMinDisplacement");
      tf.AssertEqualInt(cfg.safetyOffsetPoints, 20, "DefaultConstructor_SafetyOffsetPoints");
      tf.AssertTrue(cfg.enableSwapAmortization, "DefaultConstructor_EnableSwapAmortization");
      tf.AssertFalse(cfg.consecutiveSlotFilter, "DefaultConstructor_ConsecutiveSlotFilter");
      tf.AssertFalse(cfg.enableOpposingRegimeFilter, "DefaultConstructor_EnableOpposingRegimeFilter");
      tf.AssertEqualInt(cfg.opposingStreakThreshold, 2, "DefaultConstructor_OpposingStreakThreshold");
      tf.AssertEqualInt((int)cfg.opposingAction, (int)OPPOSING_ACTION_CLOSE_IF_PROFIT, "DefaultConstructor_OpposingAction");
      tf.AssertEqualInt(cfg.opposingTrailingPoints, 50, "DefaultConstructor_OpposingTrailingPoints");
      tf.AssertEqualDouble(cfg.opposingRecalculateRatio, 0.5, 1e-6, "DefaultConstructor_OpposingRecalculateRatio");
   }

   // Scenario 2: SetConfig and GetConfig update and retrieve configurations correctly
   static void Test_SetConfig_GetConfig(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      SConsecutiveConfig cfg;
      cfg.mode                          = CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET;
      cfg.maxConsecutiveOrders          = 5;
      cfg.hurdleProfitPct               = 60.0;
      cfg.profitLockPct                 = 40.0;
      cfg.antiChopMinDisplacementPoints = 200;
      cfg.safetyOffsetPoints            = 30;
      cfg.enableSwapAmortization        = false;
      cfg.consecutiveSlotFilter         = true;

      mgr.SetConfig(cfg);
      SConsecutiveConfig readCfg = mgr.GetConfig();

      tf.AssertEqualInt((int)readCfg.mode, (int)CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET, "SetConfig_Mode");
      tf.AssertEqualInt(readCfg.maxConsecutiveOrders, 5, "SetConfig_MaxOrders");
      tf.AssertEqualDouble(readCfg.hurdleProfitPct, 60.0, 1e-6, "SetConfig_HurdleProfitPct");
      tf.AssertEqualDouble(readCfg.profitLockPct, 40.0, 1e-6, "SetConfig_ProfitLockPct");
      tf.AssertEqualInt(readCfg.antiChopMinDisplacementPoints, 200, "SetConfig_AntiChopMinDisplacement");
      tf.AssertEqualInt(readCfg.safetyOffsetPoints, 30, "SetConfig_SafetyOffsetPoints");
      tf.AssertFalse(readCfg.enableSwapAmortization, "SetConfig_EnableSwapAmortization");
      tf.AssertTrue(readCfg.consecutiveSlotFilter, "SetConfig_ConsecutiveSlotFilter");
   }

   // Scenario 3: CalculateSwapAmortizationPoints returns 0.0 when swap is non-negative
   static void Test_CalculateSwapAmortization_PositiveOrZeroSwap_ReturnsZero(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      double ptsZero = mgr.CalculateSwapAmortizationPoints(0.01, 0.0);
      tf.AssertEqualDouble(ptsZero, 0.0, 1e-6, "SwapAmortization_ZeroSwap");

      double ptsPositive = mgr.CalculateSwapAmortizationPoints(0.01, 5.50);
      tf.AssertEqualDouble(ptsPositive, 0.0, 1e-6, "SwapAmortization_PositiveSwap");
   }

   // Scenario 4: CalculateSwapAmortizationPoints returns 0.0 when feature toggle is disabled
   static void Test_CalculateSwapAmortization_Disabled_ReturnsZero(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      SConsecutiveConfig cfg = mgr.GetConfig();
      cfg.enableSwapAmortization = false;
      mgr.SetConfig(cfg);

      double ptsDisabled = mgr.CalculateSwapAmortizationPoints(0.01, -10.0);
      tf.AssertEqualDouble(ptsDisabled, 0.0, 1e-6, "SwapAmortization_DisabledToggle");
   }

   // Scenario 5: CalculateSwapAmortizationPoints defends against zero volume
   static void Test_CalculateSwapAmortization_ZeroVolume_ReturnsZero(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      double ptsZeroVol = mgr.CalculateSwapAmortizationPoints(0.0, -10.0);
      tf.AssertEqualDouble(ptsZeroVol, 0.0, 1e-6, "SwapAmortization_ZeroVolume");
   }

   // Scenario 6: CountActivePositions on clean environment returns zero count
   static void Test_CountActivePositions_EmptyBook_ReturnsZero(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      SConsecutiveConfig cfg = mgr.GetConfig();
      mgr.Init("UNKNOWN_SYM", 999999, cfg);

      ulong firstT = 0, lastT = 0;
      double firstOpen = 0.0, lastOpen = 0.0, firstSL = 0.0, firstTP = 0.0;
      double totalVol = 0.0, totalSwap = 0.0;

      int count = mgr.CountActivePositions(POSITION_TYPE_BUY, firstT, lastT, firstOpen, lastOpen, firstSL, firstTP, totalVol, totalSwap);
      tf.AssertEqualInt(count, 0, "CountActivePositions_UnknownSymbol");
      tf.AssertEqualInt((int)firstT, 0, "CountActivePositions_FirstTicketZero");
      tf.AssertEqualDouble(totalVol, 0.0, 1e-6, "CountActivePositions_TotalVolZero");
   }

   // Scenario 7: Validate ENUM_CONSECUTIVE_SIGNAL_MODE enumeration values
   static void Test_ModeEnumeration_Invariants(CMqlTestFramework &tf)
   {
      tf.AssertEqualInt((int)CONSECUTIVE_MODE_LEGACY_INDEPENDENT, 0, "Enum_Legacy");
      tf.AssertEqualInt((int)CONSECUTIVE_MODE_SINGLE_HURDLE_RATCHET, 1, "Enum_HurdleRatchet");
      tf.AssertEqualInt((int)CONSECUTIVE_MODE_SINGLE_CHAIN_LINK, 2, "Enum_ChainLink");
      tf.AssertEqualInt((int)CONSECUTIVE_MODE_UNIFIED_BASKET, 3, "Enum_UnifiedBasket");
      tf.AssertEqualInt((int)CONSECUTIVE_MODE_PYRAMIDING_STEP_LOCK, 4, "Enum_PyramidingStepLock");
   }

   // Scenario 8: Validate ENUM_OPPOSING_DEFENSIVE_ACTION enumeration values
   static void Test_OpposingActionEnumeration_Invariants(CMqlTestFramework &tf)
   {
      tf.AssertEqualInt((int)OPPOSING_ACTION_CLOSE_IF_PROFIT, 0, "Enum_OpposingCloseIfProfit");
      tf.AssertEqualInt((int)OPPOSING_ACTION_CLOSE_IMMEDIATE, 1, "Enum_OpposingCloseImmediate");
      tf.AssertEqualInt((int)OPPOSING_ACTION_TRAILING_DEFENSIVE, 2, "Enum_OpposingTrailingDefensive");
      tf.AssertEqualInt((int)OPPOSING_ACTION_BREAKEVEN_NET, 3, "Enum_OpposingBreakevenNet");
      tf.AssertEqualInt((int)OPPOSING_ACTION_RECALCULATE_DEFENSIVE, 4, "Enum_OpposingRecalculateDefensive");
      tf.AssertEqualInt((int)OPPOSING_ACTION_STOP_AND_REVERSE, 5, "Enum_OpposingStopAndReverse");
   }

   // Scenario 9: CheckAndProcessOpposingRegime returns false when feature toggle is disabled
   static void Test_OpposingRegime_DisabledFilter_ReturnsFalse(CMqlTestFramework &tf)
   {
      CConsecutiveManager mgr;
      SConsecutiveConfig cfg = mgr.GetConfig();
      cfg.enableOpposingRegimeFilter = false;
      mgr.SetConfig(cfg);

      CTrade dummyTrade;
      bool res = mgr.CheckAndProcessOpposingRegime(dummyTrade, POSITION_TYPE_BUY, true,
                                                   1.0850, 1.0848, 5, 10, 20,
                                                   1.0870, 1.0800, 0.01, 0.75f, false);
      tf.AssertFalse(res, "OpposingRegime_DisabledFilter_ReturnsFalse");
   }
};
//+------------------------------------------------------------------+
