//+------------------------------------------------------------------+
//|                                             TestGarchEngine.mqh  |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|               Black-Box Unit Tests for CGarchEngine (MQL5)       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property strict

#include <Tests/MqlTestFramework.mqh>
#include <GarchEngine.mqh>

//+------------------------------------------------------------------+
//| CTestGarchEngine: Black-box verification of CGarchEngine         |
//+------------------------------------------------------------------+
class CTestGarchEngine
{
public:
   static void RunAll(CMqlTestFramework &tf)
   {
      tf.SetSuiteName("TestGarchEngine");
      
      Test_DefaultConstructor(tf);
      Test_ParameterizedConstructor(tf);
      Test_SetParameters_Nominal(tf);
      Test_SetParameters_LowerBoundsClamping(tf);
      Test_SetParameters_StationarityViolation(tf);
      Test_SetParameters_InvalidNegativeValues(tf);
      Test_ComputeGarchMetrics_InvalidSymbol_ReturnsFalse(tf);
      Test_CalculateDynamicRisk_ZeroPointProtection(tf);
   }

private:
   // Scenario 1: Default constructor initializes institutional baseline invariants
   static void Test_DefaultConstructor(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      tf.AssertEqualInt(garch.GetPriceSize(), 500, "DefaultConstructor_PriceSize");
      tf.AssertEqualInt(garch.GetHorizon(), 8, "DefaultConstructor_Horizon");
      tf.AssertEqualDouble(garch.GetAlpha(), 0.05, 1e-6, "DefaultConstructor_Alpha");
      tf.AssertEqualDouble(garch.GetBeta(), 0.92, 1e-6, "DefaultConstructor_Beta");
   }

   // Scenario 2: Parameterized constructor sets valid user parameters
   static void Test_ParameterizedConstructor(CMqlTestFramework &tf)
   {
      CGarchEngine garch(250, 6, 0.08, 0.85);
      tf.AssertEqualInt(garch.GetPriceSize(), 250, "ParameterizedConstructor_PriceSize");
      tf.AssertEqualInt(garch.GetHorizon(), 6, "ParameterizedConstructor_Horizon");
      tf.AssertEqualDouble(garch.GetAlpha(), 0.08, 1e-6, "ParameterizedConstructor_Alpha");
      tf.AssertEqualDouble(garch.GetBeta(), 0.85, 1e-6, "ParameterizedConstructor_Beta");
   }

   // Scenario 3: SetParameters accepts nominal parameters within bounds
   static void Test_SetParameters_Nominal(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      garch.SetParameters(300, 10, 0.07, 0.90);
      tf.AssertEqualInt(garch.GetPriceSize(), 300, "SetParameters_Nominal_PriceSize");
      tf.AssertEqualInt(garch.GetHorizon(), 10, "SetParameters_Nominal_Horizon");
      tf.AssertEqualDouble(garch.GetAlpha(), 0.07, 1e-6, "SetParameters_Nominal_Alpha");
      tf.AssertEqualDouble(garch.GetBeta(), 0.90, 1e-6, "SetParameters_Nominal_Beta");
   }

   // Scenario 4: SetParameters clamps lower bounds (priceSize < 30 -> 200, horizon < 1 -> 5)
   static void Test_SetParameters_LowerBoundsClamping(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      garch.SetParameters(10, 0, 0.05, 0.92);
      tf.AssertEqualInt(garch.GetPriceSize(), 200, "SetParameters_LowerBounds_PriceSizeClampedTo200");
      tf.AssertEqualInt(garch.GetHorizon(), 5, "SetParameters_LowerBounds_HorizonClampedTo5");
   }

   // Scenario 5: SetParameters enforces covariance stationarity (alpha + beta >= 1.0 clamped to 0.05 / 0.92)
   static void Test_SetParameters_StationarityViolation(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      garch.SetParameters(500, 8, 0.15, 0.88); // 0.15 + 0.88 = 1.03 >= 1.0
      tf.AssertEqualDouble(garch.GetAlpha(), 0.05, 1e-6, "SetParameters_Stationarity_AlphaReset");
      tf.AssertEqualDouble(garch.GetBeta(), 0.92, 1e-6, "SetParameters_Stationarity_BetaReset");
   }

   // Scenario 6: SetParameters handles non-positive values gracefully
   static void Test_SetParameters_InvalidNegativeValues(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      garch.SetParameters(500, 8, -0.05, -0.92);
      tf.AssertEqualDouble(garch.GetAlpha(), 0.05, 1e-6, "SetParameters_Negative_AlphaReset");
      tf.AssertEqualDouble(garch.GetBeta(), 0.92, 1e-6, "SetParameters_Negative_BetaReset");
   }

   // Scenario 7: ComputeGarchMetrics handles non-existent symbol with graceful failure (returns false)
   static void Test_ComputeGarchMetrics_InvalidSymbol_ReturnsFalse(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      double omega = 0.0, volRatio = 0.0, volTrend = 0.0, sigmaCond = 0.0, sigmaAgg = 0.0;
      bool ok = garch.ComputeGarchMetrics("NON_EXISTENT_SYMBOL_XYZ", PERIOD_M1, 0,
                                          omega, volRatio, volTrend, sigmaCond, sigmaAgg);
      tf.AssertFalse(ok, "ComputeGarchMetrics_InvalidSymbol_ReturnsFalse");
   }

   // Scenario 8: CalculateDynamicRisk returns false safely when symbol is invalid
   static void Test_CalculateDynamicRisk_ZeroPointProtection(CMqlTestFramework &tf)
   {
      CGarchEngine garch;
      double tpPoints = 0.0, slPoints = 0.0, sigmaAgg = 0.0;
      bool ok = garch.CalculateDynamicRisk("NON_EXISTENT_SYMBOL_XYZ", PERIOD_M1, 1.5, 1.0,
                                           tpPoints, slPoints, sigmaAgg);
      tf.AssertFalse(ok, "CalculateDynamicRisk_InvalidSymbol_ReturnsFalse");
   }
};
