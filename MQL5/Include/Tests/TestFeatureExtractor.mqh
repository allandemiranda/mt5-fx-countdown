//+------------------------------------------------------------------+
//|                                         TestFeatureExtractor.mqh |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|            Black-Box Unit Tests for CFeatureExtractor (MQL5)     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property strict

#include <Tests/MqlTestFramework.mqh>
#include <FeatureExtractor.mqh>

//+------------------------------------------------------------------+
//| CTestFeatureExtractor: Black-box tests for CFeatureExtractor     |
//+------------------------------------------------------------------+
class CTestFeatureExtractor
{
public:
   static void RunAll(CMqlTestFramework &tf)
   {
      tf.SetSuiteName("TestFeatureExtractor");
      
      Test_DefaultConstructor_ZeroState(tf);
      Test_MarketSessionCode_EETMapping(tf);
      Test_ExtractFlattenedVector_Uninitialized_ReturnsFalse(tf);
      Test_ReleaseHandles_Idempotent(tf);
   }

private:
   // Scenario 1: Default constructor initializes zero vector size and feature count
   static void Test_DefaultConstructor_ZeroState(CMqlTestFramework &tf)
   {
      CFeatureExtractor extractor;
      tf.AssertEqualInt(extractor.GetTotalVectorSize(), 0, "DefaultConstructor_TotalVectorSizeZero");
      tf.AssertEqualInt(extractor.GetBaseFeatureCount(), 0, "DefaultConstructor_BaseFeatureCountZero");
   }

   // Scenario 2: GetMarketSessionCode correctly maps all 24 hours in EET/EEST Server Time
   static void Test_MarketSessionCode_EETMapping(CMqlTestFramework &tf)
   {
      CFeatureExtractor extractor;
      
      // Hour 0, 1 -> 0.0f (Sydney)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(0), 0.0, 1e-4, "SessionCode_Hour0_Sydney");
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(1), 0.0, 1e-4, "SessionCode_Hour1_Sydney");
      
      // Hour 2 to 8 -> 1.0f (Sydney + Tokyo overlap)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(2), 1.0, 1e-4, "SessionCode_Hour2_SydTokyo");
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(8), 1.0, 1e-4, "SessionCode_Hour8_SydTokyo");
      
      // Hour 9 -> 2.0f (Tokyo pure)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(9), 2.0, 1e-4, "SessionCode_Hour9_Tokyo");
      
      // Hour 10 -> 3.0f (Tokyo + London overlap)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(10), 3.0, 1e-4, "SessionCode_Hour10_TokyoLondon");
      
      // Hour 11 to 14 -> 4.0f (London pure)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(11), 4.0, 1e-4, "SessionCode_Hour11_London");
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(14), 4.0, 1e-4, "SessionCode_Hour14_London");
      
      // Hour 15 to 18 -> 5.0f (London + New York overlap - Peak global liquidity)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(15), 5.0, 1e-4, "SessionCode_Hour15_LondonNY");
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(18), 5.0, 1e-4, "SessionCode_Hour18_LondonNY");
      
      // Hour 19 to 22 -> 6.0f (New York pure)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(19), 6.0, 1e-4, "SessionCode_Hour19_NY");
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(22), 6.0, 1e-4, "SessionCode_Hour22_NY");
      
      // Hour 23 -> 7.0f (New York close + Sydney open)
      tf.AssertEqualDouble(extractor.GetMarketSessionCode(23), 7.0, 1e-4, "SessionCode_Hour23_NYSydney");
   }

   // Scenario 3: ExtractFlattenedVector returns false if uninitialized
   static void Test_ExtractFlattenedVector_Uninitialized_ReturnsFalse(CMqlTestFramework &tf)
   {
      CFeatureExtractor extractor;
      vectorf vec;
      bool ok = extractor.ExtractFlattenedVector(0, vec);
      tf.AssertFalse(ok, "ExtractFlattenedVector_Uninitialized_ReturnsFalse");
   }

   // Scenario 4: ReleaseHandles is idempotent and safe to call multiple times
   static void Test_ReleaseHandles_Idempotent(CMqlTestFramework &tf)
   {
      CFeatureExtractor extractor;
      extractor.ReleaseHandles();
      extractor.ReleaseHandles();
      extractor.ReleaseHandles();
      tf.AssertEqualInt(extractor.GetTotalVectorSize(), 0, "ReleaseHandles_IdempotentSafe");
   }
};
