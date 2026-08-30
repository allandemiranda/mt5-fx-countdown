//+------------------------------------------------------------------+
//|                                       RunAllMQL5UnitTests.mq5    |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|               Master Test Runner for MQL5 Native Unit Tests      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs

#include <Tests/MqlTestFramework.mqh>
#include <Tests/TestGarchEngine.mqh>
#include <Tests/TestOrderTracker.mqh>
#include <Tests/TestFeatureExtractor.mqh>
#include <Tests/TestConsecutiveManager.mqh>

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("================================================================================");
   Print("[MQL5 TEST RUNNER] Starting Native MQL5 Unit Test Suite Execution...");
   Print("================================================================================");
   
   CMqlTestFramework tf;
   
   // 1. Run CGarchEngine Unit Tests
   CTestGarchEngine::RunAll(tf);
   
   // 2. Run COrderTracker Unit Tests
   CTestOrderTracker::RunAll(tf);
   
   // 3. Run CFeatureExtractor Unit Tests
   CTestFeatureExtractor::RunAll(tf);
   
   // 4. Run CConsecutiveManager Unit Tests
   CTestConsecutiveManager::RunAll(tf);
   
   // Print Grand Summary
   tf.PrintSummary();
   
   if(tf.GetFailed() > 0)
   {
      PrintFormat("[MQL5 TEST RUNNER] [RESULT: FAILED] %d assertion(s) failed out of %d.",
                  tf.GetFailed(), tf.GetTotal());
   }
   else
   {
      PrintFormat("[MQL5 TEST RUNNER] [RESULT: ALL PASSED] 100%% assertions passed (%d/%d).",
                  tf.GetPassed(), tf.GetTotal());
   }
}
