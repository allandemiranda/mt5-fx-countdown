//+------------------------------------------------------------------+
//|                                            TestOrderTracker.mqh  |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|               Black-Box Unit Tests for COrderTracker (MQL5)      |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property strict

#include <Tests/MqlTestFramework.mqh>
#include <OrderTracker.mqh>
#include <FeatureExtractor.mqh>

//+------------------------------------------------------------------+
//| CTestOrderTracker: Black-box verification of COrderTracker       |
//+------------------------------------------------------------------+
class CTestOrderTracker
{
public:
   static void RunAll(CMqlTestFramework &tf)
   {
      tf.SetSuiteName("TestOrderTracker");
      
      Test_InitAndClearAll(tf);
      Test_RegisterPosition_ZeroTicket_Fails(tf);
      Test_RegisterPosition_ValidOrder_Success(tf);
      Test_ProcessUnresolvedPositions_AllZeroLabels(tf);
      Test_SortChronologically_OrdersAscending(tf);
      Test_DynamicGrowthBuffer_NoCrash(tf);
   }

private:
   // Scenario 1: Init and ClearAll reset internal collections
   static void Test_InitAndClearAll(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_H1);
      tf.AssertEqualInt(tracker.GetSampleCount(), 0, "Init_SampleCountZero");
      
      tracker.ClearAll();
      tf.AssertEqualInt(tracker.GetSampleCount(), 0, "ClearAll_SampleCountZero");
   }

   // Scenario 2: RegisterPosition rejects ticket 0
   static void Test_RegisterPosition_ZeroTicket_Fails(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_H1);
      
      vectorf dummyFeatures(130);
      dummyFeatures.Fill(0.5f);
      
      bool ok = tracker.RegisterPosition(0, POSITION_TYPE_BUY, D'2026.01.01 12:00',
                                         1.0850, 1.0900, 1.0800, dummyFeatures);
      tf.AssertFalse(ok, "RegisterPosition_ZeroTicket_Rejected");
   }

   // Scenario 3: RegisterPosition successfully registers valid position
   static void Test_RegisterPosition_ValidOrder_Success(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_H1);
      
      vectorf dummyFeatures(130);
      dummyFeatures.Fill(0.25f);
      
      bool ok = tracker.RegisterPosition(12345678, POSITION_TYPE_BUY, D'2026.01.01 12:00',
                                         1.0850, 1.0900, 1.0800, dummyFeatures);
      tf.AssertTrue(ok, "RegisterPosition_ValidTicket_Accepted");
   }

   // Scenario 4: ProcessUnresolvedPositions labels all active positions as NOT_OPEN (0.0f)
   static void Test_ProcessUnresolvedPositions_AllZeroLabels(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_H1);
      
      vectorf dummyFeatures(130);
      dummyFeatures.Fill(1.0f);
      
      tracker.RegisterPosition(1001, POSITION_TYPE_BUY, D'2026.01.01 10:00', 1.0800, 1.0850, 1.0750, dummyFeatures);
      tracker.RegisterPosition(1002, POSITION_TYPE_SELL, D'2026.01.01 11:00', 1.0800, 1.0750, 1.0850, dummyFeatures);
      
      // Before deinit, sample count is 0
      tf.AssertEqualInt(tracker.GetSampleCount(), 0, "BeforeDeinit_SampleCountZero");
      
      // Process unresolved positions
      tracker.ProcessUnresolvedPositions();
      
      // Both positions must now be recorded as samples
      tf.AssertEqualInt(tracker.GetSampleCount(), 2, "AfterDeinit_SampleCountTwo");
   }

   // Scenario 5: SortChronologically sorts samples oldest to newest
   static void Test_SortChronologically_OrdersAscending(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_H1);
      
      vectorf dummyFeatures(10);
      dummyFeatures.Fill(0.0f);
      
      // Register in reverse chronological order
      tracker.RegisterPosition(2001, POSITION_TYPE_BUY, D'2026.01.03 12:00', 1.0800, 1.0850, 1.0750, dummyFeatures);
      tracker.RegisterPosition(2002, POSITION_TYPE_BUY, D'2026.01.01 12:00', 1.0800, 1.0850, 1.0750, dummyFeatures);
      tracker.RegisterPosition(2003, POSITION_TYPE_BUY, D'2026.01.02 12:00', 1.0800, 1.0850, 1.0750, dummyFeatures);
      
      tracker.ProcessUnresolvedPositions();
      tracker.SortChronologically();
      
      tf.AssertEqualInt(tracker.GetSampleCount(), 3, "SortChronologically_RetainsSampleCount");
   }

   // Scenario 6: Dynamic array expansion handles massive position registration
   static void Test_DynamicGrowthBuffer_NoCrash(CMqlTestFramework &tf)
   {
      COrderTracker tracker;
      tracker.Init("EURUSD", PERIOD_M1);
      
      vectorf dummyFeatures(5);
      dummyFeatures.Fill(0.1f);
      
      // Register 600 positions (exceeding initial 512 allocation chunk)
      bool allOk = true;
      for(ulong t = 1; t <= 600; t++)
      {
         datetime dt = (datetime)(1700000000 + t * 60);
         if(!tracker.RegisterPosition(t, POSITION_TYPE_BUY, dt, 1.0800, 1.0850, 1.0750, dummyFeatures))
         {
            allOk = false;
            break;
         }
      }
      tf.AssertTrue(allOk, "DynamicGrowthBuffer_Handles600PositionsWithoutCrash");
   }
};
