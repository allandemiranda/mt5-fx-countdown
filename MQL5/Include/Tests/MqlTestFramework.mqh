//+------------------------------------------------------------------+
//|                                           MqlTestFramework.mqh   |
//|                   Copyright 2026, Institutional Quantitative QA  |
//|               Lightweight Unit Testing Framework for MQL5 Native |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quantitative QA"
#property link      "https://www.mql5.com"
#property strict

//+------------------------------------------------------------------+
//| CMqlTestFramework: Manages assertions, test logging, and stats   |
//+------------------------------------------------------------------+
class CMqlTestFramework
{
private:
   int m_totalTests;
   int m_passedTests;
   int m_failedTests;
   string m_currentSuite;

public:
   CMqlTestFramework()
      : m_totalTests(0),
        m_passedTests(0),
        m_failedTests(0),
        m_currentSuite("TestSuite")
   {
   }

   void SetSuiteName(const string suiteName)
   {
      m_currentSuite = suiteName;
      PrintFormat("[TEST SUITE] >>> Starting Suite: %s <<<", m_currentSuite);
   }

   void Reset()
   {
      m_totalTests  = 0;
      m_passedTests = 0;
      m_failedTests = 0;
   }

   int GetTotal()  const { return m_totalTests; }
   int GetPassed() const { return m_passedTests; }
   int GetFailed() const { return m_failedTests; }

   bool AssertTrue(bool condition, const string testName, const string message = "")
   {
      m_totalTests++;
      if(condition)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s", m_currentSuite, testName);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Condition is FALSE. %s", m_currentSuite, testName, message);
         return false;
      }
   }

   bool AssertFalse(bool condition, const string testName, const string message = "")
   {
      return AssertTrue(!condition, testName, message != "" ? message : "Expected condition to be FALSE, but was TRUE.");
   }

   bool AssertEqualInt(long actual, long expected, const string testName, const string message = "")
   {
      m_totalTests++;
      if(actual == expected)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s (Value: %I64d)", m_currentSuite, testName, actual);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Expected: %I64d, Actual: %I64d. %s",
                     m_currentSuite, testName, expected, actual, message);
         return false;
      }
   }

   bool AssertEqualDouble(double actual, double expected, double epsilon, const string testName, const string message = "")
   {
      m_totalTests++;
      double diff = MathAbs(actual - expected);
      if(diff <= epsilon)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s (Actual: %.6f, Expected: %.6f, Diff: %.6e)",
                     m_currentSuite, testName, actual, expected, diff);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Actual: %.6f != Expected: %.6f (Diff: %.6f > Eps: %.6f). %s",
                     m_currentSuite, testName, actual, expected, diff, epsilon, message);
         return false;
      }
   }

   bool AssertEqualString(const string actual, const string expected, const string testName, const string message = "")
   {
      m_totalTests++;
      if(actual == expected)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s (String match: '%s')", m_currentSuite, testName, actual);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Expected: '%s', Actual: '%s'. %s",
                     m_currentSuite, testName, expected, actual, message);
         return false;
      }
   }

   bool AssertGreater(double actual, double threshold, const string testName, const string message = "")
   {
      m_totalTests++;
      if(actual > threshold)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s (%.6f > %.6f)", m_currentSuite, testName, actual, threshold);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Expected %.6f > %.6f. %s",
                     m_currentSuite, testName, actual, threshold, message);
         return false;
      }
   }

   bool AssertLess(double actual, double threshold, const string testName, const string message = "")
   {
      m_totalTests++;
      if(actual < threshold)
      {
         m_passedTests++;
         PrintFormat("  [PASS] %s::%s (%.6f < %.6f)", m_currentSuite, testName, actual, threshold);
         return true;
      }
      else
      {
         m_failedTests++;
         PrintFormat("  [FAIL] %s::%s -> Expected %.6f < %.6f. %s",
                     m_currentSuite, testName, actual, threshold, message);
         return false;
      }
   }

   void PrintSummary()
   {
      Print("================================================================================");
      PrintFormat("[TEST REPORT SUMMARY] Total Tests: %d | Passed: %d | Failed: %d | Success Rate: %.2f%%",
                  m_totalTests, m_passedTests, m_failedTests,
                  m_totalTests > 0 ? ((double)m_passedTests / (double)m_totalTests) * 100.0 : 0.0);
      Print("================================================================================");
   }
};
