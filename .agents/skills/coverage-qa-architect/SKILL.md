---
name: coverage-qa-architect
description: Designs and executes exhaustive test suites in tests/ ensuring 100% business scenario, line, and branch coverage without regressions.
---

# Business Scenario & Code Coverage Test Runbook

Use this skill to design test matrices, verify edge cases, and ensure 100% test coverage.

## Testing Protocol

1. **Business Scenario Coverage**:
   - Map all business rules from plans into explicit test cases (Net Liquid Profit $\le 0.0 \implies 0.0f$, broker stop clamping, chronological sorting).
2. **Method, Line & Branch Coverage**:
   - Exercise all truthy/falsy branches, error handling blocks, and boundary conditions.
3. **Execute Full Test Suite**:
   ```powershell
   pytest tests/ -v
   ```
4. **Verify Zero Regressions**:
   - Confirm all test suites pass with 100% success rate (73+ tests).
