---
name: code-quality-auditor
description: Audits Python and MQL5 code for Clean Code, SonarQube rules, low coupling, high cohesion, Flake8 compliance, and memory management.
---

# Code Quality & Clean Architecture Audit Runbook

Use this skill to audit codebase cleanliness, architectural decoupling, and linting compliance.

## Audit Checklist

1. **Python Quality Standards**:
   - Run static analysis:
     ```powershell
     flake8 src/ run_pipeline.py tests/ --max-line-length=120
     ```
   - Verify zero unused imports, zero global mutable state, and `@dataclass(frozen=True)` configuration immutability.
   - Ensure comprehensive English docstrings on all classes and public functions.
2. **MQL5 Quality Standards**:
   - Verify strict encapsulation (private attributes `m_`, public accessors).
   - Verify explicit memory cleanup (`ArrayFree()`, `ReleaseHandles()`) in destructors and deinitialization hooks.
   - Ensure defensive bounds checking on all rates and feature vector reads.
3. **Architectural Decoupling**:
   - Ensure business/quantitative domain logic is decoupled from terminal infrastructure (`MT5Client`, `ScopedCleaner`).
4. **Static Security Analysis**:
   - Verify zero hardcoded secrets or plaintext broker credentials in code.
   - Enforce secure subprocess execution with strictly validated argument lists (no raw shell string injections).
