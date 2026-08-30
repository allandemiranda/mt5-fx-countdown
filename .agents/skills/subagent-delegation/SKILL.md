---
name: subagent-delegation
description: Guidelines for decomposing complex quantitative tasks, multi-file refactoring, and test audits across specialized autonomous subagents.
---

# Subagent Task Decomposition & Delegation

Use this skill when tackling multi-stage features, extensive refactors, or parallel verification.

## Subagent Specializations

1. **`mql5_engineer`**:
   - Focus: MQL5/C++ indicators, memory management (`ArrayFree`, `ReleaseHandles`), `OnTradeTransaction` event dispatching, and `CTrade` order execution.
2. **`python_mlops_engineer`**:
   - Focus: `src/` modules, Optuna Bayesian tuning, XGBoost training loss minimization, and ONNX graph pruning.
3. **`code_quality_auditor`**:
   - Focus: SonarQube & Clean Code compliance, static analysis (`flake8`), low coupling, high cohesion, SOLID principles, and English docstring completeness.
4. **`parity_contract_guardian`**:
   - Focus: Mathematical & tensor parity verification between the data generator (`DMatrix-EA`), Python trainer (`src/trainer.py`), and live executor (`LiveONNX-EA`). Enforces zero train-serving skew in `CFeatureExtractor` and dynamic GARCH risk parity.
5. **`doc_specialist`**:
   - Focus: Continuous documentation synchronization, maintaining 100% parity across `README.md`, `docs/ARCHITECTURE.md`, `docs/FLOWCHART.md`, `docs/MLOPS_PIPELINE_GUIDE.md`, `.env.example`, and directory `README.md` files.
6. **`coverage_qa_architect`**:
   - Focus: Translates business rules, execution plans, and mathematical specifications into exhaustive test scenario matrices. Writes unit and integration tests in `tests/` enforcing 100% business scenario, method, line, and branch coverage, guaranteeing zero regressions.
7. **`research_specialist`**:
   - Focus: Deep research in financial literature (López de Prado, Bollerslev, Chen & Guestrin), econometric derivations, and Forex market microstructure.
8. **`economic_calendar_auditor`**:
   - Focus: Cross-references MT5 backtest drawdowns, trade losses, and volatility anomalies against historical macroeconomic releases (NFP, FOMC, CPI, central bank rate decisions) to distinguish between exogenous news shocks and structural model degradation.

## When to Delegate vs. When to Execute Directly

- **Delegate to Subagents**: Complex, multi-subsystem features (e.g. coordinated changes spanning MQL5, Python MLOps, multiple test suites, and documentation), architectural refactoring, or independent parallel verification.
- **Execute Directly (Primary Agent)**: Single-file tweaks, localized bugfixes, routine log updates, configuration file adjustments, or documentation-only changes. Avoid subagent dispatch overhead for simple, sequential tasks.

## Context-Aware Scoped Verification

- **No Python Changes**: If no Python files were touched, NEVER execute the full Python test suite (`pytest tests/ -v`) or Python linters.
- **MQL5-Only Changes**: Only compile via MetaEditor CLI (`python run_pipeline.py --compile-only`). Run specific Python tests only if a shared math/schema contract was modified (e.g., `tests/test_garch_math.py`). Never run the entire 136+ test suite for MQL5-only changes.
- **Python-Only Changes**: Run only relevant Python tests/linters; do not trigger MQL5 compilation unless MQL5 files were altered.
- **Documentation / Config Only**: Do not execute test suites or compilers unless validating that specific config file (e.g. `test_config.py`).

## Delegation Protocol

1. Define subagent role, clear prompt, and tool requirements via `define_subagent` or `invoke_subagent`.
2. Specify exact file boundaries and quantitative contracts.
3. Combine subagent outputs and verify with context-aware, scoped verification.
