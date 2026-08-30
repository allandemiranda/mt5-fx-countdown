# Quantitative & Architectural Documentation (`docs/`)

This directory contains the in-depth technical specifications, mathematical derivations, lifecycle diagrams, and operational guides for the MT5 MLOps Pipeline.

---

## 📚 Documentation Index

1. 📖 **[`ARCHITECTURE.md`](ARCHITECTURE.md)**:
   - **Executive System Architecture**: End-to-end data contracts and modular boundaries.
   - **Econometric Modeling**: Bollerslev (1986) GARCH(1,1) analytical multi-step volatility formulation, covariance stationarity constraints, and dynamic risk mapping.
   - **Feature Engineering**: Complete schema for the 14 feature groups (26 base features, 130 total dimensions with lookback=4) and sequential lag lookback flattening.
   - **Order Tracking & Golden Rule**: In-memory ticket tracking in RAM, `OnTradeTransaction` event capture, and net liquid profit labeling rule.
   - **Dual XGBoost ML Subsystem**: Chronological validation split, Optuna Bayesian optimization, early stopping regularization, and execution telemetry.
   - **ONNX Graph Contract**: Pure 1D Float tensor graphs without ZipMap operators for sub-millisecond execution.
   - **Native Presets**: `.set` preset generation and multi-directory deployment matrix.

2. 🗺️ **[`FLOWCHART.md`](FLOWCHART.md)**:
   - Exhaustive Mermaid flowcharts mapping the 7 quantitative lifecycle stages.
   - Detailed component diagrams for Python MLOps Orchestrator, `DMatrix-EA.mq5`, and `LiveONNX-EA.mq5`.

3. 🚀 **[`MLOPS_PIPELINE_GUIDE.md`](MLOPS_PIPELINE_GUIDE.md)**:
   - Operational guide for researchers and algorithmic traders.
   - Complete reference table for all configuration parameters in `.env`.
   - Bayesian Optuna hyperparameter optimization guide.
   - Serverless DuckDB quantitative analytics and Model Context Protocol (MCP) servers reference.
   - Live MT5 chart deployment steps and troubleshooting FAQ.

4. ⚡ **[`LIVE_ONNX_EA_GUIDE.md`](LIVE_ONNX_EA_GUIDE.md)**:
   - Complete operational and parameter guide for `LiveONNX-EA.mq5`.
   - Detailed breakdown of all inputs, expected ranges (min/max), and quantitative impacts.
   - Support & Resistance (S&R) structural exit engine and safety buffer mechanics.
   - Consecutive signal management policies (`CConsecutiveManager`), hurdle ratchets, anti-chop chain-link anchors, dynamic swap amortization, conflicting signals suppression, and opposing regime ML defense.
   - Strategy Tester optimization benchmarks (Test 1 and 207-pass optimizer sensitivity analysis).

5. 📑 **[`../knowledge_base/OUTPUT_TAXONOMY_AND_EXECUTION_SIGNALS.md`](../knowledge_base/OUTPUT_TAXONOMY_AND_EXECUTION_SIGNALS.md)**:
   - Comprehensive inventory and causal routing of all ecosystem outputs (datasets, models, artifacts, macro DB, and live market orders).
   - Detailed Mermaid flowcharts and causal state transitions from ML probabilities to market fills.
   - Rigorous quantitative and software engineering audit of error-prone logic, unhandled retcodes, rounding errors, and race conditions.

6. 🏛️ **[`../knowledge_base/SYSTEM_ONTOLOGY_AND_DATA_FLOW.md`](../knowledge_base/SYSTEM_ONTOLOGY_AND_DATA_FLOW.md)**:
   - Publication-grade system ontology, class structures, data contracts, and component responsibilities.
   - Causal state machines and cross-boundary data flow across MQL5 and Python.

7. 📊 **[`../knowledge_base/FOREX_MARKET_DYNAMICS_AND_TIMEFRAMES.md`](../knowledge_base/FOREX_MARKET_DYNAMICS_AND_TIMEFRAMES.md)**:
   - Exhaustive market microstructure, continuous 5-day cycle in EET/EEST, and cross-timeframe econometric scaling analysis ($M1$ to $D1$).
   - Noise-to-signal implications for XGBoost regularized trees and shrinkage rates.
   - GARCH(1,1) lookback window and Triple Barrier adaptation across timeframes.
   - G10 major currency pair microstructure profiles, spreads, ADRs, and cross-correlations.
   - Rigorous codebase audit of timeframe scaling bugs and architectural invariants.

8. 📖 **[`../knowledge_base/INPUT_TAXONOMY_AND_IMPACT_MATRIX.md`](../knowledge_base/INPUT_TAXONOMY_AND_IMPACT_MATRIX.md)**:
   - Exhaustive quantitative parameter dictionary and 3-tier sensitivity matrix across all 89 `.env`/`AppConfig` fields, 67 `DMatrix-EA` inputs, and 82 `LiveONNX-EA` inputs.
   - Cross-network impact propagation and dimensional coupling to ONNX graph tensors and risk execution gates.
   - Rigorous codebase parameter audit uncovering critical schema mismatches, stationarity gaps, and edge-case behaviors.

9. ⚖️ **[`../knowledge_base/TRAIN_SERVING_SKEW_AND_PARITY_AUDIT.md`](../knowledge_base/TRAIN_SERVING_SKEW_AND_PARITY_AUDIT.md)**:
   - Exhaustive mathematical and code-level zero train-serving skew proof ($D=130$).
   - 26-feature verification matrix across 5 observation lags and closed-bar GARCH recurrence theorem.
   - Triple Barrier momentum labeling vs dynamic live execution reconciliation and covariate shift governance.

10. 🛡️ **[`../knowledge_base/FMEA_AND_RESILIENCE_ENGINEERING.md`](../knowledge_base/FMEA_AND_RESILIENCE_ENGINEERING.md)**:
    - Formal Failure Mode and Effects Analysis (FMEA, IEC 60812 / SAE J1739) and RPN scoring across all five subsystems.
    - Fault Tree Analysis (FTA) with Minimal Cut Sets for catastrophic account drawdowns, execution freezing, and SQLite locking.
    - Defensive state machines, network disconnect recovery, cold restart protocols, and fail-closed institutional policies.

11. 🔬 **[`../knowledge_base/CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md`](../knowledge_base/CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md)**:
    - Complete master scorecards for McCabe Cyclomatic Complexity, Cognitive Complexity, Halstead metrics, and Maintainability Index across MQL5 and Python.
    - Robert C. Martin Coupling & Modularity matrix ($A, I, D$) and proof of zero heap allocations in `OnTick`.
    - Identification of structural complexity hotspots and modular refactoring blueprints.

12. 📐 **[`../knowledge_base/FORMAL_VERIFICATION_AND_STATE_SPACE.md`](../knowledge_base/FORMAL_VERIFICATION_AND_STATE_SPACE.md)**:
    - Formal Finite State Machine (FSM) models for order lifecycles, macroeconomic subsumption lattice, and MLOps stage transitions.
    - Mathematical invariant proofs: Safety (Stop envelope), Risk ceiling (Bounded loss), Liveness (Finite lifetime), and Deadlock-freedom (SQLite WAL).
    - Exhaustive Boundary Value Analysis (BVA) and Equivalence Partitioning Matrix across all 111 system parameters.


