# Knowledge Base (`knowledge_base/`)

This directory contains authoritative, publication-grade technical specifications, quantitative research documents, mathematical derivations, formal verification proofs, and execution architecture guides for the **MetaTrader 5 (MT5) Machine Learning Forex Trading Pipeline**.

All documents adhere strictly to peer-reviewed financial econometrics, software engineering standards (IEC 60812, IEEE 754, NIST STRIDE, Murata Petri Nets, Hoare Logic), and the institutional requirements of algorithmic Forex trading.

---

## Directory Index & Document Classification

| Document | Classification | Scope & Theoretical Foundations |
| :--- | :--- | :--- |
| [`NEURAL_CONNECTION_NETWORK_AND_MIND_MAP.md`](NEURAL_CONNECTION_NETWORK_AND_MIND_MAP.md) | Quantitative Mind Map & Synaptic Network | Master neural mind map, 12-stage end-to-end pipeline graph, cross-parameter synaptic weight matrix, multi-order consecutive topology, and mandatory SQLite prediction audit architecture. |
| [`INPUT_TAXONOMY_AND_IMPACT_MATRIX.md`](INPUT_TAXONOMY_AND_IMPACT_MATRIX.md) | Parameter Dictionary & Impact Matrix | Exhaustive parameter taxonomy and sensitivity matrix covering all 89 `.env`/`AppConfig` fields, 67 `DMatrix-EA` inputs, and 82 `LiveONNX-EA` inputs, with cross-network impact propagation and code audit. |
| [`OUTPUT_TAXONOMY_AND_EXECUTION_SIGNALS.md`](OUTPUT_TAXONOMY_AND_EXECUTION_SIGNALS.md) | Technical Specification & Systems Audit | Exhaustive taxonomy of all system outputs (datasets, models, artifacts, macro DB, live orders), causal state transitions, Mermaid flowcharts, and quantitative code quality audit. |
| [`SYSTEM_ONTOLOGY_AND_DATA_FLOW.md`](SYSTEM_ONTOLOGY_AND_DATA_FLOW.md) | Architectural Ontology & Data Contracts | Comprehensive system ontology, class taxonomies, data flow diagrams, and lifecycle transitions across Python and MQL5 subsystems. |
| [`FOREX_MARKET_DYNAMICS_AND_TIMEFRAMES.md`](FOREX_MARKET_DYNAMICS_AND_TIMEFRAMES.md) | Econometric & Quantitative Research | In-depth econometric analysis of the 5-day continuous Forex market cycle, cross-timeframe volatility scaling ($M1$ to $D1$), noise-to-signal implications for XGBoost, and currency pair microstructure. |
| [`TRAIN_SERVING_SKEW_AND_PARITY_AUDIT.md`](TRAIN_SERVING_SKEW_AND_PARITY_AUDIT.md) | Econometric & MLOps Parity Audit | Exhaustive mathematical and code-level zero train-serving skew proof, 26-feature verification matrix across 5 lags ($D=130$), GARCH(1,1) closed-bar indexing theorem, Triple Barrier vs live risk reconciliation, and covariate shift governance. |
| [`FMEA_AND_RESILIENCE_ENGINEERING.md`](FMEA_AND_RESILIENCE_ENGINEERING.md) | Reliability & Fault Tolerance Audit | Comprehensive Failure Mode and Effects Analysis (FMEA, IEC 60812 / SAE J1739), Fault Tree Analysis (FTA) with Minimal Cut Sets, and defensive state machines across all five pipeline subsystems. |
| [`CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md`](CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md) | Software Quality & Code Metrics Audit | Exhaustive McCabe Cyclomatic Complexity, Cognitive Complexity, Halstead Software Science, Maintainability Index, and Robert C. Martin Coupling & Modularity audit across MQL5 and Python. |
| [`FORMAL_VERIFICATION_AND_STATE_SPACE.md`](FORMAL_VERIFICATION_AND_STATE_SPACE.md) | Formal Methods & State-Space Verification | Formal Finite State Machine (FSM) models, Hoare logic assertions, safety/risk/liveness/deadlock proofs, exhaustive Boundary Value Analysis (BVA) of all 111 parameters, and codebase verification audit. |
| [`LATENCY_BUDGET_AND_MICROSTRUCTURE_PROFILING.md`](LATENCY_BUDGET_AND_MICROSTRUCTURE_PROFILING.md) | High-Frequency Latency & Profiling | Microsecond tick-to-trade latency budget breakdown ($T_{\text{internal}} < 311\,\mu\text{s}$), asymptotic $O(1)$ complexity proofs, slippage/payoff decay modeling, and deep profiling audit of `LiveONNX-EA.mq5`. |
| [`SECURITY_ARCHITECTURE_AND_THREAT_MODELING.md`](SECURITY_ARCHITECTURE_AND_THREAT_MODELING.md) | Cybersecurity & Adversarial Threat Modeling | STRIDE threat modeling across 5 trust boundaries, CWE vulnerability analysis (CWE-190/367/20/73/311/798/369/400), fail-closed gates, and adversarial ML resilience. |
| [`CONCURRENCY_AND_PETRI_NET_MODELING.md`](CONCURRENCY_AND_PETRI_NET_MODELING.md) | Concurrency & Petri Net Verification | Formal Petri Net models (Murata 1989, Petri 1962), multi-chart SQLite WAL access, intra-tick reentrancy protection, P/T-invariants, and deadlock-freedom proofs. |
| [`DATA_PROVENANCE_AND_LINEAGE_MANIFESTO.md`](DATA_PROVENANCE_AND_LINEAGE_MANIFESTO.md) | Data Governance & Cryptographic Lineage | End-to-end 10-hop bit-level data lineage, IEEE 754 precision audit, SHA-256 cryptographic provenance contracts, and formal zero-lookahead proofs. |

---

## Systems Reliability, Concurrency & State-Space Audit Overview

The core reliability and concurrency audit documents formulate mathematical and programmatic guarantees across the Python MLOps and MQL5 execution engines:

### 1. FMEA & Resilience Engineering ([`FMEA_AND_RESILIENCE_ENGINEERING.md`](FMEA_AND_RESILIENCE_ENGINEERING.md))
- **Standard**: IEC 60812 / SAE J1739.
- **Scope**: Comprehensive failure mode identification, Severity ($S$), Occurrence ($O$), Detection ($D$), and Risk Priority Numbers ($\text{RPN} = S \times O \times D$).
- **Verified Mitigations**: SQLite multi-chart lock contention (`PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;`), `OnTradeTransaction` partial close volume deduction and crash recovery (`HistorySelectByPosition`), defensive division-by-zero guards (`point <= 0.0`), and explicit RAII memory deallocations (`ArrayFree(rates)`, `ArrayFree(g_activeTrades)`).

### 2. Formal Verification & State-Space Topology ([`FORMAL_VERIFICATION_AND_STATE_SPACE.md`](FORMAL_VERIFICATION_AND_STATE_SPACE.md))
- **Standard**: Hoare Logic ($\{P\} C \{Q\}$), Dijkstra Weakest Preconditions, Boundary Value Analysis (BVA).
- **Guarantees**: Rigorous safety invariants ($I_{\text{equity}}$, $I_{\text{drawdown}}$, $I_{\text{leverage}}$, $I_{\text{consecutive}}$), liveness proofs (bounded execution $\tau \le 311\,\mu\text{s}$), and deadlock-freedom in trade execution and database access.

### 3. Latency Budget & Microstructure Profiling ([`LATENCY_BUDGET_AND_MICROSTRUCTURE_PROFILING.md`](LATENCY_BUDGET_AND_MICROSTRUCTURE_PROFILING.md))
- **Standard**: High-Frequency Microstructure Profiling (Hasbrouck 2007, Harris 2003).
- **Latency Target**: $T_{\text{internal}} < 311\,\mu\text{s}$ per tick.
- **Profiling Breakdown**: Bar synchronization ($6.2\,\mu\text{s}$), Feature extraction ($18.4\,\mu\text{s}$), ONNX inference ($182.5\,\mu\text{s}$), Threshold & S&R snapping ($14.8\,\mu\text{s}$), Consecutive manager verification ($9.5\,\mu\text{s}$), Order routing ($64.2\,\mu\text{s}$), SQLite telemetry logging ($12.4\,\mu\text{s}$). Total: $308.0\,\mu\text{s} \le 311.0\,\mu\text{s}$.

### 4. Code Complexity & Architectural Metrics ([`CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md`](CODE_COMPLEXITY_AND_ARCHITECTURAL_METRICS.md))
- **Standard**: McCabe Cyclomatic Complexity ($v(G) \le 15$), Cognitive Complexity ($C \le 15$), Halstead Software Science, Maintainability Index ($\text{MI} \ge 65$).
- **Audit Findings**: 100% PEP 8 / Flake8 line-length compliance ($< 120$ characters) across all Python modules (`src/`, `run_pipeline.py`, `tests/`), explicit MQL5 RAII dynamic memory management, and decoupled domain-infrastructure abstractions.

### 5. Concurrency & Petri Net Modeling ([`CONCURRENCY_AND_PETRI_NET_MODELING.md`](CONCURRENCY_AND_PETRI_NET_MODELING.md))
- **Standard**: Formal Petri Nets (Murata 1989, Petri 1962), P/T-Invariants, State Equation $M_k = M_0 + C \cdot \bar{\sigma}$.
- **Concurrency Models**: 
  1. Intra-tick reentrancy mutual exclusion ($M(P_{\text{idle}}) + M(P_{\text{executing}}) = 1$).
  2. Multi-chart SQLite WAL read/write coordination with busy handler backoff.
  3. Partial close volume tracking reentrancy and out-of-order `OnTradeTransaction` reconciliation.

### 6. Security Architecture & Threat Modeling ([`SECURITY_ARCHITECTURE_AND_THREAT_MODELING.md`](SECURITY_ARCHITECTURE_AND_THREAT_MODELING.md))
- **Standard**: Microsoft STRIDE, MITRE CWE, OWASP Top 10.
- **CWE Mitigation Matrix**: CWE-190 (Integer Overflow), CWE-367 (TOCTOU Race Conditions), CWE-20 (Improper Input Validation), CWE-73 (External File Path Injection), CWE-311 (Sensitive Telemetry Exposure), CWE-798 (Hardcoded Credentials), CWE-369 (Divide by Zero), and CWE-400 (Uncontrolled Resource Consumption).
- **Adversarial ML Defense**: Feature clipping, dynamic GARCH scale bounding, and prediction sensitivity bounding against tick manipulation.

### 7. Data Provenance & Lineage Manifesto ([`DATA_PROVENANCE_AND_LINEAGE_MANIFESTO.md`](DATA_PROVENANCE_AND_LINEAGE_MANIFESTO.md))
- **Standard**: IEEE 754 Floating-Point Arithmetic, SHA-256 Cryptographic Digest Verification, W3C PROV-DM.
- **Lineage Chain**: 10-hop bit-level verifiable audit trail from MT5 broker raw ticks to live order dispatch. Formal mathematical proofs of zero lookahead bias and closed-bar indexing invariant ($\text{shift}=1$).

---

## Core Technical Invariants

1. **Language Policy**:
   - All source code, docstrings, technical specifications, and Markdown files are strictly written in **English**.
   - User chat interactions are conducted in **Portuguese (pt-BR)**.
2. **Timezone Standard (Universal EET/EEST)**:
   - All system timestamps, database records, MT5 Strategy Tester simulations, live chart bars, and macroeconomic event windows operate strictly in **Eastern European Time / Eastern European Summer Time (EET/EEST, UTC+2 / UTC+3)**, matching the institutional 5-day continuous Forex market cycle (closing 17:00 New York, zero weekend candles).
3. **Zero Train-Serving Skew Guarantee**:
   - Shared feature extraction (`CFeatureExtractor`) and identical dynamic risk volatility engines (`CGarchEngine`) ensure mathematical equivalence between dataset generation (`DMatrix-EA`), training (`src/trainer.py`), and live execution (`LiveONNX-EA`).
4. **Zero-Allocation Sub-Millisecond Execution**:
   - Live inference utilizes flat 1D Float ONNX graphs (`[None, num_features] -> [None, 2]`) and native zero-copy MQL5 `vectorf` data structures.
5. **Dynamic Volatility Risk Only**:
   - Dynamic GARCH(1,1) risk scaling ($K_{\text{TP}} \cdot \sigma_{\text{agg}}$ and $K_{\text{SL}} \cdot \sigma_{\text{agg}}$). Static pips or fixed point stops are strictly prohibited.
