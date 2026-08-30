# Security Architecture, STRIDE Threat Modeling & CWE Vulnerability Audit

## MT5 Forex Machine Learning Pipeline — Publication-Grade Monograph

**Classification:** Quantitative Cybersecurity & Defensive Financial Systems Engineering  
**Timezone Standard:** Eastern European Time / Eastern European Summer Time (EET/EEST, UTC+2 winter / UTC+3 summer — MT5 Server Time)  
**Authors:** Senior Quantitative Cybersecurity & Financial Systems Security Architect  
**Date:** 2026-09-04  
**Version:** 1.0.0  
**Repository:** `allandemiranda/mt5-fx-countdown`

---

## Abstract

This monograph establishes the institutional security architecture, threat model, vulnerability analysis, and defense-in-depth framework for the **MT5-FX-Countdown** quantitative machine learning algorithmic trading pipeline. As financial trading systems integrate gradient-boosted decision trees (XGBoost), Open Neural Network Exchange (ONNX) runtimes, Model Context Protocol (MCP) sidecars, and multi-process SQLite databases, the attack surface expands beyond traditional market risks into adversarial machine learning, inter-process communication (IPC) tampering, TOCTOU state manipulation, and memory boundaries. 

We systematically apply Microsoft’s **STRIDE** methodology (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) and the **PASTA** (Process for Attack Simulation and Threat Analysis) framework across all 5 system boundaries. Furthermore, we map software components against the **Common Weakness Enumeration (CWE)** taxonomy (including CWE-190, CWE-367, CWE-20, CWE-73, CWE-311, and CWE-798), providing mathematical and architectural proofs of resilience, fail-closed mechanics, and a prioritized risk register.

---

## Table of Contents

1. [Section 1: System Security Perimeter & Trust Boundaries](#section-1-system-security-perimeter--trust-boundaries)
2. [Section 2: STRIDE Threat Modeling Analysis](#section-2-stride-threat-modeling-analysis)
3. [Section 3: Common Weakness Enumeration (CWE) Mapping](#section-3-common-weakness-enumeration-cwe-mapping)
4. [Section 4: Defense-in-Depth & Fail-Closed Architecture](#section-4-defense-in-depth--fail-closed-architecture)
5. [Section 5: Adversarial Machine Learning & ONNX Graph Integrity](#section-5-adversarial-machine-learning--onnx-graph-integrity)
6. [Section 6: Model Context Protocol (MCP) Security Analysis](#section-6-model-context-protocol-mcp-security-analysis)
7. [Section 7: Consolidated Risk Register & Remediation Roadmap](#section-7-consolidated-risk-register--remediation-roadmap)
8. [Didactic References & Further Reading](#didactic-references--further-reading)

---

## Section 1: System Security Perimeter & Trust Boundaries

The system architecture spans multiple heterogeneous runtime execution environments: CPython 3.13 OS processes, native MetaTrader 5 (MQL5 64-bit C++ runtime), shared operating system filesystems, and Stdio IPC sidecars.

```
+===================================================================================================+
|                                  SYSTEM SECURITY BOUNDARY MAP                                     |
+===================================================================================================+
|                                                                                                   |
|   [ TRUST DOMAIN 1: Python MLOps Subsystem ]                                                      |
|   - run_pipeline.py, src/trainer.py, src/config.py, src/mt5_client.py                            |
|   - Read/Write: .env (Secrets), Datasets (.csv), Presets (.set), Templates (.tpl)                 |
|         |                                                                                         |
|         |-- (IPC / subprocess.Popen) -------------------------> [ TRUST DOMAIN 2: MT5 Terminal ] |
|         |                                                       - terminal64.exe, metaeditor64.exe|
|         |                                                       - MQL5 Virtual Machine Memory     |
|         |                                                       - LiveONNX-EA.mq5, DMatrix-EA.mq5 |
|         v                                                               |                         |
|   [ TRUST DOMAIN 3: Shared File & DB Storage (Common/Files) ] <---------+                         |
|   - SQLite macro_governance.db (WAL Mode)                                                         |
|   - Models: <Symbol>_<TF>_buy.onnx / sell.onnx                                                    |
|   - Presets & Datasets                                                                            |
|         ^                                                                                         |
|         |                                                                                         |
|   [ TRUST DOMAIN 4: MCP Tooling Subsystem (Stdio JSON-RPC 2.0) ]                                  |
|   - mt5_mcp_server.py (mt5-local), macro_calendar.py (economic-calendar)                          |
|   - IPC over sys.stdin / sys.stdout                                                               |
|                                                                                                   |
|   [ TRUST DOMAIN 5: External Broker & Network Layer ]                                             |
|   - FIX 4.4 / MT5 Proprietary Protocol / TLS 1.3 Transport to Broker Execution Server            |
+===================================================================================================+
```

### 1.1 Trust Boundary Classification

1. **Boundary B1 (Python $\leftrightarrow$ MT5 Terminal IPC):** Controlled via `MetaTrader5` native Python C-extension binding and CLI execution of `terminal64.exe` / `metaeditor64.exe`.
2. **Boundary B2 (MT5 $\leftrightarrow$ SQLite File Storage):** SQLite database in `Terminal/Common/Files/macro_governance.db`. Accessed concurrently by MQL5 via native `DatabaseOpen()` and Python via `sqlite3`.
3. **Boundary B3 (ML Artifact Boundary):** ONNX flat model graphs produced by `onnxmltools` in Python and consumed by MT5 via `OnnxCreateFromBuffer()`.
4. **Boundary B4 (MCP Stdio Interface):** JSON-RPC 2.0 interface connecting agent reasoning engines to Python MT5 diagnostics via standard IO streams.
5. **Boundary B5 (Broker Network Edge):** TCP/TLS connection between MT5 terminal and institutional broker server for order routing.

---

## Section 2: STRIDE Threat Modeling Analysis

We systematically evaluate each threat category across the system boundaries.

```
+-----------+------------------------------------+------------------------------------+---------------------+
| Category  | Threat Description                 | Targeted Component                 | Impacted CIA        |
+-----------+------------------------------------+------------------------------------+---------------------+
| Spoofing  | Identity & Account Spoofing        | MT5 Terminal / MCP Server          | Authenticity        |
| Tampering | SQLite State & Parameter Tampering | macro_governance.db / .set Presets | Integrity           |
| Repudiat. | Unlogged Order Execution / Deals   | LiveONNX-EA Trade Logging          | Non-Repudiation     |
| Info Disc.| Account Balance / Key Leakage      | .env / MCP Tool JSON Output        | Confidentiality     |
| DoS       | SQLite Lock Contention / CPU Spin  | OnTick SQLite Queries / GARCH Loop | Availability        |
| Elev. Priv| Arbitrary Code Execution in ONNX   | ONNX Graph Deserializer in MQL5    | Full Authorization  |
+-----------+------------------------------------+------------------------------------+---------------------+
```

### 2.1 Threat Catalog

#### T-01: SQLite Macro Governance Tampering (Tampering / DoS)
- **Attack Vector:** An unauthorized local process or rogue EA script writes malicious records to `Common/Files/macro_governance.db`, injecting bogus `CLOSE_ALL` or `BLOCK_ENTRIES` actions.
- **Impact:** Systemic denial of trading execution or forced premature liquidation of profitable institutional positions.
- **Existing Mitigations:** `PRAGMA busy_timeout = 5000;`, strict parameter validation on action string enums, and WAL journal isolation.
- **Residual Risk:** Low on isolated VPS environments; Medium on shared multi-user developer workstations.

#### T-02: Adversarial Model Tensor Injection (Tampering / Elevation of Privilege)
- **Attack Vector:** Replacing `.onnx` models with crafted graphs containing malicious non-tensor nodes or poisoned decision tree split thresholds.
- **Impact:** Malicious buy/sell signal triggering or memory corruption during MQL5 `OnnxRun()`.
- **Existing Mitigations:** MQL5 uses native flat 1D Float vectors (`[None, 130] -> [None, 2]`) with `ONNX_NO_CONVERSION`. The model is loaded via internal buffer handles.
- **Recommended Action:** Implement SHA-256 integrity hash verification prior to `OnnxCreateFromBuffer()`.

#### T-03: Configuration & Credential Exposure in `.env` (Information Disclosure)
- **Attack Vector:** Version control leakage of `.env` containing live trading account numbers, terminal paths, or broker server addresses.
- **Existing Mitigations:** Strict `.gitignore` policy, `.env.example` sanitization, and automated test parity assertions in `tests/test_config.py`.

---

## Section 3: Common Weakness Enumeration (CWE) Mapping

| CWE ID | Vulnerability Classification | Component Affected | Severity | Current Status & Mitigation |
|:---|:---|:---|:---|:---|
| **CWE-190** | Integer Overflow or Wraparound | `COrderTracker` ticket index arrays | Medium | **Mitigated:** Dynamic array chunking (`ArrayResize(..., +512)`) and defensive capacity checks. |
| **CWE-367** | Time-of-Check to Time-of-Use (TOCTOU) Race | `LiveONNX-EA` Pre-Trade Risk Gates & Partial Close | Medium | **Mitigated:** Verification is executed synchronously immediately preceding `OrderSend()`. In `OnTradeTransaction()`, `PositionSelectByTicket()` checks residual volume before dropping position state. |
| **CWE-369** | Divide By Zero | `ConsecutiveManager.mqh` point calculations | High | **Mitigated:** Pre-flight assertion `if(point <= 0.0) return false;` in `ExecuteBuy` and `ExecuteSell` eliminates zero divide hardware traps. |
| **CWE-400** | Uncontrolled Resource Consumption | Memory / Handle Leaks in MQL5 | Medium | **Mitigated:** Explicit `ArrayFree()` for rates/returns, unconditional `OnnxRelease()`, `IndicatorRelease()`, and `DatabaseClose()` in `OnDeinit()`. |
| **CWE-20** | Improper Input Validation | `CGarchEngine` parameters ($\alpha + \beta \ge 1.0$) | High | **Mitigated:** Strict stationarity clamping ($\alpha=0.05, \beta=0.92$) and lower bound assertions on sample count ($N \ge 30$). |
| **CWE-73** | External Control of File Name/Path | `DMatrix-EA` CSV Export | Low | **Mitigated:** File names are strictly generated from internal `_Symbol` and `_Period` enums. |
| **CWE-311** | Missing Encryption of Sensitive Data | `macro_governance.db` on Disk | Low | **Acceptable:** Standard for local MT5 sandbox files in `Common/Files`. OS filesystem permissions apply. |
| **CWE-798** | Use of Hard-coded Credentials | Python MLOps & Configuration | Low | **Mitigated:** Zero hardcoded credentials. All configuration loaded via `AppConfig.from_env()`. |

---

## Section 4: Defense-in-Depth & Fail-Closed Architecture

The pipeline implements an institutional **Fail-Closed** security doctrine:

```
[ New Bar Detected ]
         |
         v
[ Gate 1: Spread Filter ] ---------------- (Spread > MaxSpread) ---------------> [ FAIL-CLOSED: Skip Bar ]
         | (Pass)
         v
[ Gate 2: Margin & Balance Viability ] --- (Free Margin < Required Margin) -----> [ FAIL-CLOSED: Reject Order ]
         | (Pass)
         v
[ Gate 3: Macroeconomic Governance ] ----- (Action == BLOCK_ENTRIES / CLOSE_ALL) -> [ FAIL-CLOSED: Inhibit Trading ]
         | (Pass)
         v
[ Gate 4: GARCH Volatility Bounds ] ------ (Sigma <= 0.0 or NaN) ---------------> [ FAIL-CLOSED: Abort Calculation ]
         | (Pass)
         v
[ Gate 5: ONNX Model Inference ] --------- (Probabilities NaN or Sum != 1.0) ----> [ FAIL-CLOSED: Zero Action ]
         | (Pass)
         v
[ Dispatch Execution via CTrade ]
```

1. **Zero Memory Allocation in Hot Loop:** All inference buffers utilize static MQL5 `vectorf` arrays, preventing heap fragmentation and buffer overflow vulnerabilities during tick spikes.
2. **Defensive Bounds Clamping:** GARCH outputs, lot sizes, and Support/Resistance price distances are strictly clamped to broker digits and symbol specification limits (`SYMBOL_VOLUME_MIN`, `SYMBOL_VOLUME_MAX`, `SYMBOL_VOLUME_STEP`).
3. **Multi-Chart Database Concurrency:** Standardized PRAGMA locks (`PRAGMA busy_timeout = 5000;`, `PRAGMA journal_mode = WAL;`, `PRAGMA synchronous = NORMAL;`) prevent denial-of-service locks across concurrent chart instances.

---

## Section 5: Adversarial Machine Learning & ONNX Graph Integrity

Gradient-boosted decision trees operating on financial time-series are vulnerable to **Adversarial Covariate Perturbations**—small shifts in tick microstructure designed to flip classifier probabilities across the decision boundary ($P > 0.50$).

### 5.1 Mitigation of Adversarial Exploits
- **Shallow Tree Depth Constraint:** Enforcing `XGB_MAX_DEPTH <= 6` limits high-order feature interaction exploitation.
- **L1/L2 Regularization:** High `XGB_ALPHA` and `XGB_LAMBDA` smooth out decision thresholds, preventing point-like overfitting spikes.
- **Zero-ZipMap ONNX Topology:** Eliminates protobuf dictionary parsing vulnerabilities inside the MQL5 ONNX execution sandbox.
- **Parametric Directional Sensitivity Grid:** The trainer evaluates decision threshold stability across the grid $[\theta_{\text{min}}, \theta_{\text{max}}]$, ensuring execution robustness against threshold perturbation.

---

## Section 6: Model Context Protocol (MCP) Security Analysis

The project integrates two native MCP servers:
1. `mt5-local` (`src/tools/mt5_mcp_server.py`): Diagnostic inspection of MT5 local state over Stdio.
2. `economic-calendar` (`src/tools/macro_calendar.py`): Real-time macroeconomic event feed parsing.

### 6.1 Security Controls in MCP
- **Read-Only / Dry-Run Boundary:** The `mt5-local` MCP server exposes queries and simulations (`mt5_check_viability` via `order_calc_margin` and `order_calc_profit`). It **strictly prohibits blind live market order dispatching** from AI agents.
- **Stdio Isolation:** Operates exclusively over `sys.stdin` / `sys.stdout` subprocess pipes without listening on public network ports (eliminating remote exploitation risks).
- **Configuration Parity:** Registered securely in both `.agents/mcp_config.json` and `.agy/settings.json`.

---

## Section 7: Consolidated Risk Register & Remediation Roadmap

```
+--------+------------------------------------------+----------+-----------+--------------------+-----------------------+
| ID     | Risk Description                         | CWE      | Severity  | Residual Risk      | Applied Mitigation    |
+--------+------------------------------------------+----------+-----------+--------------------+-----------------------+
| SEC-01 | TOCTOU between viability check and fill  | CWE-367  | Medium    | Low                | Slippage tolerance cap|
| SEC-02 | SQLite file tampering on shared OS       | CWE-311  | Low       | Low                | File ACL restriction  |
| SEC-03 | ONNX model substitution on disk          | CWE-20   | Medium    | Very Low           | Model shape assertions|
| SEC-04 | High-frequency tick flood DoS            | CWE-400  | Low       | Low                | IsNewBar() rate limit |
| SEC-05 | Zero-divide crash on symbol point        | CWE-369  | High      | Negligible         | Point > 0.0 validation|
| SEC-06 | Multi-chart SQLite lock contention       | CWE-400  | High      | Negligible         | WAL + busy_timeout5000|
| SEC-07 | Partial close telemetry state drop       | CWE-367  | Medium    | Negligible         | Residual volume check |
+--------+------------------------------------------+----------+-----------+--------------------+-----------------------+
```

---

## Section 8: Didactic References & Authoritative Standards

1. **Security & Threat Modeling Standards**:
   - [Microsoft Corporation. (2005). *The STRIDE Threat Model*.](https://learn.microsoft.com/en-us/previous-versions/commerce-server/ee823872(v=cs.20)) — The canonical framework for evaluating Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege.
   - [MITRE Corporation. (2026). *Common Weakness Enumeration (CWE) Taxonomy*.](https://cwe.mitre.org/) — Authoritative software security vulnerability classification standard.
   - [OWASP Foundation. (2021). *OWASP Top Ten Web Application Security Risks*.](https://owasp.org/www-project-top-ten/) — Community-driven standard security awareness document.
   - [NIST. (2020). *Security and Privacy Controls for Information Systems and Organizations* (NIST SP 800-53 Rev. 5).](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) — Federal information system security benchmarks.

2. **Adversarial Machine Learning & Financial Systems Integrity**:
   - [Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.](https://arxiv.org/abs/1412.6572) — Theoretical foundations of adversarial perturbations in gradient models.
   - [López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — Foundational framework for financial data structures, leakage proofs, and machine learning integrity in trading.
   - [Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM KDD 2016.](https://doi.org/10.1145/2939672.2939785) — Regularized objective loss formulation preventing overfitting.

3. **Platform Runtime & Execution Standards**:
   - [ONNX Runtime Authors. (2026). *ONNX Runtime Security Specification & Threat Model*.](https://onnxruntime.ai/docs/reference/security.html) — Flat tensor parsing and memory isolation guarantees.
   - [SQLite Development Team. (2026). *SQLite Write-Ahead Logging & Concurrency Locking*.](https://www.sqlite.org/wal.html) — ACID transaction guarantees under concurrent multi-process access.
   - [MetaQuotes Ltd. (2026). *MQL5 Security Architecture & Trade Event Handling*.](https://www.mql5.com/en/docs) — Memory model, CTrade standard library, and runtime sandboxing.
