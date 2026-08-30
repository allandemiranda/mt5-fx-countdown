---
name: doc-sync-specialist
description: Synchronizes and audits project documentation across README.md, docs/, .env.example, and directory READMEs upon feature changes.
---

# Continuous Documentation Synchronization Runbook

Use this skill whenever a feature, parameter, or architectural component is added, modified, or removed.

## Synchronization Checklist

1. **Environment Parameters**:
   - Verify 100% parity between `.env` and `.env.example` (exact key count and comments).
   - Update parameter table in `docs/MLOPS_PIPELINE_GUIDE.md`.
2. **Architecture & Flowcharts**:
   - Update mathematical specifications in `docs/ARCHITECTURE.md` if formulas change.
   - Update Mermaid diagrams in `docs/FLOWCHART.md`.
3. **Mandatory Directory README Updates**:
   - Every time files inside a directory are modified, created, or deleted, its corresponding local `README.md` MUST be updated immediately:
     - `src/README.md`: Update module table, parameter counts, and component descriptions.
     - `MQL5/Experts/README.md`: Update EA workflows, error handling, and parameter interactions.
     - `MQL5/Include/README.md`: Update library classes, mathematical contracts, and log categories.
     - `MQL5/README.md`: Update top-level MQL5 structure and synchronization instructions.
     - `tests/README.md`: Update the Test Matrix table with any newly added test files or updated invariants.
     - `docs/README.md`: Update index of guides and architectural references.
4. **Master Documentation & Rule Files**:
   - Ensure `README.md` and `AGENTS.md` remain concise, authoritative, and 100% synchronized with the codebase.
