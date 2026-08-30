---
name: research-specialist
description: Conducts rigorous research in financial econometrics, Forex market microstructure, and gradient boosting literature (López de Prado, Bollerslev).
---

# Quantitative Research & Literature Foundations Runbook

Use this skill when investigating econometric models, volatility forecasting, or machine learning improvements.

## Research Guidelines

1. **Financial Econometrics**:
   - Validate covariance stationarity ($lpha + eta < 1.0$) for GARCH(1,1).
   - Consult Bollerslev (1986) and Tsay (2010) for volatility clustering formulations.
2. **Financial Machine Learning (AFML)**:
   - Consult Marcos López de Prado (2018, 2020) for non-stationarity, labeling rules, and eliminating lookahead bias.
   - Enforce chronological validation (`VALIDATION_PERCENTAGE`) without random shuffling.
3. **Gradient Boosting Foundations**:
   - Consult Chen & Guestrin (2016) for XGBoost regularized tree loss, split finding, and shrinkage learning rates.
