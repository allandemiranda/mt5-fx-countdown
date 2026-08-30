---
name: parity-contract-guardian
description: Enforces zero train-serving skew between MT5 data collection (DMatrix-EA), Python training (src/trainer.py), and live execution (LiveONNX-EA).
---

# Train-Serving Parity & Contract Verification Runbook

Use this skill when modifying feature extraction, GARCH risk parameters, or ONNX export logic.

## Verification Steps

1. **Feature Extraction Parity**:
   - Verify `CFeatureExtractor::GetCSVHeader()` order matches `ExtractFlattenedVector()` lag order ($h=0..H$).
   - Verify dimension formula:
     $$\text{Total Features} = N_{\text{active\_base\_features}} \times (\text{FEATURE\_LOOKBACK} + 1)$$
2. **Risk Model Parity**:
   - Ensure both `DMatrix-EA` and `LiveONNX-EA` calculate dynamic risk strictly via `CGarchEngine` ($K_{\text{TP}} \cdot \sigma_{\text{agg}}$ and $K_{\text{SL}} \cdot \sigma_{\text{agg}}$).
   - Ensure zero static pip stops or fixed point modes exist.
3. **ONNX Graph Contract**:
   - Input: `float_input` `[None, num_features]`.
   - Output: `probabilities` `[None, 2]` (Float32, no ZipMap).
4. **Run Parity Tests**:
   ```powershell
   pytest tests/test_feature_schema.py tests/test_onnx_pipeline.py -v
   ```
