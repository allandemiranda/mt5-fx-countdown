# MetaTrader 5 (MQL5) Algorithmic Engine (`MQL5/`)

This directory houses the complete MQL5 quantitative engine, divided into shared econometrics and feature extraction libraries (`Include/`) and specialized Expert Advisors (`Experts/`).

---

## 🏛️ Directory Structure

```
MQL5/
├── Include/
│   ├── FeatureExtractor.mqh   # High-dimensional 13-group feature engineering & lookback flattener
│   ├── GarchEngine.mqh        # Bollerslev (1986) GARCH(1,1) analytical volatility forecasting engine
│   └── OrderTracker.mqh       # In-memory ticket tracking & Golden Rule net profit labeling
└── Experts/
    ├── DMatrix-EA.mq5         # Historical dataset collector EA (Strategy Tester)
    └── LiveONNX-EA.mq5        # Sub-millisecond live inference EA (Zero-copy native vectorf)
```

---

## 🔑 Key Architectural Guarantees

1. **Zero Train-Serving Skew**: `CFeatureExtractor` in `Include/FeatureExtractor.mqh` is shared verbatim between `DMatrix-EA` (training data generation) and `LiveONNX-EA` (live execution), guaranteeing exact tensor dimensional and calculation parity.
2. **In-Memory State Management**: Bypasses MT5's 31-character order comment limit by maintaining ticket-to-vector mappings in RAM.
3. **Golden Rule Invariant**: Trades with net liquid profit $\le 0.0$ (after spread, swap, and commission) are strictly labeled as $0.0f$ (`NOT_OPEN`), preventing false-positive patterns.
4. **Sub-Millisecond Inference**: Uses MetaTrader 5 native `vectorf` data structures and `ONNX_NO_CONVERSION` with pre-allocated static tensor shapes.
