"""Unit and integration tests for ONNX conversion, flat graph validation, and inference parity."""

from __future__ import annotations

import numpy as np
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import xgboost as xgb


def test_xgboost_to_flat_onnx_conversion():
    """Verify XGBoost conversion to flat ONNX without ZipMap and batch size 1 inference shape [1, N] -> [1, 2]."""
    num_features = 105
    num_samples = 200

    np.random.seed(42)
    x_train = np.random.randn(num_samples, num_features).astype(np.float32)
    y_train = np.random.randint(0, 2, size=num_samples).astype(np.int32)

    x_val = np.random.randn(50, num_features).astype(np.float32)
    y_val = np.random.randint(0, 2, size=50).astype(np.int32)

    clf = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.08,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=5,
        random_state=42,
    )
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    # 1. Feature names normalization
    clf.get_booster().feature_names = [f"f{i}" for i in range(num_features)]

    # 2. Convert via onnxmltools
    initial_types = [("float_input", FloatTensorType([None, num_features]))]
    raw_onnx = onnxmltools.convert_xgboost(clf, initial_types=initial_types)

    # 3. Verify and Prune Output to single probabilities tensor [None, 2]
    prob_output = [o for o in raw_onnx.graph.output if o.name == "probabilities"]
    assert len(prob_output) > 0, "Model missing probabilities output"

    pruned_model = onnx.ModelProto()
    pruned_model.CopyFrom(raw_onnx)
    del pruned_model.graph.output[:]
    pruned_model.graph.output.append(prob_output[0])

    # 4. Verify no ZipMap nodes in graph
    op_types = [node.op_type for node in pruned_model.graph.node]
    assert "ZipMap" not in op_types, f"ZipMap node detected in graph: {op_types}"

    # 5. Test inference with ONNX Runtime
    model_bytes = pruned_model.SerializeToString()
    sess = rt.InferenceSession(model_bytes)

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    assert len(inputs) == 1
    assert inputs[0].name == "float_input"
    assert inputs[0].type == "tensor(float)"

    assert len(outputs) == 1
    assert outputs[0].name == "probabilities"
    assert outputs[0].type == "tensor(float)"

    # 6. Test batch size 1 inference (matching MQL5 OnnxRun single tick)
    test_sample = np.random.randn(1, num_features).astype(np.float32)
    res = sess.run(None, {inputs[0].name: test_sample})

    prob_tensor = res[0]
    assert prob_tensor.shape == (1, 2), f"Expected shape (1, 2), got {prob_tensor.shape}"

    # Probabilities must be in [0, 1] and sum to 1.0
    p0, p1 = float(prob_tensor[0, 0]), float(prob_tensor[0, 1])
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert np.isclose(p0 + p1, 1.0, atol=1e-4)

    # 7. Numerical Parity Check: Compare ONNX Runtime with XGBoost predict_proba
    xgb_probs = clf.predict_proba(test_sample)
    assert np.allclose(prob_tensor, xgb_probs, atol=1e-4), (
        f"ONNX vs XGBoost probability mismatch: ONNX={prob_tensor}, XGB={xgb_probs}"
    )


def test_onnx_multi_batch_inference():
    """Verify ONNX runtime handles multi-sample batch inputs correctly [M, N] -> [M, 2]."""
    num_features = 36
    num_samples = 100
    batch_size = 16

    np.random.seed(99)
    x = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, 2, size=num_samples).astype(np.int32)

    clf = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(x, y)
    clf.get_booster().feature_names = [f"f{i}" for i in range(num_features)]

    initial_types = [("float_input", FloatTensorType([None, num_features]))]
    raw_onnx = onnxmltools.convert_xgboost(clf, initial_types=initial_types)

    pruned = onnx.ModelProto()
    pruned.CopyFrom(raw_onnx)
    del pruned.graph.output[:]
    pruned.graph.output.append([o for o in raw_onnx.graph.output if o.name == "probabilities"][0])

    sess = rt.InferenceSession(pruned.SerializeToString())
    batch_input = np.random.randn(batch_size, num_features).astype(np.float32)
    res = sess.run(None, {"float_input": batch_input})

    prob_tensor = res[0]
    assert prob_tensor.shape == (batch_size, 2)
    sums = np.sum(prob_tensor, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)
