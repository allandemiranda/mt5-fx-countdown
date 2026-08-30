"""ONNX export, validation, and multi-directory deployment module."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt
import xgboost as xgb

from src.config import AppConfig


class ONNXExporter:
    """Handles strict ONNX conversion (pure 1D float tensor, no ZipMap) and deployment."""

    def __init__(self, config: AppConfig, terminal_data_path: Path, common_path: Path):
        self.config = config
        self.terminal_data_path = terminal_data_path
        self.common_path = common_path

    def _validate_onnx_model(self, model_proto: onnx.ModelProto, num_features: int) -> None:
        """Validate ONNX model graph structure, input/output tensors, and numerical integrity."""
        # 1. Verify absence of ZipMap nodes
        op_types = [node.op_type for node in model_proto.graph.node]
        if "ZipMap" in op_types:
            raise ValueError(f"ZipMap operator detected in ONNX graph: {op_types}. MQL5 requires flat tensor output.")

        # 2. Run inference with ONNX Runtime
        model_bytes = model_proto.SerializeToString()
        sess = rt.InferenceSession(model_bytes)

        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        if len(inputs) != 1 or inputs[0].name != "float_input":
            raise ValueError(f"Expected single input 'float_input', got: {[i.name for i in inputs]}")

        if len(outputs) != 1 or outputs[0].name != "probabilities":
            raise ValueError(f"Expected single output 'probabilities', got: {[o.name for o in outputs]}")

        # 3. Numerical test with batch size 1
        rng = np.random.default_rng(42)
        dummy_input = rng.standard_normal((1, num_features), dtype=np.float32)
        result = sess.run(None, {inputs[0].name: dummy_input})
        prob_tensor = result[0]

        if prob_tensor.shape != (1, 2):
            raise ValueError(f"Expected output shape (1, 2), got: {prob_tensor.shape}")

        prob_sum = float(np.sum(prob_tensor[0]))
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities do not sum to 1.0 (Sum: {prob_sum})")

        print(f"    Graph Validated: Inputs={[(i.name, i.shape, i.type) for i in inputs]}")
        print(f"    Outputs={[(o.name, o.shape, o.type) for o in outputs]} | Test Prob Sum: {prob_sum:.4f}")

    def export_and_validate(self, clf: xgb.XGBClassifier, num_features: int, direction: str) -> Path:
        """Convert XGBoost model to strict ONNX format without ZipMap and save to disk."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        model_name = f"{sym}_{tf}_model_{direction.lower()}"
        print(f"\n[*] Converting {model_name} to pure Float Tensor ONNX graph...")

        # Format feature names to f0, f1, ... for onnxmltools parser compatibility
        clf.get_booster().feature_names = [f"f{i}" for i in range(num_features)]

        # Define pure float tensor input [None, num_features]
        initial_types = [("float_input", FloatTensorType([None, num_features]))]

        # Convert XGBoost model
        raw_onnx = onnxmltools.convert_xgboost(clf, initial_types=initial_types)

        # Prune outputs: strictly expose only the 'probabilities' [None, 2] float tensor
        prob_output = [out for out in raw_onnx.graph.output if out.name == "probabilities"]
        if not prob_output:
            raise ValueError("ONNX graph does not contain 'probabilities' tensor output.")

        pruned_model = onnx.ModelProto()
        pruned_model.CopyFrom(raw_onnx)
        del pruned_model.graph.output[:]
        pruned_model.graph.output.append(prob_output[0])

        # Validate with onnxruntime
        self._validate_onnx_model(pruned_model, num_features)

        # Save to terminal Models folder
        out_dir = self.terminal_data_path / "MQL5" / "Files" / "Models"
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_file = out_dir / f"{model_name}.onnx"

        with open(onnx_file, "wb") as f:
            f.write(pruned_model.SerializeToString())

        print(f"[+] Saved verified ONNX model: {onnx_file}")
        return onnx_file

    def deploy(self, buy_onnx: Path, sell_onnx: Path, metadata: Dict[str, Any]) -> None:
        """Deploy ONNX models and metadata to all MT5 terminal and common directories."""
        print("\n" + "=" * 80)
        print("STAGE 5: MODEL DEPLOYMENT & METADATA SYNCHRONIZATION")
        print("=" * 80)

        sym = self.config.symbol
        tf = self.config.clean_timeframe

        target_dirs: List[Path] = [
            self.terminal_data_path / "MQL5" / "Files" / "Models",
            self.common_path / "Files" / "Models",
        ]

        deployed_locations = 0
        for target_dir in target_dirs:
            target_dir.mkdir(parents=True, exist_ok=True)

            dst_buy = target_dir / buy_onnx.name
            dst_sell = target_dir / sell_onnx.name

            if buy_onnx.resolve() != dst_buy.resolve():
                shutil.copy2(buy_onnx, dst_buy)
            if sell_onnx.resolve() != dst_sell.resolve():
                shutil.copy2(sell_onnx, dst_sell)

            meta_sym_path = target_dir / f"{sym}_{tf}_metadata.json"
            with open(meta_sym_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            deployed_locations += 1
            print(f"    [+] Deployed to: {target_dir}")

        print(f"[+] Deployed {buy_onnx.name} and {sell_onnx.name} to {deployed_locations} Models directories.")
