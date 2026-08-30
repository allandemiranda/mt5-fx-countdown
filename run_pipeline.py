#!/usr/bin/env python3
"""MetaTrader 5 MLOps Pipeline Orchestrator."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from src.cleaner import ScopedCleaner
from src.config import AppConfig
from src.dataset_manager import DatasetManager
from src.mt5_client import MT5Client
from src.onnx_exporter import ONNXExporter
from src.preset_generator import PresetGenerator
from src.template_generator import TemplateGenerator
from src.trainer import DualXGBoostTrainer


def run_full_pipeline(
    config: AppConfig,
    workspace_root: Path,
    skip_dataset_override: bool | None = None,
) -> bool:
    """Execute end-to-end automated MLOps pipeline."""
    effective_skip_dataset = (
        skip_dataset_override if skip_dataset_override is not None else config.skip_dataset_generation
    )
    if effective_skip_dataset != config.skip_dataset_generation:
        config = dataclasses.replace(config, skip_dataset_generation=effective_skip_dataset)

    mt5_client = MT5Client(config, workspace_root)
    if not mt5_client.initialize():
        return False

    # 1. Clean previous scoped artifacts across active terminal and common paths
    cleaner = ScopedCleaner(config, workspace_root, mt5_client.terminal_data_path, mt5_client.common_path)
    cleaner.clean()

    dataset_mgr = DatasetManager(config, workspace_root, mt5_client.terminal_data_path, mt5_client.common_path)
    sym = config.symbol
    tf = config.clean_timeframe
    skip_tester = False
    ini_path: Path | None = None

    if effective_skip_dataset:
        if dataset_mgr.has_existing_datasets():
            skip_tester = True
            print(
                f"\n[*] [SKIP_DATASET] Existing datasets found for {sym}_{tf}. Skipping MT5 Strategy Tester execution."
            )
        else:
            print(
                f"\n[!] [WARNING] SKIP_DATASET_GENERATION is enabled, but datasets for {sym}_{tf} were not found. "
                f"Falling back to MT5 Strategy Tester execution."
            )

    try:
        mt5_client.sync_mql5()

        # Ensure macro governance SQLite database exists in Common/Files (creates empty schema if absent)
        macro_db_path = mt5_client.common_path / "Files" / "macro_governance.db"
        if not macro_db_path.exists():
            from macro_agent.db_client import init_schema
            init_schema(macro_db_path)
            print(f"[+] Initialized empty macro_governance.db (0 rows) in: {macro_db_path}")

        if not skip_tester:
            if not mt5_client.compile_ea("DMatrix-EA.mq5"):
                return False

            # 2. Generate preset (.set) for LiveONNX-EA / DMatrix-EA
            preset_gen = PresetGenerator(config, mt5_client.terminal_data_path, mt5_client.common_path)
            preset_gen.generate_all()

            # 3. Generate Strategy Tester config (.ini) and execute backtest
            ini_path = mt5_client.generate_tester_ini()
            if not mt5_client.run_strategy_tester(ini_path):
                return False
        else:
            preset_gen = PresetGenerator(config, mt5_client.terminal_data_path, mt5_client.common_path)
            preset_gen.generate_all()

        buy_csv, sell_csv, meta_json = dataset_mgr.find_and_validate_datasets()

        trainer = DualXGBoostTrainer(config)
        buy_clf, buy_metrics, feat_names = trainer.train(buy_csv, "buy")
        sell_clf, sell_metrics, _ = trainer.train(sell_csv, "sell")

        exporter = ONNXExporter(config, mt5_client.terminal_data_path, mt5_client.common_path)
        buy_onnx = exporter.export_and_validate(buy_clf, len(feat_names), "buy")
        sell_onnx = exporter.export_and_validate(sell_clf, len(feat_names), "sell")

        metadata = dataset_mgr.load_metadata(meta_json, len(feat_names), feat_names, buy_metrics, sell_metrics)
        exporter.deploy(buy_onnx, sell_onnx, metadata)

        # 4. Synchronize preset (.set) and chart template (.tpl) to all target directories
        live_set_path = preset_gen.generate_all()
        template_gen = TemplateGenerator(config, mt5_client.terminal_data_path, mt5_client.common_path)
        template_path = template_gen.generate_all()

        if not mt5_client.compile_ea("LiveONNX-EA.mq5"):
            return False

        print("\n" + "=" * 80)
        print("[SUCCESS] Full MLOps pipeline completed successfully!")
        if ini_path:
            print(f"   - Strategy Tester Config (.ini): {ini_path}")
        else:
            print("   - Strategy Tester Backtest:      Skipped (reused existing datasets)")
        print(f"   - Live Inference Preset (.set):   {live_set_path}")
        print(f"   - Chart Template (.tpl):          {template_path}")
        print(f"   - Live Model BUY (.onnx):          {buy_onnx}")
        print(f"   - Live Model SELL (.onnx):         {sell_onnx}")
        print("=" * 80)
        return True
    finally:
        mt5_client.shutdown()


def run_compile_only(config: AppConfig, workspace_root: Path) -> bool:
    """Synchronize, generate presets/templates, and compile MQL5 Expert Advisors."""
    mt5_client = MT5Client(config, workspace_root)
    if not mt5_client.initialize():
        return False
    try:
        mt5_client.sync_mql5()

        # Ensure macro governance SQLite database exists in Common/Files (creates empty schema if absent)
        macro_db_path = mt5_client.common_path / "Files" / "macro_governance.db"
        if not macro_db_path.exists():
            from macro_agent.db_client import init_schema
            init_schema(macro_db_path)
            print(f"[+] Initialized empty macro_governance.db (0 rows) in: {macro_db_path}")

        preset_gen = PresetGenerator(config, mt5_client.terminal_data_path, mt5_client.common_path)
        preset_gen.generate_all()
        template_gen = TemplateGenerator(config, mt5_client.terminal_data_path, mt5_client.common_path)
        template_gen.generate_all()
        dmatrix_ok = mt5_client.compile_ea("DMatrix-EA.mq5")
        live_ok = mt5_client.compile_ea("LiveONNX-EA.mq5")
        return dmatrix_ok and live_ok
    finally:
        mt5_client.shutdown()


def main() -> None:
    """CLI Entrypoint supporting full automated pipeline and compile-only modes."""
    parser = argparse.ArgumentParser(description="MetaTrader 5 MLOps Pipeline Orchestrator")
    parser.add_argument(
        "env_file",
        nargs="?",
        default=".env",
        help="Path to environment configuration file (default: .env)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only synchronize and compile MQL5 Expert Advisors",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip MT5 Strategy Tester execution and reuse existing datasets if available",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent
    env_path = workspace_root / args.env_file if not Path(args.env_file).is_absolute() else Path(args.env_file)
    if not env_path.exists():
        print(f"[ERROR] Configuration file not found: {env_path}")
        sys.exit(1)

    print(f"[*] Loading environment configuration from: {env_path.name}")
    config = AppConfig.from_env(env_path=env_path)

    if args.compile_only:
        success = run_compile_only(config, workspace_root)
    else:
        skip_override = True if args.skip_dataset else None
        success = run_full_pipeline(config, workspace_root, skip_dataset_override=skip_override)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
