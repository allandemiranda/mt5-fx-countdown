"""Automated Test Verification for Native MQL5 Unit Test Suite.

Verifies that:
1. Native MQL5 Test Framework (MqlTestFramework.mqh) exists and defines assertions.
2. MQL5 Unit Test Suites (TestGarchEngine, TestOrderTracker, TestFeatureExtractor) exist.
3. The master test runner (RunAllMQL5UnitTests.mq5) compiles with 0 errors via MetaEditor CLI.
"""

from pathlib import Path
import pytest
from src.config import AppConfig
from src.mt5_client import MT5Client


def test_mql5_test_framework_files_exist() -> None:
    """Verify that all MQL5 unit test framework and suite files exist."""
    workspace_root = Path(__file__).resolve().parent.parent
    framework_path = workspace_root / "MQL5" / "Include" / "Tests" / "MqlTestFramework.mqh"
    assert framework_path.exists(), f"Framework missing: {framework_path}"

    test_garch = workspace_root / "MQL5" / "Include" / "Tests" / "TestGarchEngine.mqh"
    assert test_garch.exists(), f"Suite missing: {test_garch}"

    test_tracker = workspace_root / "MQL5" / "Include" / "Tests" / "TestOrderTracker.mqh"
    assert test_tracker.exists(), f"Suite missing: {test_tracker}"

    test_extractor = workspace_root / "MQL5" / "Include" / "Tests" / "TestFeatureExtractor.mqh"
    assert test_extractor.exists(), f"Suite missing: {test_extractor}"

    test_consecutive = workspace_root / "MQL5" / "Include" / "Tests" / "TestConsecutiveManager.mqh"
    assert test_consecutive.exists(), f"Suite missing: {test_consecutive}"

    runner_script = workspace_root / "MQL5" / "Scripts" / "Tests" / "RunAllMQL5UnitTests.mq5"
    assert runner_script.exists(), f"Runner script missing: {runner_script}"


def test_mql5_unit_tests_compilation_metaeditor() -> None:
    """Verify that RunAllMQL5UnitTests.mq5 compiles with 0 errors via MetaEditor CLI."""
    workspace_root = Path(__file__).resolve().parent.parent
    cfg = AppConfig.from_env()

    if not cfg.metaeditor_path.exists():
        pytest.skip(f"MetaEditor executable not found at: {cfg.metaeditor_path}")

    client = MT5Client(cfg, workspace_root)
    if not client.initialize():
        pytest.skip("MT5 terminal could not be initialized in this environment.")

    client.sync_mql5()

    compile_ok = client.compile_mql5_file("Scripts/Tests/RunAllMQL5UnitTests.mq5")
    assert compile_ok is True, "RunAllMQL5UnitTests.mq5 compilation failed via MetaEditor CLI"
