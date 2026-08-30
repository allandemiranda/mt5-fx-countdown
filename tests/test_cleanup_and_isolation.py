"""Unit and isolation tests for ScopedCleaner and multi-symbol/timeframe artifact isolation."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.cleaner import ScopedCleaner
from src.config import AppConfig
from src.mt5_client import MT5Client
from src.preset_generator import PresetGenerator


@pytest.fixture(autouse=True)
def setup_default_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure baseline environment variables are loaded for all cleanup and isolation tests."""
    AppConfig.from_env()


def test_scoped_cleaner_matching_and_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that ScopedCleaner removes strictly the targeted symbol and timeframe artifacts across all directories."""
    monkeypatch.setenv("SYMBOL", "EURUSD")
    monkeypatch.setenv("TIMEFRAME", "H1")

    config = AppConfig.from_env()

    # Create directory structure
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"
    ws_dir = tmp_path / "workspace"

    models_dir = term_dir / "MQL5" / "Files" / "Models"
    presets_dir = term_dir / "MQL5" / "Presets"
    logs_dir = term_dir / "logs"
    agent_dir = term_dir / "Tester" / "Agent-01" / "MQL5" / "Files"

    for d in [models_dir, presets_dir, logs_dir, agent_dir, comm_dir / "Files", ws_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Matching artifacts (MUST be deleted)
    matching_files = [
        ws_dir / "tester_EURUSD_H1.ini",
        ws_dir / "EURUSD_H1_buy.csv",
        ws_dir / "EURUSD_H1_sell.csv",
        ws_dir / "DMatrix_EURUSD_H1_Report.htm",
        models_dir / "EURUSD_H1_model_buy.onnx",
        models_dir / "EURUSD_H1_model_sell.onnx",
        models_dir / "EURUSD_H1_metadata.json",
        presets_dir / "LiveONNX-EA_EURUSD_H1.set",
        presets_dir / "DMatrix-EA_EURUSD_H1.set",
        logs_dir / "compile_LiveONNX-EA_EURUSD_H1.log",
        agent_dir / "EURUSD_H1_buy.csv",
    ]

    # 2. Non-matching artifacts from other symbols/timeframes (MUST NOT be deleted)
    isolated_files = [
        ws_dir / "tester_GBPUSD_M15.ini",
        ws_dir / "GBPUSD_M15_buy.csv",
        ws_dir / "EURUSD_M5_buy.csv",
        models_dir / "GBPUSD_M15_model_buy.onnx",
        presets_dir / "LiveONNX-EA_GBPUSD_M15.set",
        logs_dir / "compile_LiveONNX-EA_GBPUSD_M15.log",
        agent_dir / "USDJPY_H1_buy.csv",
    ]

    for f in matching_files + isolated_files:
        f.write_text("data_content", encoding="utf-8")
        assert f.exists()

    cleaner = ScopedCleaner(config, ws_dir, terminal_data_path=term_dir, common_path=comm_dir)
    deleted = cleaner.clean()

    # Verify all matching files were deleted
    for mf in matching_files:
        assert mf in deleted, f"Expected {mf.name} to be in deleted list"
        assert not mf.exists(), f"File {mf} was not deleted"

    # Verify all non-matching files remain intact
    for non_mf in isolated_files:
        assert non_mf not in deleted, f"File {non_mf.name} was improperly deleted"
        assert non_mf.exists(), f"File {non_mf} should still exist"


def test_scoped_cleaner_skip_dataset_preserves_csv_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that ScopedCleaner preserves CSV and JSON files when skip_dataset_generation is True."""
    monkeypatch.setenv("SYMBOL", "EURUSD")
    monkeypatch.setenv("TIMEFRAME", "H1")
    monkeypatch.setenv("SKIP_DATASET_GENERATION", "1")

    config = AppConfig.from_env()

    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"
    ws_dir = tmp_path / "workspace"

    models_dir = term_dir / "MQL5" / "Files" / "Models"
    presets_dir = term_dir / "MQL5" / "Presets"
    logs_dir = term_dir / "logs"
    files_dir = term_dir / "MQL5" / "Files"

    for d in [models_dir, presets_dir, logs_dir, files_dir, comm_dir / "Files", ws_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Preserved files (CSVs and JSONs)
    preserved_files = [
        ws_dir / "EURUSD_H1_buy.csv",
        ws_dir / "EURUSD_H1_sell.csv",
        files_dir / "EURUSD_H1_buy.csv",
        files_dir / "EURUSD_H1_sell.csv",
        files_dir / "EURUSD_H1_metadata.json",
        comm_dir / "Files" / "EURUSD_H1_buy.csv",
    ]

    # Files that MUST still be deleted
    deleted_target_files = [
        ws_dir / "tester_EURUSD_H1.ini",
        models_dir / "EURUSD_H1_model_buy.onnx",
        models_dir / "EURUSD_H1_model_sell.onnx",
        presets_dir / "LiveONNX-EA_EURUSD_H1.set",
        logs_dir / "compile_LiveONNX-EA_EURUSD_H1.log",
    ]

    for f in preserved_files + deleted_target_files:
        f.write_text("data_content", encoding="utf-8")
        assert f.exists()

    cleaner = ScopedCleaner(config, ws_dir, terminal_data_path=term_dir, common_path=comm_dir)
    deleted = cleaner.clean()

    # Verify deleted files
    for df in deleted_target_files:
        assert df in deleted, f"Expected {df.name} to be deleted"
        assert not df.exists(), f"File {df} was not deleted"

    # Verify preserved files
    for pf in preserved_files:
        assert pf not in deleted, f"Expected {pf.name} to be preserved"
        assert pf.exists(), f"File {pf} was deleted"


def test_scoped_cleaner_nonexistent_directories_graceful(tmp_path: Path):
    """Verify ScopedCleaner executes cleanly without error when directories do not exist."""
    config = AppConfig.from_env()
    non_existent_term = tmp_path / "does_not_exist_term"
    non_existent_comm = tmp_path / "does_not_exist_comm"
    ws = tmp_path / "ws"
    ws.mkdir()

    cleaner = ScopedCleaner(config, ws, terminal_data_path=non_existent_term, common_path=non_existent_comm)
    deleted = cleaner.clean()
    assert isinstance(deleted, list)
    assert len(deleted) == 0


def test_multi_symbol_timeframe_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that generated tester INI and presets across different symbols and timeframes never collide."""
    pairs = [("EURUSD", "H1"), ("GBPUSD", "M15"), ("USDJPY", "M5"), ("AUDUSD", "D1")]

    generated_inis = []
    generated_presets = []

    for sym, tf in pairs:
        monkeypatch.setenv("SYMBOL", sym)
        monkeypatch.setenv("TIMEFRAME", tf)

        config = AppConfig.from_env()
        client = MT5Client(config, tmp_path)
        client.terminal_data_path = tmp_path

        ini_path = client.generate_tester_ini()
        assert ini_path.name == f"tester_{sym}_{tf}.ini"
        generated_inis.append(ini_path)

        generator = PresetGenerator(config, tmp_path, tmp_path)
        live_set = generator.generate_all([tmp_path])
        assert live_set.name == f"LiveONNX-EA_{sym}_{tf}.set"
        generated_presets.append(live_set)

    # All generated file paths must be strictly distinct
    assert len(generated_inis) == len(set(generated_inis))
    assert len(generated_presets) == len(set(generated_presets))


def test_tester_ini_generation_and_stale_file_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that generate_tester_ini replaces stale file content and configures parameters correctly."""
    monkeypatch.setenv("SYMBOL", "USDCHF")
    monkeypatch.setenv("TIMEFRAME", "M30")
    monkeypatch.setenv("SHUTDOWN_TERMINAL", "1")

    config = AppConfig.from_env()
    client = MT5Client(config, tmp_path)
    client.terminal_data_path = tmp_path

    ini_file = tmp_path / "tester_USDCHF_M30.ini"
    ini_file.write_text("CORRUPTED_OLD_CONFIG_DATA", encoding="ascii")
    assert ini_file.exists()

    generated_path = client.generate_tester_ini()
    assert generated_path == ini_file
    assert generated_path.exists()

    content = generated_path.read_text(encoding="ascii")
    assert "CORRUPTED_OLD_CONFIG_DATA" not in content
    assert "Expert=DMatrix-EA.ex5" in content
    assert "Symbol=USDCHF" in content
    assert "Period=M30" in content
    assert "Deposit=1000000000000000" in content
    assert "Leverage=500" in content
    assert "Model=4" in content
    assert "ProfitInPips=0" in content
    assert "ShutdownTerminal=1" in content
    assert "Report=" not in content


def test_sync_mql5_excludes_markdown_and_docs_and_removes_old_readmes(tmp_path: Path):
    """Verify that sync_mql5 only copies code/presets (.mq5, .mqh, .set, .mq4),

    excludes markdown/documentation, and purges pre-existing README.md files from terminal directories.
    """
    config = AppConfig.from_env()
    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    # Setup workspace MQL5 directory
    ws_experts = ws_dir / "MQL5" / "Experts"
    ws_include = ws_dir / "MQL5" / "Include"
    ws_presets = ws_dir / "MQL5" / "Presets"
    for d in (ws_experts, ws_include, ws_presets):
        d.mkdir(parents=True, exist_ok=True)

    # Valid code and preset files to be copied
    valid_files = [
        ws_experts / "DMatrix-EA.mq5",
        ws_experts / "LiveONNX-EA.mq5",
        ws_include / "FeatureExtractor.mqh",
        ws_presets / "Default.set",
        ws_experts / "Legacy.mq4",
    ]
    for f in valid_files:
        f.write_text("// valid code or preset", encoding="utf-8")

    # Documentation and non-code files that MUST be excluded
    doc_files = [
        ws_dir / "MQL5" / "README.md",
        ws_experts / "README.md",
        ws_experts / "notes.txt",
        ws_include / "README.md",
        ws_include / "guide.md",
        ws_presets / "README.md",
        ws_presets / "info.txt",
    ]
    for f in doc_files:
        f.write_text("# documentation", encoding="utf-8")

    # Pre-existing README.md files in terminal and common paths (MUST be purged)
    old_term_readme1 = term_dir / "MQL5" / "README.md"
    old_term_readme2 = term_dir / "MQL5" / "Experts" / "README.md"
    old_comm_readme = comm_dir / "Files" / "README.md"
    for r in (old_term_readme1, old_term_readme2, old_comm_readme):
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text("# stale readme", encoding="utf-8")
        assert r.exists()

    client = MT5Client(config, ws_dir)
    client.terminal_data_path = term_dir
    client.common_path = comm_dir

    client.sync_mql5()

    # Verify valid files are copied
    assert (term_dir / "MQL5" / "Experts" / "DMatrix-EA.mq5").exists()
    assert (term_dir / "MQL5" / "Experts" / "LiveONNX-EA.mq5").exists()
    assert (term_dir / "MQL5" / "Include" / "FeatureExtractor.mqh").exists()
    assert (term_dir / "MQL5" / "Presets" / "Default.set").exists()
    assert (term_dir / "MQL5" / "Experts" / "Legacy.mq4").exists()

    # Verify documentation files were NOT copied
    assert not (term_dir / "MQL5" / "Experts" / "README.md").exists()
    assert not (term_dir / "MQL5" / "Experts" / "notes.txt").exists()
    assert not (term_dir / "MQL5" / "Include" / "README.md").exists()
    assert not (term_dir / "MQL5" / "Include" / "guide.md").exists()
    assert not (term_dir / "MQL5" / "Presets" / "README.md").exists()
    assert not (term_dir / "MQL5" / "Presets" / "info.txt").exists()

    # Verify pre-existing terminal/common README.md files were removed
    assert not old_term_readme1.exists()
    assert not old_term_readme2.exists()
    assert not old_comm_readme.exists()

    # Verify workspace documentation files remain intact
    for f in doc_files:
        assert f.exists(), f"Workspace file {f} should never be deleted!"


def test_scoped_cleaner_purges_stray_readmes_and_preserves_workspace(tmp_path: Path):
    """Verify ScopedCleaner purges stray README.md files from terminal/common without touching workspace."""
    config = AppConfig.from_env()
    ws_dir = tmp_path / "workspace"
    term_dir = tmp_path / "terminal"
    comm_dir = tmp_path / "common"

    ws_readme = ws_dir / "README.md"
    ws_mql5_readme = ws_dir / "MQL5" / "README.md"
    term_readme = term_dir / "MQL5" / "Files" / "README.md"
    comm_readme = comm_dir / "Files" / "README.md"

    for r in (ws_readme, ws_mql5_readme, term_readme, comm_readme):
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text("# readme content", encoding="utf-8")
        assert r.exists()

    cleaner = ScopedCleaner(config, ws_dir, terminal_data_path=term_dir, common_path=comm_dir)
    deleted = cleaner.clean()

    # Terminal and common README.md files should be purged
    assert term_readme in deleted
    assert comm_readme in deleted
    assert not term_readme.exists()
    assert not comm_readme.exists()

    # Workspace README.md files MUST be strictly preserved
    assert ws_readme not in deleted
    assert ws_mql5_readme not in deleted
    assert ws_readme.exists()
    assert ws_mql5_readme.exists()
