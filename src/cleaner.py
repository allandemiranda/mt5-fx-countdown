"""Atomic file cleaner module strictly scoped to Symbol and Timeframe."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set
from src.config import AppConfig


class ScopedCleaner:
    """Performs atomic scoped cleanup for target Symbol and Timeframe artifacts."""

    def __init__(
        self,
        config: AppConfig,
        workspace_root: Path,
        terminal_data_path: Path | None = None,
        common_path: Path | None = None,
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.terminal_data_path = terminal_data_path or config.mt5_data_path
        self.common_path = common_path or config.mt5_common_path

    def _resolve_target_directories(self) -> List[Path]:
        """Collect all potential directories where scoped artifacts could reside."""
        dirs: List[Path] = [self.workspace_root]

        if self.terminal_data_path and self.terminal_data_path.exists():
            dirs.extend([
                self.terminal_data_path,
                self.terminal_data_path / "MQL5" / "Files",
                self.terminal_data_path / "MQL5" / "Files" / "Models",
                self.terminal_data_path / "MQL5" / "Presets",
                self.terminal_data_path / "logs",
                self.terminal_data_path / "MQL5" / "Profiles" / "Templates",
            ])
            tester_dir = self.terminal_data_path / "Tester"
            if tester_dir.exists():
                dirs.append(tester_dir)
                for agent in tester_dir.glob("Agent*"):
                    dirs.append(agent / "MQL5" / "Files")

        if self.common_path and self.common_path.exists():
            dirs.extend([
                self.common_path / "Files",
                self.common_path / "Files" / "Models",
                self.common_path / "Files" / "Presets",
                self.common_path / "Files" / "Templates",
            ])

        # Also inspect MetaQuotes Tester agents directory if present
        appdata_tester = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Tester"
        if appdata_tester.exists():
            for agent_files in appdata_tester.glob("**/MQL5/Files"):
                if agent_files.is_dir():
                    dirs.append(agent_files)
            for agent_dir in appdata_tester.glob("**/Agent*"):
                if agent_dir.is_dir():
                    dirs.append(agent_dir / "MQL5" / "Files")
                    dirs.append(agent_dir / "Files")

        # Deduplicate paths
        unique_dirs: List[Path] = []
        seen: Set[Path] = set()
        for d in dirs:
            try:
                resolved = d.resolve()
                if resolved not in seen and d.exists():
                    seen.add(resolved)
                    unique_dirs.append(d)
            except Exception:
                if d not in unique_dirs and d.exists():
                    unique_dirs.append(d)
        return unique_dirs

    def clean(self) -> List[Path]:
        """Atomically remove pre-existing artifacts strictly scoped to Symbol and Timeframe."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        print(f"\n[*] Cleaning previous scoped artifacts for {sym}_{tf}...")

        patterns = [
            f"tester_{sym}_{tf}.ini",
            f"DMatrix_{sym}_{tf}_Report*",
            f"{sym}_{tf}_*.onnx",
            f"*{sym}_{tf}*.set",
            f"*{sym}_{tf}*.tpl",
            f"compile_*{sym}_{tf}*.log",
            "compile_DMatrix-EA.log",
            "compile_LiveONNX-EA.log",
            "pipeline_metadata.json",
            "LiveONNX-EA.set",
            "DMatrix-EA.set",
            "DMatrix-EA_*.set",
        ]

        if not self.config.skip_dataset_generation:
            patterns.extend([
                f"{sym}_{tf}_*.csv",
                f"{sym}_{tf}_*.json",
            ])

        target_dirs = self._resolve_target_directories()
        deleted_files: List[Path] = []

        for directory in target_dirs:
            for pattern in patterns:
                for match in directory.glob(pattern):
                    if match.is_file():
                        try:
                            match.unlink()
                            deleted_files.append(match)
                        except Exception as exc:
                            print(f"    [!] Could not delete {match.name}: {exc}")

        # Also purge any misplaced root-level .onnx files in MQL5/Files and Common/Files
        for root_files_dir in [
            self.terminal_data_path / "MQL5" / "Files" if self.terminal_data_path else None,
            self.common_path / "Files" if self.common_path else None,
        ]:
            if root_files_dir and root_files_dir.exists():
                for misplaced in root_files_dir.glob("*.onnx"):
                    if misplaced.is_file():
                        try:
                            misplaced.unlink()
                            deleted_files.append(misplaced)
                        except Exception:
                            pass
                if not self.config.skip_dataset_generation:
                    for misplaced_json in root_files_dir.glob("*.json"):
                        if misplaced_json.is_file():
                            try:
                                misplaced_json.unlink()
                                deleted_files.append(misplaced_json)
                            except Exception:
                                pass

        # Purge any stray README.md files previously copied to MT5 terminal or common paths
        for base_path in (self.terminal_data_path, self.common_path):
            if base_path and base_path.exists() and base_path != self.workspace_root:
                for readme in base_path.glob("**/README.md"):
                    if readme.is_file():
                        try:
                            readme.unlink()
                            deleted_files.append(readme)
                        except Exception:
                            pass

        print(f"[+] Cleaned {len(deleted_files)} scoped artifact files for {sym}_{tf}.")
        return deleted_files
