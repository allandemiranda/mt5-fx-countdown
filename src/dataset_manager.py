"""Dataset discovery, validation, and metadata management module."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from src.config import AppConfig


class DatasetManager:
    """Discovers, validates, and manages training datasets across MT5 directories."""

    def __init__(
        self,
        config: AppConfig,
        workspace_root: Path,
        terminal_data_path: Path,
        common_path: Path,
    ):
        self.config = config
        self.workspace_root = workspace_root
        self.terminal_data_path = terminal_data_path
        self.common_path = common_path

    def _resolve_search_directories(self) -> List[Path]:
        """Resolve all directories where MT5 Strategy Tester may output dataset files."""
        dirs: List[Path] = [
            self.common_path / "Files",
            self.terminal_data_path / "MQL5" / "Files",
            self.workspace_root,
        ]

        tester_dir = self.terminal_data_path / "Tester"
        if tester_dir.exists():
            for agent in tester_dir.glob("Agent*"):
                dirs.append(agent / "MQL5" / "Files")

        appdata_tester = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Tester"
        if appdata_tester.exists():
            for agent_files in appdata_tester.glob("**/MQL5/Files"):
                if agent_files.is_dir():
                    dirs.append(agent_files)
            for agent_dir in appdata_tester.glob("**/Agent*"):
                if agent_dir.is_dir():
                    dirs.append(agent_dir / "MQL5" / "Files")
                    dirs.append(agent_dir / "Files")

        unique_dirs: List[Path] = []
        seen: Set[Path] = set()
        for d in dirs:
            try:
                resolved = d.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    unique_dirs.append(d)
            except Exception:
                if d not in unique_dirs:
                    unique_dirs.append(d)
        return unique_dirs

    def _dump_tester_logs(self, max_lines: int = 40) -> None:
        """Find and display recent MT5 tester logs for diagnostics."""
        print("\n" + "-" * 80)
        print("[DIAGNOSTIC] Inspecting MT5 Tester Logs...")
        print("-" * 80)

        potential_log_dirs = [
            self.terminal_data_path / "Tester" / "logs",
            self.terminal_data_path / "logs",
        ]
        appdata_tester = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Tester"
        if appdata_tester.exists():
            for agent in appdata_tester.glob("**/logs"):
                if agent.is_dir():
                    potential_log_dirs.append(agent)

        all_logs: List[Path] = []
        excluded_logs = {"metaeditor.log", "favorites_diagnostic.log"}
        for log_dir in potential_log_dirs:
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    if not log_file.name.startswith("compile_") and log_file.name not in excluded_logs:
                        all_logs.append(log_file)

        if not all_logs:
            print("[-] No MT5 tester log files found.")
            return

        all_logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_log = all_logs[0]
        print(f"[*] Most recent tester log: {latest_log}")
        try:
            try:
                content = latest_log.read_text(encoding="utf-16", errors="ignore")
            except Exception:
                content = latest_log.read_text(encoding="utf-8", errors="ignore")

            lines = [line for line in content.splitlines() if line.strip()]
            for line in lines[-max_lines:]:
                print(f"    {line}")
        except Exception as exc:
            print(f"[-] Could not read log file: {exc}")
        print("-" * 80)

    def has_existing_datasets(self) -> bool:
        """Check whether both BUY and SELL dataset CSV files exist and are non-empty across search paths."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        buy_name = f"{sym}_{tf}_buy.csv"
        sell_name = f"{sym}_{tf}_sell.csv"

        search_dirs = self._resolve_search_directories()
        has_buy = False
        has_sell = False

        for directory in search_dirs:
            if not directory.exists():
                continue
            b_path = directory / buy_name
            s_path = directory / sell_name

            if not has_buy and b_path.exists() and b_path.stat().st_size > 0:
                has_buy = True
            if not has_sell and s_path.exists() and s_path.stat().st_size > 0:
                has_sell = True

            if has_buy and has_sell:
                return True

        return False

    def find_and_validate_datasets(self) -> Tuple[Path, Path, Optional[Path]]:
        """Locate and validate BUY and SELL dataset CSV files across MT5 search paths."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        buy_name = f"{sym}_{tf}_buy.csv"
        sell_name = f"{sym}_{tf}_sell.csv"
        meta_name = f"{sym}_{tf}_metadata.json"

        search_dirs = self._resolve_search_directories()
        found_buy: Optional[Path] = None
        found_sell: Optional[Path] = None
        found_meta: Optional[Path] = None

        for directory in search_dirs:
            if not directory.exists():
                continue
            b_path = directory / buy_name
            s_path = directory / sell_name
            m_path = directory / meta_name

            if b_path.exists() and b_path.stat().st_size > 0:
                found_buy = b_path
            if s_path.exists() and s_path.stat().st_size > 0:
                found_sell = s_path
            if m_path.exists() and m_path.stat().st_size > 0:
                found_meta = m_path

        if not found_buy or not found_sell:
            self._dump_tester_logs()
            raise RuntimeError(
                f"\n[FATAL ERROR] MT5 Strategy Tester finished, but required datasets for {sym} {tf} were not found.\n"
                f"Expected: '{buy_name}' and '{sell_name}'.\n"
                f"Please verify broker historical data availability and MQL5 runtime logs."
            )

        # Quick validation of CSV structure
        df_buy = pd.read_csv(found_buy, nrows=5)
        df_sell = pd.read_csv(found_sell, nrows=5)

        if "label" not in df_buy.columns or "label" not in df_sell.columns:
            raise ValueError(f"Dataset CSVs must contain 'label' column. BUY columns: {list(df_buy.columns)}")

        print(f"[+] Verified datasets for {sym}_{tf}:")
        print(f"    BUY Dataset:  {found_buy} ({found_buy.stat().st_size} bytes)")
        print(f"    SELL Dataset: {found_sell} ({found_sell.stat().st_size} bytes)")
        return found_buy, found_sell, found_meta

    def load_metadata(
        self,
        meta_path: Optional[Path],
        num_features: int,
        feature_names: List[str],
        metrics_buy: Dict[str, Any],
        metrics_sell: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load or build execution and model metadata."""
        meta: Dict[str, Any] = {}
        if meta_path and meta_path.exists():
            for attempt in range(5):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    time.sleep(1.0)

        if not meta:
            meta = {
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
            }

        meta["num_features"] = num_features
        meta["feature_names"] = feature_names
        meta["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        meta["metrics"] = {
            "buy": metrics_buy,
            "sell": metrics_sell,
        }
        return meta
