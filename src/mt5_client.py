"""MetaTrader 5 API client wrapper and execution manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from src.config import AppConfig


class MT5Client:
    """Encapsulates MetaTrader 5 terminal interaction, compilation, and backtesting."""

    def __init__(self, config: AppConfig, workspace_root: Path):
        self.config = config
        self.workspace_root = workspace_root
        self.terminal_data_path = self._resolve_terminal_data_path(config.mt5_data_path)
        self.common_path = self._resolve_common_path(config.mt5_common_path)

    def _resolve_terminal_data_path(self, explicit_path: Optional[Path]) -> Path:
        """Resolve MT5 terminal data directory."""
        if explicit_path and explicit_path.exists():
            return explicit_path

        appdata_terminal = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Terminal"
        if appdata_terminal.exists():
            for entry in appdata_terminal.iterdir():
                if entry.is_dir() and len(entry.name) == 32 and (entry / "MQL5").exists():
                    return entry
        return appdata_terminal / "Default"

    def _resolve_common_path(self, explicit_path: Optional[Path]) -> Path:
        """Resolve MT5 shared Common directory."""
        if explicit_path and explicit_path.exists():
            return explicit_path

        appdata_common = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common"
        return appdata_common

    def initialize(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal and verify asset symbol."""
        print("\n" + "=" * 80)
        print("STAGE 1: MT5 TERMINAL VERIFICATION & INITIALIZATION")
        print("=" * 80)

        if mt5 is None:
            print("[ERROR] MetaTrader5 Python package is not available. Please install MetaTrader5.")
            return False

        if not self.config.mt5_path.exists():
            print(f"[ERROR] MT5 executable not found at: {self.config.mt5_path}")
            return False

        print(f"[*] Initializing MT5 connection with path: {self.config.mt5_path}...")
        if not mt5.initialize(path=str(self.config.mt5_path)):
            err = mt5.last_error()
            print(f"[ERROR] MT5 initialization failed: {err}")
            return False

        terminal_info = mt5.terminal_info()
        version = mt5.version()
        print(f"[+] MT5 Connected! Version: {version[0]}, Build: {version[1]} ({version[2]})")

        if terminal_info is not None:
            print(f"    Terminal Path:     {terminal_info.path}")
            print(f"    Data Path:         {terminal_info.data_path}")
            print(f"    Common Data Path:  {terminal_info.commondata_path}")
            print(f"    Connected Account: {terminal_info.connected}")

            # Dynamically update terminal paths from live MT5 instance if not set
            if not self.config.mt5_data_path:
                self.terminal_data_path = Path(terminal_info.data_path)
            if not self.config.mt5_common_path:
                self.common_path = Path(terminal_info.commondata_path)

        # Symbol Verification and Selection
        sym_info = mt5.symbol_info(self.config.symbol)
        if sym_info is None:
            print(f"[!] Warning: Symbol '{self.config.symbol}' not selected. Attempting to select...")
            if not mt5.symbol_select(self.config.symbol, True):
                print(f"[!] Warning: Could not select '{self.config.symbol}'. Ensure broker offers this symbol.")
            sym_info = mt5.symbol_info(self.config.symbol)

        if sym_info is not None:
            print(
                f"[+] Symbol '{self.config.symbol}' verified "
                f"(Digits: {sym_info.digits}, Point: {sym_info.point}, Spread: {sym_info.spread})"
            )

        return True

    def sync_mql5(self) -> None:
        """Synchronize workspace MQL5 directory into the MT5 terminal data directory."""
        src_mql5 = self.workspace_root / "MQL5"
        dst_mql5 = self.terminal_data_path / "MQL5"

        if not src_mql5.exists():
            return

        print(f"[*] Syncing MQL5 files from '{src_mql5}' -> '{dst_mql5}'...")
        allowed_extensions = {".mq5", ".mqh", ".set", ".mq4"}

        for sub in ("Include", "Experts", "Presets", "Scripts"):
            src_sub = src_mql5 / sub
            dst_sub = dst_mql5 / sub
            if src_sub.exists():
                dst_sub.mkdir(parents=True, exist_ok=True)
                for item in src_sub.rglob("*"):
                    if (
                        item.is_file()
                        and not item.name.lower().endswith((".md", ".txt"))
                        and item.suffix.lower() in allowed_extensions
                    ):
                        rel_path = item.relative_to(src_sub)
                        target_file = dst_sub / rel_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_file)

        # Remove any existing README.md files previously copied to terminal_data_path and common_path
        for base_path in (self.terminal_data_path, self.common_path):
            if base_path and base_path.exists() and base_path != self.workspace_root:
                for readme in base_path.glob("**/README.md"):
                    if readme.is_file():
                        try:
                            readme.unlink()
                        except Exception:
                            pass

        print("[+] MQL5 file synchronization complete.")

    def compile_mql5_file(self, rel_file_path: str | Path) -> bool:
        """Compile any MQL5 file (Expert Advisor or Script) using MetaEditor CLI."""
        target_path = self.workspace_root / "MQL5" / rel_file_path
        if not target_path.exists():
            print(f"[ERROR] MQL5 source file not found: {target_path}")
            return False

        if not self.config.metaeditor_path.exists():
            print(f"[ERROR] MetaEditor executable not found at: {self.config.metaeditor_path}")
            return False

        log_dir = self.terminal_data_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"compile_{target_path.stem}.log"

        print(f"[*] Compiling '{rel_file_path}' using MetaEditor CLI...")
        cmd = [
            str(self.config.metaeditor_path),
            f"/compile:{target_path}",
            f"/log:{log_path}",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            time.sleep(1.5)

            log_content = ""
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding="utf-16", errors="ignore")
                except Exception:
                    log_content = log_path.read_text(encoding="utf-8", errors="ignore")

                if not log_content.strip():
                    log_content = log_path.read_text(encoding="utf-8", errors="ignore")

            if "0 errors" in log_content:
                print(f"[+] Compilation SUCCESS for {rel_file_path} (0 errors)")
                ex5_file = target_path.with_suffix(".ex5")
                if ex5_file.exists():
                    target_ex5 = self.terminal_data_path / "MQL5" / rel_file_path
                    target_ex5 = target_ex5.with_suffix(".ex5")
                    target_ex5.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ex5_file, target_ex5)
                    try:
                        ex5_file.unlink()
                    except Exception:
                        pass
                return True

            print(f"[ERROR] Compilation failed for {rel_file_path}:\n{log_content}")
            return False
        except Exception as exc:
            print(f"[ERROR] Error executing MetaEditor CLI: {exc}")
            return False

    def compile_ea(self, ea_name: str) -> bool:
        """Compile an MQL5 Expert Advisor using MetaEditor CLI."""
        return self.compile_mql5_file(Path("Experts") / ea_name)

    def _get_tester_log_directories(self) -> list[Path]:
        """Resolve all directories where MT5 Strategy Tester may output log files."""
        dirs: list[Path] = [
            self.terminal_data_path / "Tester" / "logs",
            self.terminal_data_path / "logs",
        ]
        tester_sub = self.terminal_data_path / "Tester"
        if tester_sub.exists():
            for agent_logs in tester_sub.glob("**/logs"):
                if agent_logs.is_dir():
                    dirs.append(agent_logs)

        # Check APPDATA/MetaQuotes/Tester only if terminal_data_path resides inside APPDATA
        appdata_terminal = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Terminal"
        if appdata_terminal.exists():
            try:
                self.terminal_data_path.resolve().relative_to(appdata_terminal.resolve())
                appdata_tester = Path(os.getenv("APPDATA", "")) / "MetaQuotes" / "Tester"
                if appdata_tester.exists():
                    for agent_logs in appdata_tester.glob("**/logs"):
                        if agent_logs.is_dir():
                            dirs.append(agent_logs)
            except ValueError:
                pass

        unique_dirs: list[Path] = []
        seen: set[Path] = set()
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

    def _stream_new_tester_logs(
        self,
        file_offsets: dict[Path, int],
        excluded_files: set[str] | None = None,
    ) -> list[str]:
        """Read and stream new log lines from active MT5 tester and terminal logs."""
        if excluded_files is None:
            excluded_files = {"metaeditor.log", "favorites_diagnostic.log"}

        new_lines: list[str] = []
        for log_dir in self._get_tester_log_directories():
            if not log_dir.exists():
                continue
            for log_file in log_dir.glob("*.log"):
                if log_file.name in excluded_files or log_file.name.startswith("compile_"):
                    continue

                try:
                    curr_size = log_file.stat().st_size
                except OSError:
                    continue

                prev_offset = file_offsets.get(log_file, 0)
                if curr_size <= prev_offset:
                    continue

                try:
                    with open(log_file, "rb") as f:
                        f.seek(prev_offset)
                        raw_bytes = f.read()
                        file_offsets[log_file] = f.tell()

                    # Detect UTF-16 (standard MT5 log with null bytes) vs UTF-8
                    if b"\x00" in raw_bytes:
                        try:
                            text = raw_bytes.decode("utf-16", errors="ignore")
                        except Exception:
                            text = raw_bytes.decode("utf-8", errors="ignore")
                    else:
                        try:
                            text = raw_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            text = raw_bytes.decode("utf-16", errors="ignore")

                    for raw_line in text.splitlines():
                        clean = raw_line.strip()
                        if clean:
                            new_lines.append(clean)
                except Exception:
                    pass

        return new_lines

    def generate_tester_ini(self) -> Path:
        """Generate Strategy Tester configuration (.ini) with institutional high-liquidity parameters."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        ini_path = self.terminal_data_path / f"tester_{sym}_{tf}.ini"

        if ini_path.exists():
            try:
                ini_path.unlink()
            except Exception as exc:
                print(f"[!] Warning: Could not remove old ini {ini_path.name}: {exc}")

        def to_bool_str(val: bool) -> str:
            return "true" if val else "false"

        cfg = self.config
        content = f"""[Tester]
Expert=DMatrix-EA.ex5
Symbol={cfg.symbol}
Period={cfg.timeframe}
Optimization=0
Model=4
FromDate={cfg.from_date}
ToDate={cfg.to_date}
ForwardMode=0
Deposit=1000000000000000
Currency=USD
ProfitInPips=0
Leverage=500
ExecutionMode=0
OptimizationCriterion=0
Visual=0
ShutdownTerminal={cfg.shutdown_terminal}

[TesterInputs]
InpUseADX={to_bool_str(cfg.use_adx)}
InpUseATR={to_bool_str(cfg.use_atr)}
InpUseBands={to_bool_str(cfg.use_bands)}
InpUseMACD={to_bool_str(cfg.use_macd)}
InpUseFastMA={to_bool_str(cfg.use_fast_ma)}
InpUseSlowMA={to_bool_str(cfg.use_slow_ma)}
InpUseRSI={to_bool_str(cfg.use_rsi)}
InpUseStochastic={to_bool_str(cfg.use_stochastic)}
InpUseCandlestick={to_bool_str(cfg.use_candlestick)}
InpUseTimestampWeek={to_bool_str(cfg.use_timestamp_week)}
InpUseTimestampDay={to_bool_str(cfg.use_timestamp_day)}
InpUseOpenMarkets={to_bool_str(cfg.use_open_markets)}
InpUseSpread={to_bool_str(cfg.use_spread)}
InpUseGarchFeatures={to_bool_str(cfg.use_garch_features)}
InpFeatureLookback={cfg.feature_lookback}
InpGarchHorizon={cfg.garch_horizon}
InpPriceSize={cfg.price_size}
InpGarchAlpha={cfg.garch_alpha}
InpGarchBeta={cfg.garch_beta}
InpLabelHorizonBars={cfg.label_horizon_bars}
InpLabelMinPoints={cfg.label_min_points}
InpLabelMaxAdversePoints={cfg.label_max_adverse_points}
InpTradeMonday={to_bool_str(cfg.trade_monday)}
InpMondayStartTime={cfg.trade_monday_start}
InpMondayEndTime={cfg.trade_monday_end}
InpTradeTuesday={to_bool_str(cfg.trade_tuesday)}
InpTuesdayStartTime={cfg.trade_tuesday_start}
InpTuesdayEndTime={cfg.trade_tuesday_end}
InpTradeWednesday={to_bool_str(cfg.trade_wednesday)}
InpWednesdayStartTime={cfg.trade_wednesday_start}
InpWednesdayEndTime={cfg.trade_wednesday_end}
InpTradeThursday={to_bool_str(cfg.trade_thursday)}
InpThursdayStartTime={cfg.trade_thursday_start}
InpThursdayEndTime={cfg.trade_thursday_end}
InpTradeFriday={to_bool_str(cfg.trade_friday)}
InpFridayStartTime={cfg.trade_friday_start}
InpFridayEndTime={cfg.trade_friday_end}
InpAvoidPandemicTime={to_bool_str(cfg.avoid_pandemictime)}
InpPandemicStartTime={cfg.pandemic_start_date}
InpPandemicEndTime={cfg.pandemic_end_date}
InpLotSize=0.01
InpADXPeriod={cfg.adx_period}
InpATRPeriod={cfg.atr_period}
InpBandsPeriod={cfg.bands_period}
InpBandsShift={cfg.bands_shift}
InpBandsDev={cfg.bands_dev}
InpBandsAppliedPrice={cfg.bands_applied_price}
InpMACDFastPeriod={cfg.macd_fast}
InpMACDSlowPeriod={cfg.macd_slow}
InpMACDSignalPeriod={cfg.macd_signal}
InpMACDAppliedPrice={cfg.macd_applied_price}
InpFastMAPeriod={cfg.fast_ma_period}
InpFastMAShift={cfg.fast_ma_shift}
InpFastMAMethod={cfg.fast_ma_method}
InpFastMAAppliedPrice={cfg.fast_ma_applied_price}
InpSlowMAPeriod={cfg.slow_ma_period}
InpSlowMAShift={cfg.slow_ma_shift}
InpSlowMAMethod={cfg.slow_ma_method}
InpSlowMAAppliedPrice={cfg.slow_ma_applied_price}
InpRSIPeriod={cfg.rsi_period}
InpRSIAppliedPrice={cfg.rsi_applied_price}
InpStochK={cfg.stoch_k}
InpStochD={cfg.stoch_d}
InpStochSlowing={cfg.stoch_slowing}
InpStochMethod={cfg.stoch_method}
InpStochPriceField={cfg.stoch_price_field}
"""
        with open(ini_path, "w", encoding="ascii") as f:
            f.write(content)

        print(f"[+] Created Strategy Tester config (.ini): {ini_path}")
        return ini_path

    def _terminate_running_mt5(self) -> None:
        """Terminate any background MT5 terminal processes to avoid single-instance lock."""
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass
        time.sleep(0.5)
        try:
            subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True, text=True, check=False)
            time.sleep(1.0)
        except Exception:
            pass

    def run_strategy_tester(self, ini_path: Path) -> bool:
        """Execute MT5 Strategy Tester subprocess, stream logs in real-time, and detect fatal errors."""
        print("\n" + "=" * 80)
        print("STAGE 2: STRATEGY TESTER BACKTEST & DATASET GENERATION")
        print("=" * 80)
        print(f"[*] Running Strategy Tester with configuration (.ini): {ini_path}")
        print(
            f"    Symbol: {self.config.symbol} | Timeframe: {self.config.timeframe} | "
            f"Dates: {self.config.from_date} to {self.config.to_date}"
        )

        self._terminate_running_mt5()

        # Initialize log file byte offsets to tail only newly produced log lines
        file_offsets: dict[Path, int] = {}
        for log_dir in self._get_tester_log_directories():
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    try:
                        file_offsets[log_file] = log_file.stat().st_size
                    except OSError:
                        pass

        cmd = [
            str(self.config.mt5_path),
            f"/config:{ini_path.resolve()}",
        ]

        fatal_patterns = [
            "[dmatrix-ea] [error]",
            "critical runtime error",
            "cannot load expert",
            "zero divide",
            "array out of range",
            "pointer cannot be used",
        ]

        def check_lines(lines: list[str], proc_to_kill: subprocess.Popen) -> None:
            for line in lines:
                print(f"    [MT5] {line}")
                lower_line = line.lower()
                is_warmup = (
                    "[warmup]" in lower_line
                    or "[warning]" in lower_line
                    or "market closed" in lower_line
                    or "no real ticks" in lower_line
                    or "real ticks discarded" in lower_line
                    or "real ticks absent" in lower_line
                    or "insufficient historical rates" in lower_line
                    or "waiting for history buffer" in lower_line
                    or "insufficient rates" in lower_line
                    or "invalid stops" in lower_line
                )
                if not is_warmup:
                    for pattern in fatal_patterns:
                        if pattern in lower_line:
                            print(f"\n[FATAL ERROR DETECTED IN MT5 LOGS] {line}")
                            proc_to_kill.kill()
                            self._terminate_running_mt5()
                            raise RuntimeError(
                                f"Critical error detected during MT5 Strategy Tester execution: {line}"
                            )

        try:
            timeout = self.config.backtest_timeout
            proc = subprocess.Popen(cmd)
            wait_msg = f" (Timeout: {timeout}s)" if timeout > 0 else " (Infinite / Manual Control)"
            print(
                f"[+] MT5 Terminal started (PID: {proc.pid}). Streaming live tester logs{wait_msg}..."
            )

            start_time = time.time()
            poll_interval = 0.5

            while True:
                try:
                    time.sleep(poll_interval)
                except KeyboardInterrupt:
                    print("\n[!] Backtest interrupted by user (Ctrl+C). Terminating MT5...")
                    proc.kill()
                    self._terminate_running_mt5()
                    return False

                # Stream and inspect new log lines
                new_lines = self._stream_new_tester_logs(file_offsets)
                if new_lines:
                    check_lines(new_lines, proc)

                elapsed = int(time.time() - start_time)
                retcode = proc.poll()

                if retcode is not None:
                    time.sleep(1.0)
                    # Flush any final logs written upon exit
                    final_lines = self._stream_new_tester_logs(file_offsets)
                    if final_lines:
                        check_lines(final_lines, proc)

                    if retcode != 0:
                        print(f"    [!] Warning: MT5 Terminal exited with return code {retcode}")
                    else:
                        print(
                            f"    [+] MT5 Strategy Tester process finished with return code {retcode} "
                            f"(Total: {elapsed}s)."
                        )
                    return True

                if timeout > 0 and elapsed >= timeout:
                    print(f"[ERROR] MT5 Strategy Tester timed out after {timeout} seconds.")
                    proc.kill()
                    self._terminate_running_mt5()
                    return False

        except RuntimeError:
            raise
        except Exception as exc:
            print(f"[ERROR] Failed to execute MT5 Strategy Tester: {exc}")
            return False

    def shutdown(self) -> None:
        """Shutdown MT5 Python API connection."""
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass
