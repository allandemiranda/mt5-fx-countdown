"""MetaTrader 5 Chart Template (.tpl) generator synchronized with active feature schema."""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.config import AppConfig


class TemplateGenerator:
    """Generates native MT5 chart template (.tpl) with customized candlestick colors and active indicators."""

    def __init__(self, config: AppConfig, terminal_data_path: Path, common_path: Path):
        self.config = config
        self.terminal_data_path = terminal_data_path
        self.common_path = common_path

    def _map_period_size(self) -> tuple[int, int]:
        """Map clean timeframe to MT5 period_type and period_size."""
        tf = self.config.clean_timeframe.upper()
        if tf == "M1":
            return 1, 1
        elif tf == "M5":
            return 1, 5
        elif tf == "M15":
            return 1, 15
        elif tf == "M30":
            return 1, 30
        elif tf == "H1":
            return 1, 60
        elif tf == "H2":
            return 2, 2
        elif tf == "H4":
            return 2, 4
        elif tf == "D1":
            return 3, 1
        elif tf == "W1":
            return 4, 1
        elif tf == "MN1" or tf == "MN":
            return 5, 1
        return 1, 60

    def build_template_content(self) -> str:
        """Construct full parameter content for MT5 Chart Template (.tpl)."""
        cfg = self.config
        sym = cfg.symbol
        period_type, period_size = self._map_period_size()

        # Build Main Window Indicators
        main_indicators: List[str] = [
            """<indicator>
name=Main
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1
</indicator>"""
        ]

        # 1. Bollinger Bands (style=2 dashed line, color=13749760 cyan/dark turquoise)
        if cfg.use_bands:
            main_indicators.append(
                f"""<indicator>
name=Bollinger Bands
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=131
style=2
width=1
color=13749760
</graph>

<graph>
name=
draw=131
style=2
width=1
color=13749760
</graph>

<graph>
name=
draw=131
style=2
width=1
color=13749760
</graph>
period={cfg.bands_period}
deviation={cfg.bands_dev:.6f}
shift={cfg.bands_shift}
applied_price={cfg.bands_applied_price}
</indicator>"""
            )

        # 2. Fast Moving Average (style=1 dashed line, width=1, color=65535 yellow)
        if cfg.use_fast_ma:
            main_indicators.append(
                f"""<indicator>
name=Moving Average
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=129
style=1
width=1
color=65535
</graph>
period={cfg.fast_ma_period}
shift={cfg.fast_ma_shift}
method={cfg.fast_ma_method}
applied_price={cfg.fast_ma_applied_price}
</indicator>"""
            )

        # 3. Slow Moving Average (style=1 dashed line, width=1, color=16711935 magenta)
        if cfg.use_slow_ma:
            main_indicators.append(
                f"""<indicator>
name=Moving Average
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=129
style=1
width=1
color=16711935
</graph>
period={cfg.slow_ma_period}
shift={cfg.slow_ma_shift}
method={cfg.slow_ma_method}
applied_price={cfg.slow_ma_applied_price}
</indicator>"""
            )

        # Build Sub-Windows
        sub_windows: List[str] = []

        # 4. MACD
        if cfg.use_macd:
            sub_windows.append(
                f"""<window>
height=50
objects=0

<indicator>
name=MACD
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=128
style=0
width=1
arrow=251
color=12632256
</graph>

<graph>
name=
draw=1
style=2
width=1
arrow=251
color=255
</graph>
fast_ema={cfg.macd_fast}
slow_ema={cfg.macd_slow}
macd_sma={cfg.macd_signal}
applied_price={cfg.macd_applied_price}
</indicator>
</window>"""
            )

        # 5. RSI
        if cfg.use_rsi:
            sub_windows.append(
                f"""<window>
height=50
objects=0

<indicator>
name=Relative Strength Index
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=1
scale_fix_min_val=0.000000
scale_fix_max=1
scale_fix_max_val=100.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=1
style=0
width=2
arrow=251
color=16748574
</graph>

<level>
level=30.000000
style=2
color=12632256
width=1
descr=
</level>

<level>
level=70.000000
style=2
color=12632256
width=1
descr=
</level>
period={cfg.rsi_period}
applied_price={cfg.rsi_applied_price}
</indicator>
</window>"""
            )

        # 6. Stochastic
        if cfg.use_stochastic:
            sub_windows.append(
                f"""<window>
height=50
objects=0

<indicator>
name=Stochastic Oscillator
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=1
scale_fix_min_val=0.000000
scale_fix_max=1
scale_fix_max_val=100.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=1
style=0
width=1
arrow=251
color=3329330
</graph>

<graph>
name=
draw=1
style=2
width=1
arrow=251
color=255
</graph>

<level>
level=20.000000
style=2
color=12632256
width=1
descr=
</level>

<level>
level=80.000000
style=2
color=12632256
width=1
descr=
</level>
kperiod={cfg.stoch_k}
dperiod={cfg.stoch_d}
slowing={cfg.stoch_slowing}
price_apply={cfg.stoch_price_field}
method={cfg.stoch_method}
</indicator>
</window>"""
            )

        # 7. ATR
        if cfg.use_atr:
            sub_windows.append(
                f"""<window>
height=45
objects=0

<indicator>
name=Average True Range
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=1
style=0
width=2
arrow=251
color=14772545
</graph>
period={cfg.atr_period}
</indicator>
</window>"""
            )

        # 8. ADX
        if cfg.use_adx:
            sub_windows.append(
                f"""<window>
height=50
objects=0

<indicator>
name=Average Directional Movement Index
path=
apply=1
show_data=1
scale_inherit=0
scale_line=0
scale_line_percent=50
scale_line_value=0.000000
scale_fix_min=0
scale_fix_min_val=0.000000
scale_fix_max=0
scale_fix_max_val=0.000000
expertmode=0
fixed_height=-1

<graph>
name=
draw=1
style=0
width=2
arrow=251
color=11186720
</graph>

<graph>
name=
draw=1
style=2
width=1
arrow=251
color=3329434
</graph>

<graph>
name=
draw=1
style=2
width=1
arrow=251
color=11788021
</graph>
period={cfg.adx_period}
</indicator>
</window>"""
            )

        main_window_content = "\n\n".join(main_indicators)
        sub_windows_content = "\n\n".join(sub_windows)

        # Bullish Green: 65280, Bearish Red: 255, Grid: 0
        return f"""<chart>
symbol={sym}
period_type={period_type}
period_size={period_size}
digits=5
tick_size=0.000000
position_time=0
scale_fix=0
scale_fix11=0
scale_bar=0
scale_bar_val=1.000000
scale=8
mode=1
fore=0
grid=0
volume=0
scroll=1
shift=1
shift_size=20.000000
fixed_pos=0.000000
ticker=1
ohlc=0
one_click=0
one_click_btn=1
bidline=1
askline=1
lastline=0
days=1
descriptions=0
tradelines=1
tradehistory=1
window_left=0
window_top=0
window_right=0
window_bottom=0
window_type=1
background_color=0
foreground_color=16777215
barup_color=65280
bardown_color=255
bullcandle_color=65280
bearcandle_color=255
chartline_color=65280
volumes_color=3329330
grid_color=10061943
bidline_color=10061943
askline_color=255
lastline_color=49152
stops_color=255

<window>
height=100
objects=0

{main_window_content}
</window>

{sub_windows_content}
</chart>
"""

    def generate_all(self, target_directories: List[Path] | None = None) -> Path:
        """Generate and write template file to all target directories, replacing old files cleanly."""
        sym = self.config.symbol
        tf = self.config.clean_timeframe
        template_name = f"{sym}_{tf}.tpl"
        content = self.build_template_content()

        dirs = target_directories if target_directories is not None else [
            self.terminal_data_path / "MQL5" / "Profiles" / "Templates",
            self.terminal_data_path / "Profiles" / "Templates",
            self.common_path / "Files" / "Templates",
        ]

        primary_path: Path | None = None
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            out_file = d / template_name
            if out_file.exists():
                try:
                    out_file.unlink()
                except Exception:
                    pass
            out_file.write_text(content, encoding="ascii")
            print(f"    [+] Saved Chart Template (.tpl): {out_file}")
            if primary_path is None:
                primary_path = out_file

        print(f"[+] Chart template generation complete: '{template_name}'")
        return primary_path if primary_path is not None else Path(template_name)
