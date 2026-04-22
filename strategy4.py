"""
选股策略 4（对应「核心指标...txt」第 27–34 行）：趋势开始走强 + 控制追高 + MACD/量能确认。

复用 strategy_common；命令行入口见 main.py（本文件可直接运行，默认仅策略 4）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from strategy_common import (
    IndicatorParams,
    close_lte_ma20_multiple,
    enrich_indicators,
    fmt_date,
    fmt_num,
)


@dataclass
class Strategy4Params(IndicatorParams):
    """策略 4 可调参数。"""

    # （2）-6% < (MA5 - MA20)/MA20 < 10%
    ma5_vs_ma20_low: float = -0.06
    ma5_vs_ma20_high: float = 0.10

    # （3）收盘价 <= MA20 * 1.10
    max_close_vs_ma20: float = 1.10

    # （4）MACD：DIF > DEA > -0.08，且 DIF > 0
    min_dea: float = -0.08

    # （4）柱能：今日柱 > 过去4日柱均值
    hist_lookback_days: int = 4

    # （5）量能：VOL_MA5 > VOL_MA20 * 110%
    vol_ma5_vs_ma20: float = 1.10

    # （1）MA60 > 3 日前 MA60
    ma60_compare_lag: int = 3


def min_bars_required_s4(p: Strategy4Params) -> int:
    # ma60 至少需要 ma60 + lag；hist 需要 lookback+1；均线/量能需要 20；MACD 稳定期保守取 slow+signal+5
    return max(
        p.ma60 + p.ma60_compare_lag + 1,
        p.ma_long + 1,
        p.ma_short + 1,
        p.hist_lookback_days + 2,
        p.macd_slow + p.macd_signal + 5,
    )


def _ma5_ma20_deviation_ok(row: pd.Series, p: Strategy4Params) -> bool:
    ma5 = float(row["ma5"])
    ma20 = float(row["ma20"])
    if pd.isna(ma5) or pd.isna(ma20) or ma20 == 0:
        return False
    dev = (ma5 - ma20) / ma20
    return bool(p.ma5_vs_ma20_low < dev < p.ma5_vs_ma20_high)


def _macd_ok(d: pd.DataFrame, p: Strategy4Params) -> bool:
    row = d.iloc[-1]
    if pd.isna(row["dif"]) or pd.isna(row["dea"]) or pd.isna(row["macd_hist"]):
        return False
    if not (row["dif"] > row["dea"] > p.min_dea and row["dif"] > 0):
        return False
    if not (row["macd_hist"] > 0):
        return False
    hist_prev_mean = (
        d["macd_hist"].shift(1).rolling(p.hist_lookback_days, min_periods=p.hist_lookback_days).mean().iloc[-1]
    )
    if pd.isna(hist_prev_mean):
        return False
    return bool(row["macd_hist"] > hist_prev_mean)


def screen_strategy4_last_row(
    df: pd.DataFrame, p: Optional[Strategy4Params] = None
) -> bool:
    p = p or Strategy4Params()
    if df is None or len(df) < min_bars_required_s4(p):
        return False
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    lag = int(p.ma60_compare_lag)

    if not (row["close"] > row["ma60"]):
        return False
    ma60_lag = d["ma60"].iloc[-(lag + 1)]
    if pd.isna(ma60_lag) or pd.isna(row["ma60"]) or not (row["ma60"] > ma60_lag):
        return False

    if not (row["ma5"] > row["ma10"]):
        return False
    if not _ma5_ma20_deviation_ok(row, p):
        return False

    if not close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    ):
        return False

    if not _macd_ok(d, p):
        return False

    if pd.isna(row["vol_ma5"]) or pd.isna(row["vol_ma20"]):
        return False
    if not (row["vol_ma5"] > p.vol_ma5_vs_ma20 * row["vol_ma20"]):
        return False

    return True


def screen_strategy4_detail(
    df: pd.DataFrame, p: Optional[Strategy4Params] = None
) -> tuple[bool, dict]:
    p = p or Strategy4Params()
    checks: dict[str, bool] = {}
    if df is None or len(df) < min_bars_required_s4(p):
        checks["enough_bars"] = False
        return False, checks
    checks["enough_bars"] = True

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    lag = int(p.ma60_compare_lag)
    ma60_lag = d["ma60"].iloc[-(lag + 1)] if len(d) >= lag + 1 else float("nan")

    checks["(1a)_close_above_ma60"] = bool(row["close"] > row["ma60"])
    checks["(1b)_ma60_gt_ma60_lag3"] = bool(
        (not pd.isna(row["ma60"]))
        and (not pd.isna(ma60_lag))
        and (row["ma60"] > ma60_lag)
    )

    checks["(2a)_ma5_gt_ma10"] = bool(row["ma5"] > row["ma10"])
    checks["(2b)_ma5_vs_ma20_dev_in_range"] = _ma5_ma20_deviation_ok(row, p)

    checks["(3)_close_vs_ma20_cap"] = close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    )

    checks["(4)_macd_rule"] = _macd_ok(d, p)

    checks["(5)_vol_ma5_vs_ma20"] = bool(
        (not pd.isna(row["vol_ma5"]))
        and (not pd.isna(row["vol_ma20"]))
        and (row["vol_ma5"] > p.vol_ma5_vs_ma20 * row["vol_ma20"])
    )

    all_ok = all(checks[k] for k in checks if k != "enough_bars")
    return all_ok, checks


def strategy4_calculation_report(
    df: pd.DataFrame,
    p: Optional[Strategy4Params] = None,
    *,
    title: str = "",
) -> str:
    p = p or Strategy4Params()
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    if title:
        ln(title)
    need = min_bars_required_s4(p)
    if df is None or len(df) < need:
        ln(
            f"【数据长度】当前 {0 if df is None else len(df)} 根K线，"
            f"策略4至少需要 {need} 根。"
        )
        return "\n".join(lines)

    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    lag = int(p.ma60_compare_lag)
    ma60_lag = d["ma60"].iloc[-(lag + 1)]

    ln(f"【基准日】{fmt_date(d.index[-1])}  （MA60 对比 T-{lag}）")
    ln(f"  收盘={fmt_num(row['close'], 4)}  成交量={fmt_num(row['volume'], 4)}")
    ln("")

    ln("（1）收盘价 > MA60；且 MA60 > 3 日前的 MA60")
    ln(
        f"  MA60={fmt_num(row['ma60'], 4)}  MA60(T-{lag})={fmt_num(ma60_lag, 4)}  "
        f"收盘>MA60 → {row['close'] > row['ma60']}  "
        f"MA60抬高 → {bool((not pd.isna(row['ma60'])) and (not pd.isna(ma60_lag)) and (row['ma60'] > ma60_lag))}"
    )
    ln("")

    ln("（2）MA5 > MA10；且 -6% < (MA5 - MA20)/MA20 < 10%")
    ma5 = float(row["ma5"])
    ma20 = float(row["ma20"])
    dev = (ma5 - ma20) / ma20 if (not pd.isna(ma5) and not pd.isna(ma20) and ma20 != 0) else float("nan")
    ln(
        f"  MA5={fmt_num(row['ma5'], 4)}  MA10={fmt_num(row['ma10'], 4)}  MA20={fmt_num(row['ma20'], 4)}"
    )
    ln(
        f"  MA5>MA10 → {row['ma5'] > row['ma10']}  "
        f"偏离度={(fmt_num(dev * 100, 4) if not pd.isna(dev) else 'NaN')}%  "
        f"区间({p.ma5_vs_ma20_low*100:.0f}%, {p.ma5_vs_ma20_high*100:.0f}%) → {_ma5_ma20_deviation_ok(row, p)}"
    )
    ln("")

    ln("（3）收盘价 ≤ MA20 × 110%")
    cap = p.max_close_vs_ma20 * row["ma20"]
    ln(f"  上限={fmt_num(cap, 4)}  → {close_lte_ma20_multiple(float(row['close']), float(row['ma20']), p.max_close_vs_ma20)}")
    ln("")

    ln("（4）MACD：DIF > DEA > -0.08 且 DIF>0；柱能>0，且今日柱>过去4日柱均值")
    hist_prev_mean = (
        d["macd_hist"]
        .shift(1)
        .rolling(p.hist_lookback_days, min_periods=p.hist_lookback_days)
        .mean()
        .iloc[-1]
    )
    ln(
        f"  DIF={fmt_num(row['dif'], 6)}  DEA={fmt_num(row['dea'], 6)}  柱={fmt_num(row['macd_hist'], 6)}"
    )
    ln(
        f"  过去{p.hist_lookback_days}日柱均值(不含今日)={fmt_num(hist_prev_mean, 6)}  "
        f"满足 → {_macd_ok(d, p)}"
    )
    ln("")

    ln("（5）量能：5 日成交量均值 > 20 日成交量均值 × 110%")
    thr_v = p.vol_ma5_vs_ma20 * row["vol_ma20"]
    ln(
        f"  VOL_MA5={fmt_num(row['vol_ma5'], 4)}  VOL_MA20={fmt_num(row['vol_ma20'], 4)}  "
        f"阈值={fmt_num(thr_v, 4)}  满足 → {bool((not pd.isna(row['vol_ma5'])) and (not pd.isna(row['vol_ma20'])) and (row['vol_ma5'] > thr_v))}"
    )
    ln("")

    all_ok, _ = screen_strategy4_detail(df, p)
    ln(f"【策略4 总判定】{'通过' if all_ok else '未通过'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from main import main

    av = sys.argv[1:]
    if not any(x == "--strategies" or x.startswith("--strategies=") for x in av):
        av = ["--strategies", "4", *av]
    raise SystemExit(main(av))

