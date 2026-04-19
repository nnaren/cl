"""
选股策略 2（对应「核心指标...txt」第 10–16 行）：放量突破 20 日均线。

复用 strategy_common；命令行入口见 main.py（本文件可直接运行，默认仅策略 2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from strategy_common import (
    IndicatorParams,
    enrich_indicators,
    fmt_date,
    fmt_num,
    mas_group_rising_daily,
    macd_histogram_positive_rising,
    volume_gt_prev_ma5,
)


@dataclass
class Strategy2Params(IndicatorParams):
    """策略 2 可调参数。"""

    # （4）当日成交量 > 过去 5 个交易日均量 × 该倍数（默认 180%）
    vol_spike_vs_prev_ma5: float = 1.80
    ma_trend_days: int = 5
    macd_trend_days: int = 3


def min_bars_required_s2(p: Strategy2Params) -> int:
    return max(
        p.ma60 + 1,
        p.ma_trend_days + p.ma_mid + 2,
        p.macd_slow + p.macd_signal + p.macd_trend_days + 5,
        p.ma_short + 6,
        p.ma_long + 5,
    )


def screen_strategy2_last_row(
    df: pd.DataFrame, p: Optional[Strategy2Params] = None
) -> bool:
    p = p or Strategy2Params()
    if df is None or len(df) < min_bars_required_s2(p):
        return False
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    prev = d.iloc[-2]

    if not (row["close"] > row["ma60"]):
        return False

    if not (row["ma5"] > row["ma10"]):
        return False
    if not mas_group_rising_daily(d["ma5"], d["ma10"], days=p.ma_trend_days):
        return False

    if not (prev["close"] < prev["ma20"] and row["close"] > row["ma20"]):
        return False

    if not volume_gt_prev_ma5(row, ratio=p.vol_spike_vs_prev_ma5):
        return False

    if not macd_histogram_positive_rising(d["macd_hist"], p.macd_trend_days):
        return False

    return True


def screen_strategy2_detail(
    df: pd.DataFrame, p: Optional[Strategy2Params] = None
) -> tuple[bool, dict]:
    p = p or Strategy2Params()
    checks: dict[str, bool] = {}
    if df is None or len(df) < min_bars_required_s2(p):
        checks["enough_bars"] = False
        return False, checks
    checks["enough_bars"] = True

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    prev = d.iloc[-2]

    checks["(1)_close_above_ma60"] = bool(row["close"] > row["ma60"])
    checks["(2a)_ma5_gt_ma10"] = bool(row["ma5"] > row["ma10"])
    checks["(2b)_ma5_ma10_rising_5d"] = mas_group_rising_daily(
        d["ma5"], d["ma10"], days=p.ma_trend_days
    )
    checks["(3)_break_ma20"] = bool(
        prev["close"] < prev["ma20"] and row["close"] > row["ma20"]
    )
    checks["(4)_volume_spike"] = volume_gt_prev_ma5(
        row, ratio=p.vol_spike_vs_prev_ma5
    )
    checks["(5)_macd_hist_bull"] = macd_histogram_positive_rising(
        d["macd_hist"], p.macd_trend_days
    )

    all_ok = all(checks[k] for k in checks if k != "enough_bars")
    return all_ok, checks


def strategy2_calculation_report(
    df: pd.DataFrame,
    p: Optional[Strategy2Params] = None,
    *,
    title: str = "",
) -> str:
    p = p or Strategy2Params()
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    if title:
        ln(title)
    need = min_bars_required_s2(p)
    if df is None or len(df) < need:
        ln(
            f"【数据长度】当前 {0 if df is None else len(df)} 根K线，"
            f"策略2至少需要 {need} 根。"
        )
        return "\n".join(lines)

    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    prev = d.iloc[-2]

    ln(f"【基准日】{fmt_date(d.index[-1])}  （T-1 为 {fmt_date(d.index[-2])}）")
    ln(f"  T 收盘={fmt_num(row['close'], 4)}  成交量={fmt_num(row['volume'], 4)}")
    ln(
        f"  T-1 收盘={fmt_num(prev['close'], 4)}  MA20(T-1)={fmt_num(prev['ma20'], 4)}"
    )
    ln("")

    ln("（1）收盘价 > 60 日均价")
    ln(f"  MA60={fmt_num(row['ma60'], 4)}  收盘>MA60 → {row['close'] > row['ma60']}")
    ln("")

    ln("（2）MA5 > MA10；且最近 5 日 MA5/MA10 各自逐日抬高")
    ln(
        f"  MA5={fmt_num(row['ma5'], 4)}  MA10={fmt_num(row['ma10'], 4)}  "
        f"MA5>MA10 → {row['ma5'] > row['ma10']}"
    )
    need_m = p.ma_trend_days + 1
    tail = d.iloc[-need_m:]
    for label, ser in [("MA5", d["ma5"]), ("MA10", d["ma10"])]:
        parts = [f"{fmt_date(tail.index[i])}:{fmt_num(ser.loc[tail.index[i]], 4)}" for i in range(len(tail))]
        ln(f"  {label}: " + " | ".join(parts))
    ok2b = mas_group_rising_daily(d["ma5"], d["ma10"], days=p.ma_trend_days)
    ln(f"  5 日逐日抬高 → {ok2b}")
    ln("")

    ln("（3）T-1 收盘 < MA20(T-1)，且 T 收盘 > MA20(T)")
    ln(
        f"  T-1: 收盘<{fmt_num(prev['ma20'], 4)} → {prev['close'] < prev['ma20']}  "
        f"({fmt_num(prev['close'], 4)} < {fmt_num(prev['ma20'], 4)})"
    )
    ln(
        f"  T:   收盘>MA20 → {row['close'] > row['ma20']}  "
        f"({fmt_num(row['close'], 4)} > {fmt_num(row['ma20'], 4)})"
    )
    ln("")

    thr_v = p.vol_spike_vs_prev_ma5 * row["vol_ma5_prev"]
    ln("（4）当日成交量 > 过去 5 个交易日均量 × {:.0f}%".format(p.vol_spike_vs_prev_ma5 * 100))
    ln(
        "  vol_ma5_prev = mean(vol[T-1]…vol[T-5]) = "
        f"{fmt_num(row['vol_ma5_prev'], 4)}"
    )
    ln(
        f"  阈值 = {fmt_num(thr_v, 4)}  当日量={fmt_num(row['volume'], 4)}  "
        f"放量 → {volume_gt_prev_ma5(row, ratio=p.vol_spike_vs_prev_ma5)}"
    )
    ln("")

    ln("（5）MACD 柱：最近 3 根均 > 0 且严格递增")
    tail_h = d["macd_hist"].iloc[-p.macd_trend_days :]
    ln(f"  柱值: {', '.join(fmt_num(x, 6) for x in tail_h)}")
    ln(f"  满足 → {macd_histogram_positive_rising(d['macd_hist'], p.macd_trend_days)}")
    ln("")

    all_ok, _ = screen_strategy2_detail(df, p)
    ln(f"【策略2 总判定】{'通过' if all_ok else '未通过'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from main import main

    av = sys.argv[1:]
    if not any(x == "--strategies" or x.startswith("--strategies=") for x in av):
        av = ["--strategies", "2", *av]
    raise SystemExit(main(av))
