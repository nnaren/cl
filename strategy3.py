"""
选股策略 3（对应「核心指标...txt」第 18–24 行）：MACD 0 轴上金叉。

复用 strategy_common；命令行入口见 main.py（本文件可直接运行，默认仅策略 3）。
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
    mas_group_rising_daily,
    macd_golden_cross_above_zero,
    volume_gt_prev_ma5,
)


@dataclass
class Strategy3Params(IndicatorParams):
    """策略 3：与策略 1 相同的 20 日偏离上限默认 110%。"""

    max_close_vs_ma20: float = 1.10
    ma_trend_days: int = 5


def min_bars_required_s3(p: Strategy3Params) -> int:
    return max(
        p.ma60 + 1,
        p.ma_trend_days + p.ma_long + 5,
        p.macd_slow + p.macd_signal + 5,
        p.ma_short + 6,
        2,
    )


def screen_strategy3_last_row(
    df: pd.DataFrame, p: Optional[Strategy3Params] = None
) -> bool:
    p = p or Strategy3Params()
    if df is None or len(df) < min_bars_required_s3(p):
        return False
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]

    if not (row["close"] > row["ma60"]):
        return False

    if not (row["ma5"] > row["ma10"] > row["ma20"]):
        return False
    if not mas_group_rising_daily(
        d["ma5"], d["ma10"], d["ma20"], days=p.ma_trend_days
    ):
        return False

    if not close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    ):
        return False

    if not macd_golden_cross_above_zero(d):
        return False

    if not volume_gt_prev_ma5(row, ratio=1.0):
        return False

    return True


def screen_strategy3_detail(
    df: pd.DataFrame, p: Optional[Strategy3Params] = None
) -> tuple[bool, dict]:
    p = p or Strategy3Params()
    checks: dict[str, bool] = {}
    if df is None or len(df) < min_bars_required_s3(p):
        checks["enough_bars"] = False
        return False, checks
    checks["enough_bars"] = True

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]

    checks["(1)_close_above_ma60"] = bool(row["close"] > row["ma60"])
    checks["(2a)_ma_order"] = bool(row["ma5"] > row["ma10"] > row["ma20"])
    checks["(2b)_mas_rising_5d"] = mas_group_rising_daily(
        d["ma5"], d["ma10"], d["ma20"], days=p.ma_trend_days
    )
    checks["(3)_close_vs_ma20_cap"] = close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    )
    checks["(4)_macd_golden_cross_above_zero"] = macd_golden_cross_above_zero(d)
    checks["(5)_volume_gt_prev_ma5"] = volume_gt_prev_ma5(row, ratio=1.0)

    all_ok = all(checks[k] for k in checks if k != "enough_bars")
    return all_ok, checks


def strategy3_calculation_report(
    df: pd.DataFrame,
    p: Optional[Strategy3Params] = None,
    *,
    title: str = "",
) -> str:
    p = p or Strategy3Params()
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    if title:
        ln(title)
    need = min_bars_required_s3(p)
    if df is None or len(df) < need:
        ln(
            f"【数据长度】当前 {0 if df is None else len(df)} 根K线，"
            f"策略3至少需要 {need} 根。"
        )
        return "\n".join(lines)

    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]
    prev = d.iloc[-2]

    ln(f"【基准日】{fmt_date(d.index[-1])}  （昨 {fmt_date(d.index[-2])}）")
    ln(f"  收盘={fmt_num(row['close'], 4)}  成交量={fmt_num(row['volume'], 4)}")
    ln("")

    ln("（1）收盘价 > 60 日均价")
    ln(f"  MA60={fmt_num(row['ma60'], 4)}  → {row['close'] > row['ma60']}")
    ln("")

    ln("（2）MA5>MA10>MA20；且最近 5 日三线各自逐日抬高")
    ln(
        f"  MA5={fmt_num(row['ma5'], 4)} MA10={fmt_num(row['ma10'], 4)} "
        f"MA20={fmt_num(row['ma20'], 4)}"
    )
    need_m = p.ma_trend_days + 1
    tail = d.iloc[-need_m:]
    for label, ser in [("MA5", d["ma5"]), ("MA10", d["ma10"]), ("MA20", d["ma20"])]:
        parts = [
            f"{fmt_date(tail.index[i])}:{fmt_num(ser.loc[tail.index[i]], 4)}"
            for i in range(len(tail))
        ]
        ln(f"  {label}: " + " | ".join(parts))
    ln(
        f"  5 日逐日抬高 → "
        f"{mas_group_rising_daily(d['ma5'], d['ma10'], d['ma20'], days=p.ma_trend_days)}"
    )
    ln("")

    cap = p.max_close_vs_ma20 * row["ma20"]
    ok3 = close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    )
    ln("（3）收盘 ≤ MA20 × {:.0f}%".format(p.max_close_vs_ma20 * 100))
    ln(f"  上限={fmt_num(cap, 4)}  → {ok3}")
    ln("")

    ln("（4）MACD 0 轴上金叉：昨 DIF<DEA，今 DIF>DEA；今 DIF>0 且 DEA>0")
    ln(
        f"  昨 DIF={fmt_num(prev['dif'], 6)} DEA={fmt_num(prev['dea'], 6)}  "
        f"DIF<DEA → {prev['dif'] < prev['dea']}"
    )
    ln(
        f"  今 DIF={fmt_num(row['dif'], 6)} DEA={fmt_num(row['dea'], 6)}  "
        f"DIF>DEA → {row['dif'] > row['dea']}  "
        f"DIF>0且DEA>0 → {row['dif'] > 0 and row['dea'] > 0}"
    )
    ln(f"  金叉总判 → {macd_golden_cross_above_zero(d)}")
    ln("")

    ln("（5）当日成交量 > 过去 5 个交易日均量（不含当日）")
    ln(f"  vol_ma5_prev={fmt_num(row['vol_ma5_prev'], 4)}")
    ln(f"  当日量 > 均量 → {volume_gt_prev_ma5(row, ratio=1.0)}")
    ln("")

    all_ok, _ = screen_strategy3_detail(df, p)
    ln(f"【策略3 总判定】{'通过' if all_ok else '未通过'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from main import main

    av = sys.argv[1:]
    if not any(x == "--strategies" or x.startswith("--strategies=") for x in av):
        av = ["--strategies", "3", *av]
    raise SystemExit(main(av))
