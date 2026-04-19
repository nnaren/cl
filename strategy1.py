"""
选股策略 1（对应「核心指标...txt」第 1–9 行）

数据与指标计算见 strategy_common；命令行统一入口见 main.py（本文件可直接运行，默认仅策略 1）。
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
    macd_histogram_positive_rising,
    strictly_increasing_last_n,
)


@dataclass
class Strategy1Params(IndicatorParams):
    max_close_vs_ma20: float = 1.10
    vol_ma5_vs_ma60: float = 1.10
    ma_trend_days: int = 5
    macd_trend_days: int = 3


def min_bars_required(p: Strategy1Params) -> int:
    return max(
        p.ma120,
        p.ma60 + 1,
        p.macd_slow + p.macd_signal + p.macd_trend_days + 5,
        p.ma_trend_days + p.ma_long + 5,
    )


def screen_last_row(df: pd.DataFrame, p: Optional[Strategy1Params] = None) -> bool:
    p = p or Strategy1Params()
    if df is None or len(df) < min_bars_required(p):
        return False
    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]

    if not (row["close"] > row["ma120"] and row["close"] > row["ma60"]):
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

    if not (row["dif"] > row["dea"] > 0):
        return False
    if not strictly_increasing_last_n(d["dif"], p.macd_trend_days):
        return False
    if not strictly_increasing_last_n(d["dea"], p.macd_trend_days):
        return False
    if not macd_histogram_positive_rising(d["macd_hist"], p.macd_trend_days):
        return False

    if row["vol_ma5"] <= p.vol_ma5_vs_ma60 * row["vol_ma60"]:
        return False

    return True


def screen_detail_last_row(
    df: pd.DataFrame, p: Optional[Strategy1Params] = None
) -> tuple[bool, dict]:
    p = p or Strategy1Params()
    checks: dict[str, bool] = {}
    if df is None or len(df) < min_bars_required(p):
        checks["enough_bars"] = False
        return False, checks
    checks["enough_bars"] = True

    d = enrich_indicators(df.sort_index(), p)
    row = d.iloc[-1]

    c1 = bool(row["close"] > row["ma120"] and row["close"] > row["ma60"])
    checks["(1)_above_ma120_ma60"] = c1

    c2a = bool(row["ma5"] > row["ma10"] > row["ma20"])
    checks["(2a)_ma_order"] = c2a
    c2b = mas_group_rising_daily(
        d["ma5"], d["ma10"], d["ma20"], days=p.ma_trend_days
    )
    checks["(2b)_ma5_ma10_ma20_rising_5d"] = c2b

    c3 = close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    )
    checks["(3)_close_vs_ma20_cap"] = c3

    c4a = bool(row["dif"] > row["dea"] > 0)
    checks["(4a)_dif_dea_positive_order"] = c4a
    c4b = strictly_increasing_last_n(d["dif"], p.macd_trend_days)
    checks["(4b)_dif_rising_3d"] = c4b
    c4c = strictly_increasing_last_n(d["dea"], p.macd_trend_days)
    checks["(4c)_dea_rising_3d"] = c4c
    tail_hist = d["macd_hist"].iloc[-p.macd_trend_days :]
    c4d = bool(not tail_hist.isna().any() and (tail_hist > 0).all())
    checks["(4d)_hist_positive_3d"] = c4d
    c4e = strictly_increasing_last_n(d["macd_hist"], p.macd_trend_days)
    checks["(4e)_hist_rising_3d"] = c4e

    c5 = bool(row["vol_ma5"] > p.vol_ma5_vs_ma60 * row["vol_ma60"])
    checks["(5)_volume_ma5_vs_ma60"] = c5

    passed = all(checks[k] for k in checks if k != "enough_bars")
    return passed, checks


def strategy1_calculation_report(
    df: pd.DataFrame,
    p: Optional[Strategy1Params] = None,
    *,
    title: str = "",
) -> str:
    p = p or Strategy1Params()
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    if title:
        ln(title)
    need = min_bars_required(p)
    if df is None or len(df) < need:
        ln(
            f"【数据长度】当前 {0 if df is None else len(df)} 根K线，"
            f"策略至少需要 {need} 根（MA120 与 MACD 稳定期等），无法完整计算。"
        )
        return "\n".join(lines)

    if not {"close", "volume"}.issubset(df.columns):
        raise ValueError("df 必须包含列: close, volume")

    d = enrich_indicators(df.sort_index(), p)
    last_idx = d.index[-1]
    row = d.iloc[-1]

    ln(f"【基准日】{fmt_date(last_idx)}")
    ln(f"  收盘价={fmt_num(row['close'], 4)}  成交量={fmt_num(row['volume'], 4)}")
    ln("")

    ln("（1）收盘价 > 120 日均价 且 > 60 日均价")
    ln(
        f"  MA60 = {fmt_num(row['ma60'], 4)}  MA120 = {fmt_num(row['ma120'], 4)}"
    )
    ok1a = row["close"] > row["ma120"]
    ok1b = row["close"] > row["ma60"]
    ln(f"  收盘 > MA120 → {ok1a}  ({fmt_num(row['close'], 4)} > {fmt_num(row['ma120'], 4)})")
    ln(f"  收盘 > MA60  → {ok1b}  ({fmt_num(row['close'], 4)} > {fmt_num(row['ma60'], 4)})")
    ln(f"  条件(1) 总结果: {bool(ok1a and ok1b)}")
    ln("")

    ln("（2）均线多头：MA5 > MA10 > MA20；且最近 5 日各自逐日抬高")
    ln(
        f"  最新 MA5={fmt_num(row['ma5'], 4)}  MA10={fmt_num(row['ma10'], 4)}  "
        f"MA20={fmt_num(row['ma20'], 4)}"
    )
    ok2a = bool(row["ma5"] > row["ma10"] > row["ma20"])
    ln(
        f"  MA5>MA10>MA20 → {ok2a}  "
        f"({fmt_num(row['ma5'], 4)} > {fmt_num(row['ma10'], 4)} > {fmt_num(row['ma20'], 4)})"
    )
    need_m = p.ma_trend_days + 1
    tail = d.iloc[-need_m:]
    ln(f"  最近 {p.ma_trend_days} 日逐日抬高核对:")
    for name, ser in [("MA5", d["ma5"]), ("MA10", d["ma10"]), ("MA20", d["ma20"])]:
        parts = [f"{fmt_date(tail.index[i])}:{fmt_num(ser.loc[tail.index[i]], 4)}" for i in range(len(tail))]
        ln(f"    {name}: " + " | ".join(parts))
    ok2b = mas_group_rising_daily(
        d["ma5"], d["ma10"], d["ma20"], days=p.ma_trend_days
    )
    for nm, ser in [("MA5", d["ma5"]), ("MA10", d["ma10"]), ("MA20", d["ma20"])]:
        seg = ser.iloc[-need_m:]
        diffs = seg.diff().iloc[1:]
        ln(
            f"    {nm} 日差分(最近{p.ma_trend_days}次): "
            + ", ".join(fmt_num(v, 6) for v in diffs)
            + f" → 均>0: {bool((diffs > 0).all())}"
        )
    ln(f"  条件(2) 总结果: {bool(ok2a and ok2b)}")
    ln("")

    cap = p.max_close_vs_ma20 * row["ma20"]
    ok3 = close_lte_ma20_multiple(
        float(row["close"]), float(row["ma20"]), p.max_close_vs_ma20
    )
    ln("（3）最新收盘价 ≤ 20 日均价 × {:.0f}%".format(p.max_close_vs_ma20 * 100))
    ln(f"  MA20 = {fmt_num(row['ma20'], 4)}  上限 = {fmt_num(cap, 4)}")
    ln(f"  收盘 ≤ 上限 → {ok3}  ({fmt_num(row['close'], 4)} ≤ {fmt_num(cap, 4)})")
    ln(f"  条件(3) 总结果: {ok3}")
    ln("")

    ln("（4）MACD（DIF/DEA/柱）")
    ln(
        f"  参数: fast={p.macd_fast} slow={p.macd_slow} signal={p.macd_signal}  "
        f"考察最近 {p.macd_trend_days} 根K线"
    )
    tail_macd = d.iloc[-p.macd_trend_days :]
    for i in range(len(tail_macd)):
        ix = tail_macd.index[i]
        r = tail_macd.iloc[i]
        ln(
            f"  {fmt_date(ix)}  DIF={fmt_num(r['dif'], 6)}  "
            f"DEA={fmt_num(r['dea'], 6)}  柱={fmt_num(r['macd_hist'], 6)}"
        )
    ok4a = bool(row["dif"] > row["dea"] > 0)
    ln(f"  最新 DIF>DEA>0 → {ok4a}  (DIF={fmt_num(row['dif'], 6)}, DEA={fmt_num(row['dea'], 6)})")
    ok4b = strictly_increasing_last_n(d["dif"], p.macd_trend_days)
    ok4c = strictly_increasing_last_n(d["dea"], p.macd_trend_days)
    tail_hist = d["macd_hist"].iloc[-p.macd_trend_days :]
    ok4d = bool(not tail_hist.isna().any() and (tail_hist > 0).all())
    ok4e = strictly_increasing_last_n(d["macd_hist"], p.macd_trend_days)
    ln(f"  DIF 最近{p.macd_trend_days}根严格递增 → {ok4b}")
    ln(f"  DEA 最近{p.macd_trend_days}根严格递增 → {ok4c}")
    ln(
        f"  柱 最近{p.macd_trend_days}根均>0 → {ok4d}  "
        f"值: {', '.join(fmt_num(v, 6) for v in tail_hist)}"
    )
    ln(f"  柱 最近{p.macd_trend_days}根严格递增 → {ok4e}")
    ln(f"  条件(4) 总结果: {bool(ok4a and ok4b and ok4c and ok4d and ok4e)}")
    ln("")

    thr = p.vol_ma5_vs_ma60 * row["vol_ma60"]
    ok5 = bool(row["vol_ma5"] > thr)
    ln("（5）量能：5 日均量 > 60 日均量 × {:.0f}%".format(p.vol_ma5_vs_ma60 * 100))
    ln(
        f"  VOL_MA5 = {fmt_num(row['vol_ma5'], 4)}  VOL_MA60 = {fmt_num(row['vol_ma60'], 4)}"
    )
    ln(f"  阈值 = VOL_MA60×{p.vol_ma5_vs_ma60} = {fmt_num(thr, 4)}  VOL_MA5>阈值 → {ok5}")
    ln(f"  条件(5) 总结果: {ok5}")
    ln("")

    passed, _ = screen_detail_last_row(df, p)
    ln(f"【策略1 总判定】{'通过' if passed else '未通过'}  （分项布尔见 --detail）")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from main import main

    av = sys.argv[1:]
    if not any(x == "--strategies" or x.startswith("--strategies=") for x in av):
        av = ["--strategies", "1", *av]
    raise SystemExit(main(av))
