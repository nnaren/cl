"""
统一入口：选择策略 1 / 2 / 3 或组合（多策略同屏输出，可选 AND/OR 汇总）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from strategy_common import (
    WidePanelBundle,
    load_dual_sheet_wide_excel,
    load_ohlcv_from_csv,
    load_ohlcv_from_excel,
    parse_sheet_arg,
)
from strategy1 import (
    Strategy1Params,
    screen_detail_last_row,
    screen_last_row,
    strategy1_calculation_report,
)
from strategy2 import (
    Strategy2Params,
    screen_strategy2_detail,
    screen_strategy2_last_row,
    strategy2_calculation_report,
)
from strategy3 import (
    Strategy3Params,
    screen_strategy3_detail,
    screen_strategy3_last_row,
    strategy3_calculation_report,
)

_DEFAULT_EXCEL = Path(__file__).resolve().parent / "data01.xlsx"

_P1 = Strategy1Params()
_P2 = Strategy2Params()
_P3 = Strategy3Params()

_SCREEN = {
    1: lambda df: screen_last_row(df, _P1),
    2: lambda df: screen_strategy2_last_row(df, _P2),
    3: lambda df: screen_strategy3_last_row(df, _P3),
}
_DETAIL = {
    1: lambda df: screen_detail_last_row(df, _P1),
    2: lambda df: screen_strategy2_detail(df, _P2),
    3: lambda df: screen_strategy3_detail(df, _P3),
}
_EXPLAIN = {
    1: lambda df, title: strategy1_calculation_report(df, _P1, title=title),
    2: lambda df, title: strategy2_calculation_report(df, _P2, title=title),
    3: lambda df, title: strategy3_calculation_report(df, _P3, title=title),
}
_STRATEGY_LABEL = {1: "s1", 2: "s2", 3: "s3"}


def parse_strategies_arg(s: str) -> list[int]:
    s = s.strip().lower()
    if s in ("all", "*", "1,2,3"):
        return [1, 2, 3]
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"非法策略编号: {p!r}，应为 1、2、3 或逗号分隔组合")
        n = int(p)
        if n not in (1, 2, 3):
            raise ValueError(f"策略编号须为 1–3: {n}")
        out.append(n)
    if not out:
        raise ValueError("至少选择一种策略")
    # 去重保序
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


CombineMode = Literal["none", "and", "any"]


def _load_data(
    args: argparse.Namespace,
) -> tuple[str, Union[pd.DataFrame, WidePanelBundle]]:
    if args.csv:
        return "single", load_ohlcv_from_csv(args.csv)
    if args.format == "long":
        return "single", load_ohlcv_from_excel(
            args.excel, sheet_name=parse_sheet_arg(args.sheet)
        )
    bundle = load_dual_sheet_wide_excel(
        args.excel,
        sheet_close=parse_sheet_arg(args.sheet_close),
        sheet_volume=parse_sheet_arg(args.sheet_vol),
    )
    return "panels", bundle


def _export_wide_history_csv(
    bundle: WidePanelBundle,
    symbols: list[str],
    strategy_ids: list[int],
    out_path: Path,
    *,
    as_of_end: pd.Timestamp,
    full_history: bool,
) -> None:
    """
    第一行：空 + 各列代码；第二行：空 + 股票名称；
    之后每行：YYYY-MM-DD-策略n，各列在该日及以前数据上是否通过（True/False）。
    full_history=False 时只写 as_of_end 这一天的多策略行；为 True 时写所有交易日直至 as_of_end。
    """
    union = sorted({d for sym in symbols for d in bundle.panels[sym].index})
    as_of_end = pd.Timestamp(as_of_end)
    if full_history:
        dates_iter = [x for x in union if pd.Timestamp(x) <= as_of_end]
    else:
        dates_iter = [as_of_end]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow([""] + symbols)
        w.writerow([""] + [bundle.names.get(sym, "") for sym in symbols])
        for d in dates_iter:
            d_ts = pd.Timestamp(d)
            ds = d_ts.strftime("%Y-%m-%d")
            for sid in strategy_ids:
                row: list[str] = [f"{ds}-策略{sid}"]
                scr: Callable[[pd.DataFrame], bool] = _SCREEN[sid]
                for sym in symbols:
                    bars = bundle.panels[sym].loc[:d_ts]
                    row.append(str(scr(bars)))
                w.writerow(row)


def _run_explain_detail_single(
    data: pd.DataFrame,
    title: str,
    strategy_ids: list[int],
    *,
    detail: bool,
    explain: bool,
) -> None:
    if explain:
        for sid in strategy_ids:
            print(_EXPLAIN[sid](data, title=f"{title}  [{_STRATEGY_LABEL[sid]}]"))
            print()
    if detail:
        for sid in strategy_ids:
            ok, chk = _DETAIL[sid](data)
            print(f"{title}\t{_STRATEGY_LABEL[sid]}\t通过={ok}")
            for k, v in chk.items():
                print(f"  {k}: {v}")
            print()


def _print_wide_row(
    sym: str,
    results: dict[int, bool],
    strategy_ids: list[int],
    combine_mode: CombineMode,
) -> None:
    cols = [sym] + [str(results[s]) for s in strategy_ids]
    if combine_mode == "and":
        cols.append(str(all(results[s] for s in strategy_ids)))
    elif combine_mode == "any":
        cols.append(str(any(results[s] for s in strategy_ids)))
    print("\t".join(cols))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="选股策略统一入口：单策略或 1,2,3 组合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --strategies 1
  python main.py --strategies 2,3 --symbol 600519.SH --detail
  python main.py --strategies all --combine-mode and
  python main.py --strategies 1,3 --format long --explain
  python main.py --strategies 1 --export-csv result.csv
  python main.py --strategies 1 --as-of-date 2025-07-01 --export-csv result.csv
  python main.py --strategies 1 --export-csv hist.csv --export-full-history
        """.strip(),
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help='执行策略：1、2、3、逗号组合（如 1,3）、或 all（默认）',
    )
    parser.add_argument(
        "--combine-mode",
        choices=("none", "and", "any"),
        default="none",
        help="多策略时额外汇总列：none 仅各策略结果；and 全部通过；any 任一通过",
    )
    parser.add_argument("--excel", type=str, default=str(_DEFAULT_EXCEL))
    parser.add_argument("--format", choices=("wide", "long"), default="wide")
    parser.add_argument("--sheet-close", type=str, default="0")
    parser.add_argument("--sheet-vol", type=str, default="1")
    parser.add_argument("--sheet", type=str, default="0")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--symbol", type=str, default="")
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        metavar="PATH",
        help="仅 wide：写出 CSV（首行代码、次行名称、行头 日期-策略n）；默认只写截至日的快照",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default="",
        metavar="YYYY-MM-DD",
        help="分析/导出所截至的日期（含该日）；默认取所选标的中最晚一根K线的日期",
    )
    parser.add_argument(
        "--export-full-history",
        action="store_true",
        help="与 --export-csv 合用：写出从最早到 --as-of-date（或默认最晚日）的每个交易日；默认仅最后一天",
    )
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)

    try:
        strategy_ids = parse_strategies_arg(args.strategies)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    mode, payload = _load_data(args)
    combine_mode: CombineMode = args.combine_mode  # type: ignore[assignment]
    export_csv = (args.export_csv or "").strip()
    as_of_raw = (args.as_of_date or "").strip()
    export_full_history = bool(args.export_full_history)

    if export_csv and mode != "panels":
        print("--export-csv 仅支持默认 wide 双 Sheet 宽表（不要用 --csv / --format long）", file=sys.stderr)
        return 1

    if mode == "single":
        data = payload  # type: ignore[assignment]
        assert isinstance(data, pd.DataFrame)
        if as_of_raw:
            try:
                cut = pd.Timestamp(pd.to_datetime(as_of_raw))
            except (ValueError, TypeError):
                print(f"无效的 --as-of-date: {as_of_raw!r}", file=sys.stderr)
                return 1
            if pd.isna(cut):
                print(f"无效的 --as-of-date: {as_of_raw!r}", file=sys.stderr)
                return 1
            data = data.sort_index().loc[:cut]
        if args.explain or args.detail:
            _run_explain_detail_single(
                data,
                title="【单序列】",
                strategy_ids=strategy_ids,
                detail=args.detail,
                explain=args.explain,
            )
        if not args.detail and not args.explain:
            if len(strategy_ids) == 1:
                sid = strategy_ids[0]
                print(_SCREEN[sid](data))
            else:
                header = ["symbol"] + [_STRATEGY_LABEL[s] for s in strategy_ids]
                if combine_mode == "and":
                    header.append("all_pass")
                elif combine_mode == "any":
                    header.append("any_pass")
                print("\t".join(header))
                vals = ["series"] + [str(_SCREEN[sid](data)) for sid in strategy_ids]
                if combine_mode == "and":
                    vals.append(str(all(_SCREEN[sid](data) for sid in strategy_ids)))
                elif combine_mode == "any":
                    vals.append(str(any(_SCREEN[sid](data) for sid in strategy_ids)))
                print("\t".join(vals))
        return 0

    # wide panels
    bundle: WidePanelBundle = payload  # type: ignore[assignment]
    panels = bundle.panels
    sym_filter = args.symbol.strip()
    # 保持与 Excel 列顺序一致（不按代码字典序重排）
    items = [(k, panels[k]) for k in panels.keys()]
    if sym_filter:
        items = [(k, v) for k, v in items if k == sym_filter]
        if not items:
            print(
                f"未找到代码: {sym_filter!r}，可选: {', '.join(sorted(panels))}",
                file=sys.stderr,
            )
            return 1

    try:
        if as_of_raw:
            as_of_ts = pd.Timestamp(pd.to_datetime(as_of_raw))
            if pd.isna(as_of_ts):
                raise ValueError("nan")
        else:
            as_of_ts = max(bars.index.max() for _, bars in items)
    except (ValueError, TypeError):
        print(f"无效的 --as-of-date: {as_of_raw!r}", file=sys.stderr)
        return 1

    items = [(sym, bars.loc[:as_of_ts]) for sym, bars in items]

    if export_csv:
        symbols = [sym for sym, _ in items]
        _export_wide_history_csv(
            WidePanelBundle(
                panels={s: panels[s] for s in symbols},
                names={s: bundle.names.get(s, "") for s in symbols},
            ),
            symbols,
            strategy_ids,
            Path(export_csv),
            as_of_end=as_of_ts,
            full_history=export_full_history,
        )
        print(f"已写入 {Path(export_csv).resolve()}")
        if not args.explain and not args.detail:
            return 0

    if args.explain or args.detail:
        for sym, bars in items:
            _run_explain_detail_single(
                bars,
                title=f"======== {sym} ========",
                strategy_ids=strategy_ids,
                detail=args.detail,
                explain=args.explain,
            )
        return 0

    # 简洁输出：表头 + 每行
    header = ["symbol"] + [_STRATEGY_LABEL[s] for s in strategy_ids]
    if combine_mode == "and":
        header.append("all_pass")
    elif combine_mode == "any":
        header.append("any_pass")
    print("\t".join(header))

    for sym, bars in items:
        results = {sid: _SCREEN[sid](bars) for sid in strategy_ids}
        _print_wide_row(sym, results, strategy_ids, combine_mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
