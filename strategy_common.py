"""
策略 1 / 2 / 3 共用：数据读取、均线与 MACD 计算、通用判定工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

_MIN_TICKER_HITS = 2


def _norm_col(name: object) -> str:
    return str(name).strip()


def _pick_column(columns: list[str], aliases: list[str]) -> Optional[str]:
    stripped = {_norm_col(c): c for c in columns}
    lower_map = {_norm_col(c).lower(): c for c in columns}
    for a in aliases:
        key = a.strip()
        if key in stripped:
            return stripped[key]
        lk = key.lower()
        if lk in lower_map:
            return lower_map[lk]
    return None


def load_ohlcv_from_excel(
    path: Union[str, Path],
    sheet_name: Union[int, str] = 0,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到 Excel 文件: {path.resolve()}")

    raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if raw.empty:
        raise ValueError(f"工作表为空: {path} sheet={sheet_name!r}")

    cols = list(raw.columns)
    col_date = _pick_column(
        cols,
        ["日期", "date", "Date", "交易日期", "时间", "datetime", "trade_date"],
    )
    col_close = _pick_column(
        cols,
        ["收盘", "close", "Close", "收盘价", "收盘价(元)", "收", "C"],
    )
    col_vol = _pick_column(
        cols,
        ["成交量", "volume", "Volume", "成交股数", "vol", "V", "成交量(手)"],
    )

    if col_close is None or col_vol is None:
        raise ValueError(
            "Excel 需包含可识别的「收盘」与「成交量」列。"
            f"当前列: {cols}"
        )

    out = pd.DataFrame(
        {
            "close": pd.to_numeric(raw[col_close], errors="coerce"),
            "volume": pd.to_numeric(raw[col_vol], errors="coerce"),
        }
    )

    if col_date is not None:
        out.index = pd.to_datetime(raw[col_date], errors="coerce")
        out = out[~out.index.isna()]
    else:
        out.index = pd.RangeIndex(len(out))

    out = out.dropna(subset=["close", "volume"])
    out = out.sort_index()
    return out.astype({"close": float, "volume": float})


def _looks_like_ticker(val: object) -> bool:
    s = _norm_col(val)
    if not s or s.lower() in ("nan", "none", "nat"):
        return False
    if "." not in s:
        return False
    u = s.upper()
    if any(u.endswith(suf) for suf in (".SZ", ".SH", ".HK", ".BJ")):
        return True
    return bool(s[0].isdigit())


def _to_float_cell(val: object) -> float:
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (bool, np.bool_)):
        return float(val)
    if isinstance(val, (int, np.integer)):
        return float(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    s = str(val).strip().replace(",", "")
    if not s or s.lower() in ("nan", "--", "-", "none"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_first_col_datetime(val: object) -> pd.Timestamp:
    return pd.to_datetime(val, errors="coerce")


def _wide_scan_sheet(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[str], int, list[str]]:
    """
    解析宽表：返回 (工作表df, 代码行号, 代码列表, 数据起始行, 与代码等长的名称列表)。
    若无单独名称行，名称列表为与代码等长的空串。
    """
    df = raw.copy()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.columns = range(df.shape[1])
    nrows, ncols = int(df.shape[0]), int(df.shape[1])
    if ncols < 2:
        raise ValueError("宽表至少需要 2 列（日期 + 至少 1 只标的）")

    ticker_row: Optional[int] = None
    scan = min(15, nrows)
    for r in range(scan):
        hits = 0
        total = 0
        for c in range(1, ncols):
            v = df.iat[r, c]
            if pd.isna(v):
                continue
            if isinstance(v, str) and not v.strip():
                continue
            total += 1
            if _looks_like_ticker(v):
                hits += 1
        if total >= _MIN_TICKER_HITS and hits >= max(
            _MIN_TICKER_HITS, int(np.ceil(total * 0.45))
        ):
            ticker_row = r
            break

    if ticker_row is None:
        ticker_row = 0

    tickers: list[str] = []
    for c in range(1, ncols):
        v = df.iat[ticker_row, c]
        tickers.append(_norm_col(v) if not pd.isna(v) else f"_col{c}")

    data_start: Optional[int] = None
    for r in range(ticker_row + 1, nrows):
        dt = _parse_first_col_datetime(df.iat[r, 0])
        if pd.isna(dt):
            continue
        num_ok = 0
        for c in range(1, ncols):
            x = _to_float_cell(df.iat[r, c])
            if not np.isnan(x):
                num_ok += 1
        if num_ok >= 1:
            data_start = r
            break

    if data_start is None:
        raise ValueError("宽表中未找到「首列为日期且存在数值」的数据行")

    name_cells: list[str] = [""] * len(tickers)
    if ticker_row + 1 < data_start:
        r = ticker_row + 1
        dt0 = _parse_first_col_datetime(df.iat[r, 0])
        if pd.isna(dt0):
            for j, c in enumerate(range(1, ncols)):
                if j >= len(tickers):
                    break
                v = df.iat[r, c]
                name_cells[j] = _norm_col(v) if not pd.isna(v) else ""

    return df, ticker_row, tickers, data_start, name_cells


def _wide_sheet_to_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df, _ticker_row, tickers, data_start, _name_cells = _wide_scan_sheet(raw)
    nrows, ncols = int(df.shape[0]), int(df.shape[1])

    idx_list: list[pd.Timestamp] = []
    rows_mat: list[list[float]] = []
    L = ncols - 1
    for r in range(data_start, nrows):
        dt = _parse_first_col_datetime(df.iat[r, 0])
        if pd.isna(dt):
            continue
        row_vals: list[float] = []
        any_num = False
        for c in range(1, ncols):
            x = _to_float_cell(df.iat[r, c])
            row_vals.append(x)
            if not np.isnan(x):
                any_num = True
        if not any_num:
            continue
        while len(row_vals) < L:
            row_vals.append(float("nan"))
        row_vals = row_vals[:L]
        idx_list.append(pd.Timestamp(dt))
        rows_mat.append(row_vals)

    if not idx_list:
        raise ValueError("宽表无有效数据行")

    use_L = min(L, len(tickers))
    tickers = tickers[:use_L]
    rows_mat = [rv[:use_L] for rv in rows_mat]

    out = pd.DataFrame(rows_mat, index=pd.DatetimeIndex(idx_list), columns=tickers)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


@dataclass
class WidePanelBundle:
    """双 Sheet 宽表加载结果：多标的 OHLCV + 展示名称。"""

    panels: dict[str, pd.DataFrame]
    names: dict[str, str]


def load_dual_sheet_wide_excel(
    path: Union[str, Path],
    sheet_close: Union[int, str] = 0,
    sheet_volume: Union[int, str] = 1,
) -> WidePanelBundle:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到 Excel 文件: {path.resolve()}")

    close_raw = pd.read_excel(
        path, sheet_name=sheet_close, header=None, engine="openpyxl"
    )
    vol_raw = pd.read_excel(
        path, sheet_name=sheet_volume, header=None, engine="openpyxl"
    )

    c = _wide_sheet_to_frame(close_raw)
    v = _wide_sheet_to_frame(vol_raw)
    _, _, _, _, name_cells = _wide_scan_sheet(close_raw)

    m = min(c.shape[1], v.shape[1])
    c = c.iloc[:, :m].copy()
    v = v.iloc[:, :m].copy()
    v.columns = c.columns

    names_list = (name_cells + [""] * m)[:m]
    names: dict[str, str] = {}
    for i, sym in enumerate(c.columns):
        names[str(sym)] = names_list[i] if i < len(names_list) else ""

    out: dict[str, pd.DataFrame] = {}
    for sym in c.columns:
        sub = pd.DataFrame({"close": c[sym], "volume": v[sym]}).sort_index()
        sub = sub.dropna(subset=["close", "volume"], how="any")
        out[str(sym)] = sub.astype({"close": float, "volume": float})
    return WidePanelBundle(panels=out, names=names)


def load_ohlcv_from_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date")
    raw = raw.sort_index()
    return raw


@dataclass
class IndicatorParams:
    """均线与 MACD 计算参数（策略 1 / 2 / 3 共用）。"""

    ma_short: int = 5
    ma_mid: int = 10
    ma_long: int = 20
    ma60: int = 60
    ma120: int = 120
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = (dif - dea) * 2.0
    return dif, dea, hist


def strictly_increasing_last_n(s: pd.Series, n: int) -> bool:
    """最后 n 根 K 线对应值严格递增。"""
    if len(s) < n or s.iloc[-n:].isna().any():
        return False
    tail = s.iloc[-n:].to_numpy(dtype=float)
    return bool(np.all(np.diff(tail) > 0))


def mas_group_rising_daily(*series: pd.Series, days: int) -> bool:
    """
    多条均线在最近 `days` 个交易日：每条均线均满足「当日值 > 昨日值」。
    """
    need = days + 1
    if not series or any(len(s) < need for s in series):
        return False
    for ser in series:
        seg = ser.iloc[-need:]
        if seg.isna().any():
            return False
        d = seg.diff().iloc[1:]
        if not (d > 0).all():
            return False
    return True


def macd_histogram_positive_rising(hist: pd.Series, n: int) -> bool:
    """MACD 柱：最近 n 根均 > 0 且严格递增。"""
    if len(hist) < n:
        return False
    tail = hist.iloc[-n:]
    if tail.isna().any() or not (tail > 0).all():
        return False
    return strictly_increasing_last_n(hist, n)


def close_lte_ma20_multiple(close: float, ma20: float, multiple: float) -> bool:
    """收盘价是否不超过 MA20 的 multiple 倍（如 1.10 即 110%）。"""
    if pd.isna(close) or pd.isna(ma20):
        return False
    return bool(close <= multiple * ma20)


def macd_golden_cross_above_zero(d: pd.DataFrame) -> bool:
    """
    最近一根 K 线为 0 轴上 MACD 金叉：昨 DIF<DEA，今 DIF>DEA，且今 DIF>0、DEA>0。
    `d` 须含列 dif、dea（由 enrich_indicators 生成）。
    """
    if len(d) < 2:
        return False
    prev = d.iloc[-2]
    row = d.iloc[-1]
    for k in ("dif", "dea"):
        if pd.isna(prev[k]) or pd.isna(row[k]):
            return False
    if not (row["dif"] > 0 and row["dea"] > 0):
        return False
    return bool(prev["dif"] < prev["dea"] and row["dif"] > row["dea"])


def volume_gt_prev_ma5(row: pd.Series, ratio: float = 1.0) -> bool:
    """
    当日成交量 > vol_ma5_prev × ratio。
    vol_ma5_prev 为不含当日的过去 5 日成交量均值；ratio=1 即 txt「大于过去5日均量」。
    """
    if pd.isna(row["vol_ma5_prev"]) or pd.isna(row["volume"]):
        return False
    return bool(float(row["volume"]) > ratio * float(row["vol_ma5_prev"]))


def enrich_indicators(df: pd.DataFrame, p: IndicatorParams) -> pd.DataFrame:
    """
    增加 MA5/10/20/60/120、成交量均线、MACD；
    vol_ma5_prev：不含当日的「过去 5 日」成交量均值（用于放量突破等）。
    """
    out = df.copy()
    c = out["close"].astype(float)
    vol_s = out["volume"].astype(float)

    out["ma5"] = c.rolling(p.ma_short, min_periods=p.ma_short).mean()
    out["ma10"] = c.rolling(p.ma_mid, min_periods=p.ma_mid).mean()
    out["ma20"] = c.rolling(p.ma_long, min_periods=p.ma_long).mean()
    out["ma60"] = c.rolling(p.ma60, min_periods=p.ma60).mean()
    out["ma120"] = c.rolling(p.ma120, min_periods=p.ma120).mean()
    out["vol_ma5"] = vol_s.rolling(p.ma_short, min_periods=p.ma_short).mean()
    out["vol_ma20"] = vol_s.rolling(p.ma_long, min_periods=p.ma_long).mean()
    out["vol_ma60"] = vol_s.rolling(p.ma60, min_periods=p.ma60).mean()
    out["vol_ma5_prev"] = (
        vol_s.shift(1).rolling(p.ma_short, min_periods=p.ma_short).mean()
    )

    dif, dea, hist = compute_macd(c, p.macd_fast, p.macd_slow, p.macd_signal)
    out["dif"] = dif
    out["dea"] = dea
    out["macd_hist"] = hist
    return out


def parse_sheet_arg(s: str) -> Union[int, str]:
    s = s.strip()
    return int(s) if s.isdigit() else s


def fmt_num(x: object, nd: int = 4) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "NaN"
    if pd.isna(x):
        return "NaN"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def fmt_date(ix: object) -> str:
    if isinstance(ix, pd.Timestamp):
        return ix.strftime("%Y-%m-%d")
    return str(ix)
