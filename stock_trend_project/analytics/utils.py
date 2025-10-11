import pandas as pd
import numpy as np
import re
from typing import Union, Optional

ALLOWED_EXTENSIONS = {"csv"}

# allow csv, etc
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _flatten_name(c: Union[str, tuple]) -> str:
    if isinstance(c, tuple):
        parts = [str(x) for x in c if x is not None and str(x).strip() != ""]
        return "_".join(parts) if parts else "col"
    return str(c)

# removes matching ticker
def _strip_ticker(name: str, ticker: Optional[str]) -> str:
    """Remove suffix/prefix like '_AAPL' or 'AAPL_' if it matches the ticker."""
    if not ticker:
        return name
    pat = rf"^({re.escape(ticker)}_)?(.+?)(_({re.escape(ticker)}))?$"
    m = re.match(pat, name, flags=re.IGNORECASE)
    if m:
        # Take the middle group (the real field name)
        return m.group(2)
    return name

# standadizes all column names to DOHLCV
def normalize_columns(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Standardize column names so we always end up with:
    Date, Open, High, Low, Close, Volume
    """
    # flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(_flatten_name)
    else:
        df.columns = [_flatten_name(c) for c in df.columns]

    # strip ticker prefix/suffix if present (e.g., 'Open_AAPL', 'AAPL_Close')
    df.columns = [_strip_ticker(c, ticker) for c in df.columns]

    # normalize common names (variants)
    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("date",):
            rename_map[c] = "Date"
        elif lc in ("open", "open price", "adj open"):
            rename_map[c] = "Open"
        elif lc in ("high", "high price"):
            rename_map[c] = "High"
        elif lc in ("low", "low price"):
            rename_map[c] = "Low"
        elif lc in ("close", "close price"):
            rename_map[c] = "Close"
        elif lc in ("adj close", "adj_close", "adjclose"):
            # keep it, might copy to Close in ensure_ohlcv
            rename_map[c] = "Adj Close"
        elif lc in ("volume", "vol"):
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)
    return df

# ensure that data exists and not missing
def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Date, Open, High, Low, Close, Volume exist. Use Adj Close if Close missing."""
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    missing = [c for c in ["Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns after normalization: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df

# ensure date dont overlap and ensure proper datetime column exists
def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        else:
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors="raise")
                    df = df.rename(columns={c: "Date"})
                    df["Date"] = parsed
                    return df
                except Exception:
                    continue
            raise ValueError("No date column found.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"])