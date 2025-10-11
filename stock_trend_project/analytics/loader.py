import pandas as pd
from datetime import timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

from analytics.metrics.sma import required_buffer


def load_from_yfinance(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Fetch adjusted OHLCV data from Yahoo Finance for a SINGLE ticker symbol.

    Always returns a DataFrame with standard single-level columns:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. "AAPL".
    period : str
        Yahoo Finance period string, e.g. "1y", "2y", "5y", "max".
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download(
        ticker,
        period=period,
        group_by="column",
        auto_adjust=True,
        progress=False,
    )

    # Handle MultiIndex columns (some tickers return multi-level data)
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df[ticker]
        elif ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)

    df = df.reset_index()  # ensures a 'Date' column exists
    return df


def fetch_data_with_buffer(
    ticker: str,
    start_date,
    end_date,
    sma_windows=None,
) -> tuple[pd.DataFrame, int]:
    """
    Fetch Yahoo Finance data for a specific date range, with automatic SMA buffer padding.

    This function fetches extra historical data BEFORE the selected start date so that
    all Simple Moving Average (SMA) calculations are valid from the first visible day.

    Example:
        sma_windows = [20, 50]  → fetch (max(20, 50) - 1) = 49 extra days before start_date

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    start_date, end_date : datetime.date
        Visible analysis date range.
    sma_windows : list[int] | None
        SMA window sizes the user selected (e.g. [20, 50]).
        If None, no buffer is added.

    Returns
    -------
    (pd.DataFrame, int)
        - The full DataFrame including buffered days.
        - The number of buffer rows (to align display start index later).
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    buffer_days = 0
    if sma_windows:
        buffer_days = required_buffer(max(sma_windows))  # e.g. SMA50 → need 49 extra days

    adjusted_start = start_date - timedelta(days=buffer_days)

    df = yf.download(
        ticker,
        start=adjusted_start,
        end=end_date,
        group_by="column",
        auto_adjust=True,
        progress=False,
    )

    # Normalize MultiIndex just in case
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(0):
            df = df[ticker]
        elif ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)

    df = df.reset_index()
    return df, buffer_days