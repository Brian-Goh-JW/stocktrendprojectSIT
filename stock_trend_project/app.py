from flask import Flask, render_template, request, flash
import pandas as pd
from math import sqrt

# Project modules
from analytics.metrics import compute_sma, compute_runs, daily_returns, extract_trades
from analytics.plotting import plot_price_vs_sma, plot_runs_overlay
from analytics.utils import normalize_columns, parse_date_column, ensure_ohlcv
from analytics.loader import load_from_yfinance

# yfinance direct (for start/end date fetching)
try:
    import yfinance as yf
except Exception:
    yf = None

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"


def _fetch_data(ticker: str, start_date: str | None, end_date: str | None, fallback_period: str = "3y") -> pd.DataFrame:
    """If start/end provided, fetch with date range, else fall back to period (default 3y)."""
    if (start_date or end_date) and yf is not None:
        df = yf.download(ticker, start=start_date or None, end=end_date or None, progress=False)
        df = df.reset_index()  # Date becomes a column
        return df
    # fallback to our loader by period
    return load_from_yfinance(ticker, period=fallback_period)


def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "—"


def _fmt_pct(x, digits=2):
    if x is None:
        return "—"
    return f"{x:.{digits}f}%"


def _safe_div(a, b):
    try:
        return a / b if b not in (0, None) else None
    except Exception:
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    chart1_html = None
    chart2_html = None
    summary = {}
    df_preview = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        # Date pickers
        start_date = request.form.get("start_date", "").strip() or None
        end_date = request.form.get("end_date", "").strip() or None
        fallback_period = "3y"  # if start/end left blank

        # SMA inputs
        sma1 = int(request.form.get("sma_window1", 20))
        sma2_enabled = request.form.get("sma2_enabled") is not None
        raw_sma2 = request.form.get("sma_window2", "").strip()
        sma2 = int(raw_sma2) if (sma2_enabled and raw_sma2.isdigit()) else None

        # Runs
        min_run_len = int(request.form.get("min_run_len", 2))

        # Chart currency (label only)
        currency = "USD"

        try:
            if not ticker:
                flash("Please enter a ticker (e.g., AAPL).", "warning")
            else:
                # ---- Load & normalize ----
                df = _fetch_data(ticker, start_date, end_date, fallback_period=fallback_period)
                flash(
                    f"Loaded {ticker} data via Yahoo Finance"
                    + (f" for {start_date or 'beginning'} → {end_date or 'today'}." if (start_date or end_date) else f" ({fallback_period})."),
                    "success"
                )

                df = normalize_columns(df, ticker=ticker)
                df = ensure_ohlcv(df)
                df = parse_date_column(df)
                df = df.sort_values("Date").reset_index(drop=True)

                # ---- Metrics ----
                df[f"SMA_{sma1}d"] = compute_sma(df["Close"].values, sma1)
                if sma2:
                    df[f"SMA_{sma2}d"] = compute_sma(df["Close"].values, sma2)
                df["DailyReturn"] = daily_returns(df["Close"].values)

                run_stats, run_df = compute_runs(df["Close"].values)
                total_profit, trades = extract_trades(df["Close"].tolist())

                # ---- Friendly summary numbers ----
                start_dt = df["Date"].min()
                end_dt = df["Date"].max()
                period_days = (end_dt - start_dt).days + 1
                years = period_days / 365.25 if period_days else 0

                start_price = float(df["Close"].iloc[0])
                last_price = float(df["Close"].iloc[-1])

                cum_ret = _safe_div(last_price, start_price)
                cum_ret_pct = (cum_ret - 1) * 100 if cum_ret else None
                cagr_pct = ((cum_ret ** (1 / years) - 1) * 100) if (cum_ret and years > 0) else None

                up_days_pct = float((df["DailyReturn"] > 0).mean() * 100) if len(df) else None
                best_row = df.loc[df["DailyReturn"].idxmax()] if len(df) else None
                worst_row = df.loc[df["DailyReturn"].idxmin()] if len(df) else None
                best_day_pct = float(best_row["DailyReturn"] * 100) if best_row is not None else None
                best_day_date = pd.to_datetime(best_row["Date"]).strftime("%Y-%m-%d") if best_row is not None else None
                worst_day_pct = float(worst_row["DailyReturn"] * 100) if worst_row is not None else None
                worst_day_date = pd.to_datetime(worst_row["Date"]).strftime("%Y-%m-%d") if worst_row is not None else None

                ann_vol_pct = float(df["DailyReturn"].std() * sqrt(252) * 100) if len(df) > 5 else None

                # 52-week high/low (last ~252 trading days)
                window = min(252, len(df))
                sub = df.tail(window) if window else df
                hi_52w = float(sub["High"].max()) if window else None
                lo_52w = float(sub["Low"].min()) if window else None
                off_high_pct = ((last_price / hi_52w - 1) * 100) if (hi_52w and hi_52w > 0) else None

                avg_vol = int(df["Volume"].mean()) if "Volume" in df.columns else None

                # ---- Summary for template ----
                summary = {
                    "ticker": ticker,
                    "rows": int(len(df)),
                    "start": start_dt.strftime("%Y-%m-%d"),
                    "end": end_dt.strftime("%Y-%m-%d"),
                    "duration_days": period_days,
                    "duration_years": round(years, 2),

                    "sma1": f"{sma1}d",
                    "sma2": f"{sma2}d" if sma2 else "",

                    "last_price": round(last_price, 2),
                    "start_price": round(start_price, 2),
                    "cum_return_pct": round(cum_ret_pct, 2) if cum_ret_pct is not None else None,
                    "cagr_pct": round(cagr_pct, 2) if cagr_pct is not None else None,

                    "up_days_pct": round(up_days_pct, 1) if up_days_pct is not None else None,
                    "best_day_pct": round(best_day_pct, 2) if best_day_pct is not None else None,
                    "best_day_date": best_day_date,
                    "worst_day_pct": round(worst_day_pct, 2) if worst_day_pct is not None else None,
                    "worst_day_date": worst_day_date,
                    "ann_vol_pct": round(ann_vol_pct, 1) if ann_vol_pct is not None else None,

                    "hi_52w": round(hi_52w, 2) if hi_52w is not None else None,
                    "lo_52w": round(lo_52w, 2) if lo_52w is not None else None,
                    "off_high_pct": round(off_high_pct, 2) if off_high_pct is not None else None,

                    "avg_volume": _fmt_int(avg_vol),

                    "runs_total_up": run_stats["total_up_runs"],
                    "runs_total_down": run_stats["total_down_runs"],
                    "runs_days_up": run_stats["days_up"],
                    "runs_days_down": run_stats["days_down"],
                    "runs_longest_up": run_stats["longest_up_run"],
                    "runs_longest_down": run_stats["longest_down_run"],

                    "max_profit": round(total_profit, 2),
                    "num_trades": len(trades),

                    "currency": currency,
                }

                # ---- Titles ----
                sma_list = [f"{sma1}d"] + ([f"{sma2}d"] if sma2 else [])
                sma_suffix = f" ({', '.join(sma_list)})" if sma_list else ""
                line_title = f"{ticker} — Price & SMA{sma_suffix}"
                candle_title = f"{ticker} — Candlesticks"

                # ---- Charts ----
                sma_cols = [f"SMA_{sma1}d"] + ([f"SMA_{sma2}d"] if sma2 else [])
                chart1_html = plot_price_vs_sma(df, sma_cols=sma_cols, title=line_title, currency=currency)
                chart2_html = plot_runs_overlay(df, run_df, title=candle_title, min_run_len=min_run_len, currency=currency)

                # ---- Preview: most recent first ----
                preview_df = (
                    df.sort_values("Date", ascending=False)
                      .head(100)
                      .copy()
                )
                preview_df["Date"] = pd.to_datetime(preview_df["Date"]).dt.strftime("%Y-%m-%d")
                df_preview = preview_df.to_html(
                    classes="table table-sm table-striped table-hover table-dark mb-0",
                    index=False,
                    border=0,
                )

        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template(
        "index.html",
        chart1_html=chart1_html,
        chart2_html=chart2_html,
        df_preview=df_preview,
        summary=summary,
    )


if __name__ == "__main__":
    app.run(debug=True)