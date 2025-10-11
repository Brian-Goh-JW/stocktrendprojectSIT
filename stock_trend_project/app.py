from __future__ import annotations
from markupsafe import Markup
from flask import (
    Flask, render_template, request, flash, session,
    redirect, url_for
)
import pandas as pd
from pandas.tseries.offsets import BDay
from math import sqrt

# ---- Project modules (your existing package) ----
from analytics.metrics import compute_sma, compute_runs, daily_returns, extract_trades
from analytics.plotting import plot_price_vs_sma, plot_runs_overlay
from analytics.utils import normalize_columns, parse_date_column, ensure_ohlcv
from analytics.loader import load_from_yfinance

# Optional: yfinance direct (for explicit start/end fetches)
try:
    import yfinance as yf
except Exception:
    yf = None


app = Flask(__name__)
# IMPORTANT: replace in production
app.secret_key = "replace-this-with-a-random-secret"


# =================================================
#                 FAVOURITES (session)
# =================================================
FAV_KEY = "favorites_v1"
LAST_KEY = "last_preset_v1"


def _init_favs() -> None:
    """Ensure the favourites list exists in session."""
    if FAV_KEY not in session:
        session[FAV_KEY] = []


def _clean_form(form: dict) -> dict:
    """
    Standardize inputs so storage/recall is consistent.
    - ticker uppercased/stripped
    - empty dates -> ""
    - sma2_enabled -> "1" or "0"
    - numbers parsed to int where relevant
    """
    ticker = (form.get("ticker") or "").strip().upper()
    start_date = (form.get("start_date") or "").strip()
    end_date = (form.get("end_date") or "").strip()

    def to_int(val, default):
        try:
            return int(str(val).strip())
        except Exception:
            return int(default)

    sma1 = to_int(form.get("sma_window1", 20), 20)
    sma2_enabled = "1" if (form.get("sma2_enabled") is not None) else "0"
    sma2_raw = (form.get("sma_window2") or "").strip()
    sma2 = to_int(sma2_raw, 0) if sma2_enabled == "1" else 0

    min_run_len = to_int(form.get("min_run_len", 2), 2)

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "sma_window1": sma1,
        "sma2_enabled": sma2_enabled,  # "1" or "0"
        "sma_window2": sma2,            # 0 if disabled
        "min_run_len": min_run_len,
    }


def _present_key(p: dict) -> str:
    """
    Unique key for a preset, concatenating a fixed set of fields.
    Example: "AAPL | 2020-01-01 | 2023-01-01 | 20 | 0 | | 2"
    """
    parts = [
        p.get("ticker", ""),
        p.get("start_date", ""),
        p.get("end_date", ""),
        str(p.get("sma_window1", "")),
        str(p.get("sma2_enabled", "0")),
        ("" if str(p.get("sma2_enabled", "0")) == "0" else str(p.get("sma_window2", ""))),
        str(p.get("min_run_len", "")),
    ]
    return " | ".join(parts)


def _format_label(p: dict) -> str:
    """Readable label for the navbar dropdown."""
    t = p.get("ticker", "—")
    sd = p.get("start_date") or "begin"
    ed = p.get("end_date") or "today"
    s1 = p.get("sma_window1", 20)
    s2e = p.get("sma2_enabled", "0") == "1"
    s2 = p.get("sma_window2", 0)
    mrl = p.get("min_run_len", 2)
    sma_part = f"SMA {s1}/{s2}" if s2e and s2 else f"SMA {s1}"
    return f"{t} · {sd}→{ed} · {sma_part} · run≥{mrl}"


def _remember_last_preset(form: dict) -> None:
    """Save last submitted (cleaned) preset into the session."""
    session[LAST_KEY] = _clean_form(form)


@app.post("/favorites/add")
def favorites_add():
    """Add the last preset into favourites (no duplicates)."""
    _init_favs()
    last = session.get(LAST_KEY)
    if not last or not last.get("ticker"):
        flash("Nothing to favourite yet. Run an analysis first.", "warning")
        return redirect(url_for("index"))

    favs = session.get(FAV_KEY, [])
    new_key = _present_key(last)
    for f in favs:
        if f.get("key") == new_key:
            flash("That preset is already in Favourites.", "warning")
            return redirect(url_for("index"))

    favs.append({
        "key": new_key,
        "label": _format_label(last),
        "params": last
    })
    session[FAV_KEY] = favs
    flash("Saved to Favourites.", "success")
    return redirect(url_for("index"))


@app.post("/favorites/remove")
def favorites_remove():
    """Remove preset by index."""
    _init_favs()
    idx_raw = request.form.get("idx", "")
    try:
        idx = int(idx_raw)
    except Exception:
        flash("Invalid selection.", "danger")
        return redirect(url_for("index"))

    favs = session.get(FAV_KEY, [])
    if 0 <= idx < len(favs):
        favs.pop(idx)
        session[FAV_KEY] = favs
        flash("Removed from Favourites.", "success")
    else:
        flash("Invalid selection.", "danger")
    return redirect(url_for("index"))


@app.post("/favorites/use")
def favorites_use():
    """
    Recall a favourite and auto-submit to '/' with stored params.
    We render a tiny page with a hidden form that posts back to index.
    """
    _init_favs()
    idx_raw = request.form.get("idx", "")
    try:
        idx = int(idx_raw)
    except Exception:
        flash("Invalid selection.", "danger")
        return redirect(url_for("index"))

    favs = session.get(FAV_KEY, [])
    if not (0 <= idx < len(favs)):
        flash("Invalid selection.", "danger")
        return redirect(url_for("index"))

    p = favs[idx]["params"]
    html = f"""
    <html><body>
      <form id="f" method="POST" action="{url_for('index')}">
        <input type="hidden" name="ticker" value="{p.get('ticker','')}">
        <input type="hidden" name="start_date" value="{p.get('start_date','')}">
        <input type="hidden" name="end_date" value="{p.get('end_date','')}">
        <input type="hidden" name="sma_window1" value="{p.get('sma_window1',20)}">
        {"<input type='hidden' name='sma2_enabled' value='on'>" if str(p.get('sma2_enabled','0'))=='1' else ""}
        <input type="hidden" name="sma_window2" value="{p.get('sma_window2',0)}">
        <input type="hidden" name="min_run_len" value="{p.get('min_run_len',2)}">
      </form>
      <script>document.getElementById('f').submit()</script>
    </body></html>
    """
    return html


# =================================================
#                    INDEX
# =================================================
def _fetch_data_with_buffer(
    ticker: str,
    start_date: str | None,
    end_date: str | None,
    buffer_days: int = 0,
    fallback_period: str = "3y",
) -> pd.DataFrame:
    """
    If start/end provided, fetch with an extra buffer of business days before start.
    Returns the WHOLE df (including the buffer).
    """
    if (start_date or end_date) and yf is not None:
        sd = pd.to_datetime(start_date) if start_date else None
        ed = pd.to_datetime(end_date) if end_date else None

        if sd is not None and buffer_days > 0:
            # subtract business days so the first visible day has a fully-formed SMA
            sd = (sd - BDay(buffer_days)).date()

        df = yf.download(ticker, start=sd, end=ed, progress=False)
        return df.reset_index()

    # Fallback to our period-based loader (no explicit dates provided)
    return load_from_yfinance(ticker, period=fallback_period)


def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return "—"


def _safe_div(a, b):
    try:
        return a / b if b not in (0, None) else None
    except Exception:
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    _init_favs()  # ensure favourites list exists

    chart1_html = None
    chart2_html = None
    summary: dict = {}
    df_preview = None

    # For navbar dropdown
    favourites = session.get(FAV_KEY, [])

    if request.method == "POST":
        # Remember preset first (so /favorites/add has info)
        _remember_last_preset(request.form)

        ticker = (request.form.get("ticker") or "").strip().upper()

        # Date pickers
        start_date = request.form.get("start_date", "").strip() or None
        end_date = request.form.get("end_date", "").strip() or None
        fallback_period = "3y"  # used when dates are blank

        # SMA inputs
        try:
            sma1 = int(request.form.get("sma_window1", 20))
        except Exception:
            sma1 = 20

        sma2_enabled = request.form.get("sma2_enabled") is not None
        raw_sma2 = (request.form.get("sma_window2") or "").strip()
        sma2 = int(raw_sma2) if (sma2_enabled and raw_sma2.isdigit()) else None

        # Runs
        try:
            min_run_len = int(request.form.get("min_run_len", 2))
        except Exception:
            min_run_len = 2

        currency = "USD"

        try:
            if not ticker:
                flash("Please enter a ticker (e.g., AAPL).", "warning")
            else:
                # --- Fetch with buffer so SMA is correct on the first visible day ---
                buffer_days = max(sma1, sma2 or 0)
                df = _fetch_data_with_buffer(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    buffer_days=buffer_days,
                    fallback_period=fallback_period,
                )

                flash(
                    f"Loaded {ticker} data via Yahoo Finance"
                    + (
                        f" for {start_date or 'beginning'} → {end_date or 'today'} (with {buffer_days} business-day buffer)."
                        if (start_date or end_date)
                        else f" ({fallback_period})."
                    ),
                    "success",
                )

                # --- Normalize / prepare ---
                df = normalize_columns(df, ticker=ticker)
                df = ensure_ohlcv(df)
                df = parse_date_column(df)
                df = df.sort_values("Date").reset_index(drop=True)

                # --- Indicators on buffered data ---
                df[f"SMA_{sma1}d"] = compute_sma(df["Close"].values, sma1)
                if sma2:
                    df[f"SMA_{sma2}d"] = compute_sma(df["Close"].values, sma2)

                # Daily returns in percent for display
                df["DailyReturn"] = daily_returns(df["Close"].values, as_percent=True)
                df["DailyReturn"] = df["DailyReturn"].round(2)
                df["DailyReturn_str"] = df["DailyReturn"].astype(str) + "%"

                # --- Trim to user range (if provided) ---
                if start_date or end_date:
                    sd = pd.to_datetime(start_date) if start_date else df["Date"].min()
                    ed = pd.to_datetime(end_date) if end_date else df["Date"].max()
                    df = df[(df["Date"] >= sd) & (df["Date"] <= ed)].reset_index(drop=True)

                if df.empty:
                    raise ValueError("No data in the chosen date range. Try widening the range or another ticker.")

                # --- Additional metrics for summary ---
                run_stats, run_df = compute_runs(df["Close"].values)
                total_profit, trades = extract_trades(df["Close"].tolist())

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

                # --- Summary dict for template ---
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
                    "currency": "USD",
                }

                # --- Titles & charts ---
                sma_cols = [f"SMA_{sma1}d"] + ([f"SMA_{sma2}d"] if sma2 else [])
                line_title = f"{ticker} — Price & SMA ({', '.join([c.replace('SMA_','').replace('d','') for c in sma_cols])})"
                candle_title = f"{ticker} — Candlesticks"

                chart1_html = plot_price_vs_sma(df, sma_cols=sma_cols, title=line_title, currency=summary["currency"])
                run_stats, run_df = compute_runs(df["Close"].values)  # recompute to keep run_df in scope
                chart2_html = plot_runs_overlay(df, run_df, title=candle_title, min_run_len=min_run_len, currency=summary["currency"])

                # --- Preview (most recent first) ---
                preview_df = df.sort_values("Date", ascending=False).head(100).copy()
                preview_df["Date"] = pd.to_datetime(preview_df["Date"]).dt.strftime("%Y-%m-%d")
                preview_df["DailyReturn"] = preview_df["DailyReturn_str"]
                preview_df = preview_df.drop(columns=["DailyReturn_str"])
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
        df_preview=Markup(df_preview) if df_preview else None,
        summary=summary,
        favourites=favourites,
    )


# =================================================
#                 ENTRY POINT
# =================================================
if __name__ == "__main__":
    # Running via `python app.py`
    app.run(debug=True)