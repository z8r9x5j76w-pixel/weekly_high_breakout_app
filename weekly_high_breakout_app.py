"""
52-Week High Breakout Scanner — QuantGaps Research
Standalone Streamlit app + self-improver ready
===================================================
Run locally:   streamlit run weekly_high_breakout_app.py
Deploy:        push to GitHub + connect Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="52-Week High Breakout | QuantGaps",
    page_icon="🚀",
    layout="wide",
)

# ── Production params (v1.0) ──────────────────────────────────────────────────
SL              = 0.04
TP              = 0.18
MAX_HOLD        = 30
NOTIONAL        = 2000.0
MAX_POSITIONS   = 10
TREND_SMA       = 50
TRADING_DAYS    = 252
PERIOD          = "5y"
BATCH_SIZE      = 20

# Detection params
CONSOL_MIN      = 10      # min consolidation bars below 52w high
CONSOL_MAX      = 60      # max consolidation bars below 52w high
CONSOL_TOL      = 0.05    # how close to 52w high during consolidation (within 5%)
HIGH_WINDOW     = 252     # bars to define 52-week high
MIN_PULLBACK    = 0.02    # min pullback from 52w high during consolidation
MAX_PULLBACK    = 0.20    # max pullback from 52w high during consolidation
BRK_BUFFER      = 0.0025   # close must be this % above prior 52w high
VOL_RATIO       = 0.0     # breakout volume vs 20d avg (0.0 = off)

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = [
    "SPY","QQQ","DIA","IWM","SMH","XLF","XLK","XLV","XLE","XLI",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ADBE","CRM",
    "NFLX","AMD","QCOM","TXN","MU","ORCL","CSCO","NOW","AMAT","ISRG",
    "JPM","V","MA","GS","BAC","BLK","UNH","LLY","TMO","ABT",
    "HD","COST","WMT","MCD","NKE","PG","KO","PEP","ABBV","MRK",
    "XOM","CVX","CAT","DE","LMT","RTX","NEE","LIN","MMM","GE",
    "CRWD","PANW","PLTR","SNOW","DDOG","ZS","COIN","SQ","SHOP",
    "SBUX","DIS","PYPL","INTC","IBM","F","GM","AAL","DAL",
    "WFC","C","AXP","BRK-B","SCHW","CME","ICE",
    "AMGN","GILD","REGN","VRTX","BSX","MDT","SYK",
    "PLD","AMT","CCI","EQIX","PSA",
]

# ── Data download ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers):
    data = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        try:
            raw = yf.download(batch, period=PERIOD, interval="1d",
                              group_by="ticker", progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            for t in batch:
                try:
                    if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                        if t not in raw.columns.get_level_values(0):
                            continue
                        df = raw[t].copy()
                    else:
                        df = raw.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.dropna()
                    if not {"Open","High","Low","Close"}.issubset(df.columns):
                        continue
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.sort_index()
                    if len(df) < 260:
                        continue
                    cols = ["Open","High","Low","Close"]
                    if "Volume" in df.columns:
                        cols.append("Volume")
                    data[t] = df[cols]
                except Exception:
                    continue
        except Exception:
            continue
    return data


# ── Pattern detection ─────────────────────────────────────────────────────────
def detect_52w_breakout(df, di, p=None):
    """
    52-Week High Breakout with Consolidation.

    Logic:
    1. Stock set a 52-week high at some point in the past
    2. Pulled back but stayed within CONSOL_TOL% of that high (consolidation)
    3. Consolidation lasted CONSOL_MIN to CONSOL_MAX bars
    4. Today's close breaks ABOVE the prior 52-week high + buffer
    5. SMA50 trend filter
    6. Optional: volume confirmation
    """
    if p is None:
        p = dict(
            CONSOL_MIN=CONSOL_MIN, CONSOL_MAX=CONSOL_MAX,
            CONSOL_TOL=CONSOL_TOL, HIGH_WINDOW=HIGH_WINDOW,
            MIN_PULLBACK=MIN_PULLBACK, MAX_PULLBACK=MAX_PULLBACK,
            BRK_BUFFER=BRK_BUFFER, VOL_RATIO=VOL_RATIO,
            TREND_SMA=TREND_SMA,
        )

    close  = df["Close"].values
    high   = df["High"].values
    vols   = df["Volume"].values if "Volume" in df.columns else None

    min_start = p["HIGH_WINDOW"] + p["CONSOL_MAX"] + p["TREND_SMA"] + 5
    if di < min_start or di >= len(close) - 1:
        return None

    ct = close[di]
    cy = close[di - 1]

    # SMA trend filter
    sma = np.mean(close[di - p["TREND_SMA"]:di])
    if ct <= sma:
        return None

    # Try each consolidation length
    for consol_bars in range(p["CONSOL_MIN"], p["CONSOL_MAX"] + 1):
        consol_start = di - consol_bars
        if consol_start < p["HIGH_WINDOW"] + 2:
            continue

        # 52-week high = max high in HIGH_WINDOW bars ending at consol_start
        lookback_start = consol_start - p["HIGH_WINDOW"]
        if lookback_start < 0:
            continue
        prior_52w_high = float(np.max(high[lookback_start:consol_start + 1]))
        if prior_52w_high <= 0:
            continue

        # Consolidation window: bars from consol_start to di-1
        consol_highs  = high[consol_start:di]
        consol_closes = close[consol_start:di]

        if len(consol_closes) == 0:
            continue

        consol_high_max = float(np.max(consol_highs))
        consol_low_min  = float(np.min(consol_closes))

        # Pullback check: consolidation low must be within MIN/MAX pullback of 52w high
        pullback = (prior_52w_high - consol_low_min) / prior_52w_high
        if not (p["MIN_PULLBACK"] <= pullback <= p["MAX_PULLBACK"]):
            continue

        # Consolidation must stay near the 52w high (not drift too far below)
        if consol_high_max < prior_52w_high * (1 - p["CONSOL_TOL"]):
            continue

        # No close above prior 52w high during consolidation
        if np.any(consol_closes > prior_52w_high * (1 + p["BRK_BUFFER"])):
            continue

        # Breakout: previous close below 52w high, current close above
        if cy > prior_52w_high:
            continue
        if ct < prior_52w_high * (1 + p["BRK_BUFFER"]):
            continue

        # Optional volume filter
        if p["VOL_RATIO"] > 0 and vols is not None:
            vol_ma = np.mean(vols[max(0, di-20):di])
            if vol_ma > 0 and vols[di] < vol_ma * p["VOL_RATIO"]:
                continue

        pct_above = (ct - prior_52w_high) / prior_52w_high * 100
        return {
            "prior_52w_high": round(prior_52w_high, 2),
            "consol_bars":    consol_bars,
            "pullback_pct":   round(pullback * 100, 1),
            "pct_above_high": round(pct_above, 2),
            "strength":       round((1 - pullback) * 0.5 + (1 / consol_bars) * 0.5, 4),
        }

    return None


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(data, p=None):
    date_set = set()
    for df in data.values():
        date_set.update(df.index.tolist())
    dates = sorted(date_set)

    signals = {}
    for ticker, df in data.items():
        tsigs = {}
        for di in range(1, len(df)):
            date = df.index[di]
            if date not in date_set:
                continue
            sig = detect_52w_breakout(df, di, p)
            if not sig:
                continue
            bl = sig["prior_52w_high"]
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= bl < ct):
                continue
            tsigs[date] = sig
        if tsigs:
            signals[ticker] = tsigs

    sl = p["SL"] if p else SL
    tp = p["TP"] if p else TP
    mh = p["MAX_HOLD"] if p else MAX_HOLD

    open_pos, pending, trades = {}, {}, []
    daily_pnl = np.zeros(len(dates))

    for di in range(1, len(dates)):
        date = dates[di]
        if date in pending:
            for ticker, strength, sig in sorted(pending.pop(date),
                                                key=lambda x: x[1], reverse=True):
                if len(open_pos) >= MAX_POSITIONS or ticker in open_pos:
                    continue
                df = data.get(ticker)
                if df is None or date not in df.index:
                    continue
                o = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0:
                    continue
                if o < sig["prior_52w_high"]:
                    continue
                open_pos[ticker] = dict(
                    entry_price=o, shares=NOTIONAL/o,
                    sl_price=o*(1-sl), tp_price=o*(1+tp),
                    days_held=0, entry_di=di, entry_date=date,
                )

        day_pnl, to_close = 0.0, []
        for ticker, pos in open_pos.items():
            df = data.get(ticker)
            if df is None or date not in df.index:
                continue
            bar = df.loc[date]
            lo = float(bar["Low"]); hi = float(bar["High"]); cl = float(bar["Close"])
            pos["days_held"] += 1
            reason = ep = None
            if   np.isfinite(lo) and lo <= pos["sl_price"]: reason, ep = "SL", pos["sl_price"]
            elif np.isfinite(hi) and hi >= pos["tp_price"]: reason, ep = "TP", pos["tp_price"]
            elif pos["days_held"] >= mh:                    reason, ep = "MH", cl
            if reason:
                pnl = (ep - pos["entry_price"]) * pos["shares"]
                day_pnl += pnl
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date":   date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price":  round(ep, 2),
                    "pnl":         round(pnl, 2),
                    "reason":      reason,
                    "days_held":   pos["days_held"],
                })
                to_close.append(ticker)

        daily_pnl[di] = day_pnl
        for t in to_close:
            open_pos.pop(t, None)

        if di < len(dates) - 1:
            next_date = dates[di+1]
            for ticker, tsigs in signals.items():
                if ticker in open_pos or date not in tsigs:
                    continue
                sig = tsigs[date]
                pending.setdefault(next_date, []).append(
                    (ticker, sig["strength"], sig))

    return trades, daily_pnl, len(dates)


def calc_metrics(trades, daily_pnl, n_dates):
    if not trades:
        return {}
    pnls    = np.array([t["pnl"] for t in trades])
    reasons = [t["reason"] for t in trades]
    holds   = np.array([t["days_held"] for t in trades], dtype=float)
    n       = len(trades)
    capital = NOTIONAL * MAX_POSITIONS
    n_years = n_dates / TRADING_DAYS
    total   = float(pnls.sum())
    cagr    = ((1 + total/capital)**(1/n_years) - 1)*100 if n_years else 0
    cum     = np.cumsum(daily_pnl)
    std     = daily_pnl.std()
    sharpe  = daily_pnl.mean()/std*np.sqrt(TRADING_DAYS) if std > 0 else 0
    peak    = np.maximum.accumulate(cum)
    max_dd  = float((cum - peak).min())
    calmar  = cagr/abs(max_dd/capital*100) if max_dd else 0
    return dict(
        n=n, wr=round((pnls>0).sum()/n*100,1), total=round(total,2),
        cagr=round(cagr,2), sharpe=round(sharpe,3),
        calmar=round(calmar,3), max_dd=round(max_dd,2),
        avg_hold=round(float(holds.mean()),1),
        pct_sl=round(reasons.count("SL")/n*100,1),
        pct_tp=round(reasons.count("TP")/n*100,1),
        pct_mh=round(reasons.count("MH")/n*100,1),
        cum_pnl=cum,
    )


# ── Live scan ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def find_live_signals(_data):
    cutoff  = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(5)
    results = []
    for ticker, df in _data.items():
        for di in range(max(1, len(df) - 10), len(df)):
            if df.index[di] < cutoff:
                continue
            sig = detect_52w_breakout(df, di)
            if not sig:
                continue
            bl = sig["prior_52w_high"]
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            if not (cy <= bl < ct):
                continue
            results.append({
                "Ticker":         ticker,
                "Date":           df.index[di].strftime("%Y-%m-%d"),
                "Close":          round(ct, 2),
                "52W High":       round(bl, 2),
                "Pullback %":     sig["pullback_pct"],
                "Consol Bars":    sig["consol_bars"],
                "% Above High":   sig["pct_above_high"],
                "SL Price":       round(ct * (1 - SL), 2),
                "TP Price":       round(ct * (1 + TP), 2),
                "Strength":       sig["strength"],
            })
    return sorted(results, key=lambda x: x["Strength"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🚀 52-Week High Breakout Scanner")
st.caption("QuantGaps Research · v1.0 · SL 3% · TP 15% · MaxHold 20 · SMA50")

tab1, tab2 = st.tabs(["🟢 LIVE SIGNALS", "🔵 BACKTEST"])

with tab1:
    st.subheader("Live 52-Week High Breakouts — last 5 trading days")
    st.caption(f"Universe: {len(TICKERS)} tickers · Entry: next-day open > prior 52W high")

    if st.button("▶ Run Live Scan", type="primary", key="live"):
        with st.spinner("Downloading data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner(f"Scanning {len(data)} tickers..."):
            sigs = find_live_signals(data)

        if not sigs:
            st.info("No 52-week high breakouts in the last 5 trading days.")
        else:
            st.success(f"✅ {len(sigs)} signal(s) found")
            st.dataframe(pd.DataFrame(sigs), use_container_width=True, hide_index=True)
            st.caption("⚠️ Entry next trading day at open, only if open > 52W High level")

            top = sigs[0]
            st.divider()
            st.markdown(f"### 🏆 Top Signal: **{top['Ticker']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close",    f"${top['Close']}")
            c2.metric("52W High", f"${top['52W High']}")
            c3.metric("SL", f"${top['SL Price']}", f"-{SL*100:.0f}%", delta_color="inverse")
            c4.metric("TP", f"${top['TP Price']}", f"+{TP*100:.0f}%")

with tab2:
    st.subheader("5-Year Backtest · 52-Week High Breakout v1.0")
    st.caption("Reference only — historical simulation, not forward-looking")

    if st.button("▶ Run Backtest", type="primary", key="bt"):
        with st.spinner("Downloading 5y data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner("Running backtest..."):
            trades, daily_pnl, n_dates = run_backtest(data)
            m = calc_metrics(trades, daily_pnl, n_dates)

        if not m:
            st.warning("No trades generated.")
        else:
            st.divider()
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Trades",   m["n"])
            c2.metric("Win Rate", f"{m['wr']}%")
            c3.metric("CAGR",     f"{m['cagr']}%")
            c4.metric("Sharpe",   m["sharpe"])
            c5.metric("Calmar",   m["calmar"])
            c6.metric("Max DD",   f"${m['max_dd']:,.0f}")

            c1b,c2b,c3b,c4b = st.columns(4)
            c1b.metric("Total P&L", f"${m['total']:,.0f}")
            c2b.metric("Avg Hold",  f"{m['avg_hold']} days")
            c3b.metric("SL exits",  f"{m['pct_sl']}%")
            c4b.metric("TP exits",  f"{m['pct_tp']}%")

            st.divider()
            st.markdown("#### Equity Curve")
            st.line_chart(pd.DataFrame({"Cumulative P&L ($)": m["cum_pnl"]}),
                          use_container_width=True)

            st.divider()
            st.markdown("#### Trade Log")
            df_t = pd.DataFrame(trades).sort_values("exit_date", ascending=False)
            st.dataframe(df_t.astype(str), use_container_width=True, hide_index=True)
            csv = df_t.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv,
                               "52w_high_breakout_trades.csv", "text/csv")
