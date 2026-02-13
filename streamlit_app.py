#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
try:
    import yfinance as yf
except ImportError:
    yf = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
HISTORY_PATH = DATA_DIR / "run_history.jsonl"

SENT_POS = {
    "beats",
    "beat",
    "surge",
    "up",
    "upgrade",
    "bullish",
    "strong",
    "growth",
    "record",
    "profit",
    "gain",
    "rally",
}
SENT_NEG = {
    "miss",
    "misses",
    "down",
    "downgrade",
    "bearish",
    "weak",
    "decline",
    "loss",
    "drop",
    "fall",
    "lawsuit",
    "probe",
}
FILING_HINTS = {"8-k", "10-q", "10-k", "sec filing", "form 4", "proxy statement"}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_sp500_tickers() -> list[str]:
    errors: list[str] = []

    # Prefer a simple CSV endpoint first (typically more stable than HTML scraping).
    try:
        csv_url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/master/data/constituents.csv"
        )
        df = pd.read_csv(csv_url)
        if "Symbol" in df.columns:
            tickers = [t.replace(".", "-") for t in df["Symbol"].astype(str).tolist()]
            if tickers:
                return sorted(set(tickers))
    except Exception as exc:
        errors.append(f"github csv: {exc}")

    # Fallback to Wikipedia with a browser-like user agent.
    try:
        req = Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        table = pd.read_html(StringIO(html))[0]
        tickers = [t.replace(".", "-") for t in table["Symbol"].astype(str).tolist()]
        if tickers:
            return sorted(set(tickers))
    except (HTTPError, URLError, Exception) as exc:
        errors.append(f"wikipedia: {exc}")

    detail = "; ".join(errors) if errors else "unknown source failure"
    raise RuntimeError(
        f"Unable to fetch S&P 500 constituents ({detail}). "
        "Use Custom tickers as a fallback."
    )


def zscore(value: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 0.0
    return (value - mean) / std


def clamp_0_100(value: float) -> int:
    return int(max(0, min(100, round(value))))


def extract_latest_metrics(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for ticker in tickers:
        try:
            if is_multi:
                close = raw[("Close", ticker)].dropna()
                volume = raw[("Volume", ticker)].dropna()
            else:
                close = raw["Close"].dropna()
                volume = raw["Volume"].dropna()

            if len(close) < 2 or len(volume) < 2:
                continue

            latest_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            latest_volume = float(volume.iloc[-1])
            baseline_window = volume.iloc[-21:-1] if len(volume) >= 21 else volume.iloc[:-1]
            baseline_med = float(baseline_window.median()) if len(baseline_window) else latest_volume

            move_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0
            vol_ratio = latest_volume / baseline_med if baseline_med > 0 else 1.0

            rows.append(
                {
                    "ticker": ticker,
                    "price": latest_close,
                    "price_move_pct": move_pct,
                    "volume_ratio": vol_ratio,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def fetch_single_ticker_metric(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=ticker,
            period="2mo",
            interval="1d",
            group_by="column",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return extract_latest_metrics(raw, [ticker])
    except Exception:
        return pd.DataFrame()


def score_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    abs_move = df["price_move_pct"].abs()

    move_mean, move_std = abs_move.mean(), abs_move.std(ddof=0)
    vol_mean, vol_std = df["volume_ratio"].mean(), df["volume_ratio"].std(ddof=0)

    base_scores = []
    for _, row in df.iterrows():
        move_component = max(0.0, zscore(abs(row["price_move_pct"]), move_mean, move_std))
        vol_component = max(0.0, zscore(row["volume_ratio"], vol_mean, vol_std))

        boost = 0.0
        if abs(row["price_move_pct"]) >= 2.0 and row["volume_ratio"] > 1.5:
            boost += 0.8

        raw = 45 + (15 * move_component) + (12 * vol_component) + (16 * boost)
        base_scores.append(clamp_0_100(raw))

    out = df.copy()
    out["activity_score"] = base_scores
    return out


def _headline_sentiment(title: str) -> float:
    words = {w.strip(".,:;!?()[]{}\"'").lower() for w in title.split()}
    pos = len(words & SENT_POS)
    neg = len(words & SENT_NEG)
    return float(pos - neg)


def fetch_news_features(tickers: list[str], lookback_hours: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

    for ticker in tickers:
        mention_count = 0
        providers: set[str] = set()
        sent_values: list[float] = []
        filing_flag = False

        try:
            news_items = yf.Ticker(ticker).news or []
        except Exception:
            news_items = []

        for item in news_items:
            publish_ts = item.get("providerPublishTime")
            if publish_ts is None:
                continue

            published = datetime.fromtimestamp(publish_ts, tz=UTC)
            if published < cutoff:
                continue

            mention_count += 1
            provider = str(item.get("publisher") or item.get("provider") or "unknown").strip()
            if provider:
                providers.add(provider)

            title = str(item.get("title") or "")
            sent_values.append(_headline_sentiment(title))

            low_title = title.lower()
            if any(k in low_title for k in FILING_HINTS):
                filing_flag = True

        sentiment_shift = 0.0
        if sent_values:
            sentiment_shift = max(sent_values) - min(sent_values)

        out[ticker] = {
            "mention_count": mention_count,
            "source_diversity": len(providers),
            "filing_flag": filing_flag,
            "sentiment_shift": sentiment_shift,
        }

    return out


def apply_news_adjustments(df: pd.DataFrame, news: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["mention_count"] = out["ticker"].map(lambda t: news.get(t, {}).get("mention_count", 0))
    out["source_diversity"] = out["ticker"].map(lambda t: news.get(t, {}).get("source_diversity", 0))
    out["filing_flag"] = out["ticker"].map(lambda t: bool(news.get(t, {}).get("filing_flag", False)))
    out["sentiment_shift"] = out["ticker"].map(lambda t: float(news.get(t, {}).get("sentiment_shift", 0.0)))

    mention_q90 = max(float(out["mention_count"].quantile(0.9)), 1.0)
    diversity_q90 = max(float(out["source_diversity"].quantile(0.9)), 1.0)
    sentiment_q90 = max(float(out["sentiment_shift"].quantile(0.9)), 0.1)

    adjusted = []
    for _, row in out.iterrows():
        score = float(row["activity_score"])
        mention_norm = min(1.0, math.log1p(row["mention_count"]) / math.log1p(mention_q90))
        diversity_norm = min(1.0, row["source_diversity"] / diversity_q90)
        sentiment_norm = min(1.0, row["sentiment_shift"] / sentiment_q90)

        score += 14 * mention_norm
        score += 9 * diversity_norm
        score += 7 * sentiment_norm
        if row["filing_flag"]:
            score += 18

        if row["mention_count"] >= mention_q90 and row["source_diversity"] >= diversity_q90:
            score += 4

        adjusted.append(clamp_0_100(score))

    out["activity_score"] = adjusted
    return out


def selection_reason(row: pd.Series) -> str:
    parts: list[str] = []

    if row.get("filing_flag", False):
        parts.append("possible filing-related headline flow")

    move = float(row["price_move_pct"])
    vol = float(row["volume_ratio"])

    if abs(move) >= 2.0 and vol > 1.5:
        direction = "upside" if move > 0 else "downside"
        parts.append(f"{direction} move of {move:+.2f}% with {vol:.2f}x relative volume")
    elif abs(move) >= 1.2:
        parts.append(f"notable price move of {move:+.2f}%")
    elif vol > 1.5:
        parts.append(f"volume expansion ({vol:.2f}x vs baseline)")

    mentions = int(row.get("mention_count", 0))
    diversity = int(row.get("source_diversity", 0))
    if mentions > 0:
        parts.append(f"{mentions} recent news mentions across {max(1, diversity)} sources")

    sent = float(row.get("sentiment_shift", 0.0))
    if sent >= 1.0:
        parts.append("headline sentiment shifted materially")

    if not parts:
        parts.append("high composite activity relative to peer universe")

    return "; ".join(parts)


def save_run(run_record: dict[str, Any]) -> None:
    ensure_data_dir()
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, ensure_ascii=True) + "\n")


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    out = []
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def compute_run_performance(run: dict[str, Any]) -> pd.DataFrame:
    picks = run.get("top_candidates", [])
    if not picks:
        return pd.DataFrame()

    tickers = [p["ticker"] for p in picks]
    entry_map = {p["ticker"]: p.get("entry_price") for p in picks}

    latest = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        group_by="column",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []
    for t in tickers:
        try:
            if isinstance(latest.columns, pd.MultiIndex):
                close = latest[("Close", t)].dropna()
            else:
                close = latest["Close"].dropna()
            if close.empty:
                continue

            current = float(close.iloc[-1])
            entry = float(entry_map[t])
            ret = ((current - entry) / entry) * 100 if entry else 0.0
            rows.append({"ticker": t, "entry_price": entry, "current_price": current, "return_pct": ret})
        except Exception:
            continue

    return pd.DataFrame(rows).sort_values("return_pct", ascending=False)


def market_is_open_now() -> bool:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    minutes = now_et.hour * 60 + now_et.minute
    open_minutes = 9 * 60 + 30
    close_minutes = 16 * 60
    return open_minutes <= minutes < close_minutes


def fmt_money(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"${float(value):,.2f}"


def fmt_large_number(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    num = float(value)
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    return f"{num:,.0f}"


def fmt_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.2f}%"


@st.cache_data(ttl=60)
def get_ticker_detail_snapshot(ticker: str, range_key: str) -> dict[str, Any]:
    range_to_period_interval = {
        "1D": ("1d", "1m"),
        "5D": ("5d", "15m"),
        "1M": ("1mo", "1h"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
    }
    period, interval = range_to_period_interval.get(range_key, ("3mo", "1d"))

    tk = yf.Ticker(ticker)
    hist = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        group_by="column",
        auto_adjust=False,
        prepost=(range_key == "1D"),
        progress=False,
        threads=True,
    )
    close_series = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)
    if close_series.empty:
        return {
            "chart": pd.DataFrame(),
            "price_label": "Unavailable",
            "price_value": None,
            "prev_close": None,
            "change": None,
            "change_pct": None,
            "name": ticker,
            "exchange": "-",
            "stats": {},
        }

    price_label = "Market close"
    price_value = float(close_series.iloc[-1])
    prev_close = None

    daily = yf.download(
        tickers=ticker,
        period="5d",
        interval="1d",
        group_by="column",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    daily_close = daily["Close"].dropna() if "Close" in daily.columns else pd.Series(dtype=float)
    if len(daily_close) >= 2:
        prev_close = float(daily_close.iloc[-2])
    elif len(daily_close) == 1:
        prev_close = float(daily_close.iloc[-1])

    if market_is_open_now() and range_key == "1D":
        try:
            live = yf.download(
                tickers=ticker,
                period="1d",
                interval="1m",
                group_by="column",
                auto_adjust=False,
                prepost=True,
                progress=False,
                threads=True,
            )
            live_close = live["Close"].dropna() if "Close" in live.columns else pd.Series(dtype=float)
            if not live_close.empty:
                price_label = "Latest"
                price_value = float(live_close.iloc[-1])
                close_series = live_close
                chart_idx_name = "Datetime"
            else:
                chart_idx_name = close_series.index.name or "Date"
        except Exception:
            chart_idx_name = close_series.index.name or "Date"
    else:
        chart_idx_name = close_series.index.name or "Date"

    close_series = close_series[close_series > 0]
    chart = close_series.reset_index()
    chart.columns = [chart_idx_name, "Close"]
    if chart_idx_name != "Date":
        chart = chart.rename(columns={chart_idx_name: "Date"})

    info: dict[str, Any] = {}
    fast: dict[str, Any] = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        fast = dict(tk.fast_info) if tk.fast_info is not None else {}
    except Exception:
        fast = {}

    name = str(info.get("longName") or info.get("shortName") or ticker)
    exchange = str(info.get("exchange") or info.get("fullExchangeName") or "-")

    last_daily = daily.iloc[-1] if not daily.empty else None
    open_px = float(last_daily["Open"]) if last_daily is not None and "Open" in last_daily else None
    high_px = float(last_daily["High"]) if last_daily is not None and "High" in last_daily else None
    low_px = float(last_daily["Low"]) if last_daily is not None and "Low" in last_daily else None

    market_cap = info.get("marketCap") or fast.get("market_cap")
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    year_high = info.get("fiftyTwoWeekHigh") or fast.get("year_high")
    year_low = info.get("fiftyTwoWeekLow") or fast.get("year_low")
    dividend_yield = info.get("dividendYield")
    if dividend_yield is not None:
        dividend_yield = float(dividend_yield) * 100
    dividend_rate = info.get("dividendRate")

    change = None
    change_pct = None
    if prev_close and prev_close != 0:
        change = price_value - prev_close
        change_pct = (change / prev_close) * 100

    return {
        "chart": chart,
        "price_label": price_label,
        "price_value": price_value,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "name": name,
        "exchange": exchange,
        "stats": {
            "Open": open_px,
            "High": high_px,
            "Low": low_px,
            "Prev close": prev_close,
            "Mkt cap": market_cap,
            "P/E ratio": pe_ratio,
            "52-wk high": year_high,
            "52-wk low": year_low,
            "Dividend yield": dividend_yield,
            "Dividend rate": dividend_rate,
        },
    }


@st.cache_data(ttl=300)
def get_ticker_research_notes(ticker: str, lookback_hours: int, max_items: int = 12) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        news_items = []

    for item in news_items:
        ts = item.get("providerPublishTime")
        if ts is None:
            continue
        published_utc = datetime.fromtimestamp(ts, tz=UTC)
        title = str(item.get("title") or "").strip()
        if not title:
            continue

        provider = str(item.get("publisher") or item.get("provider") or "Unknown").strip()
        link = str(item.get("link") or "")
        sentiment = _headline_sentiment(title)
        low_title = title.lower()
        filing_hint = any(k in low_title for k in FILING_HINTS)
        in_window = published_utc >= cutoff

        tags: list[str] = []
        if in_window:
            tags.append("contributed to activity score window")
        if filing_hint:
            tags.append("filing-related")
        if sentiment >= 1:
            tags.append("positive sentiment signal")
        elif sentiment <= -1:
            tags.append("negative sentiment signal")
        if not tags:
            tags.append("potential price-moving headline")

        notes.append(
            {
                "published_utc": published_utc,
                "published_et": published_utc.astimezone(ZoneInfo("America/New_York")),
                "provider": provider,
                "title": title,
                "link": link,
                "tags": tags,
                "in_window": in_window,
            }
        )

    notes.sort(key=lambda x: x["published_utc"], reverse=True)
    return notes[:max_items]


def run_scan(universe: list[str], top_n: int, lookback_hours: int) -> pd.DataFrame:
    chunk_size = 80
    parts: list[pd.DataFrame] = []

    for i in range(0, len(universe), chunk_size):
        chunk = universe[i : i + chunk_size]
        try:
            market = yf.download(
                tickers=chunk,
                period="2mo",
                interval="1d",
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            chunk_metrics = extract_latest_metrics(market, chunk)
            if not chunk_metrics.empty:
                parts.append(chunk_metrics)
        except Exception:
            continue

    if not parts:
        raise RuntimeError("No market data returned from provider (possible rate-limit/block).")

    metrics = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ticker"], keep="last")
    if metrics.empty:
        return metrics

    found = set(metrics["ticker"].astype(str).tolist())
    missing = sorted(set(universe) - found)
    if missing:
        retry_parts: list[pd.DataFrame] = []
        for t in missing:
            one = fetch_single_ticker_metric(t)
            if not one.empty:
                retry_parts.append(one)
        if retry_parts:
            retry_df = pd.concat(retry_parts, ignore_index=True)
            metrics = (
                pd.concat([metrics, retry_df], ignore_index=True)
                .drop_duplicates(subset=["ticker"], keep="last")
                .reset_index(drop=True)
            )
        found = set(metrics["ticker"].astype(str).tolist())
        missing = sorted(set(universe) - found)

    scored = score_base_metrics(metrics)
    candidate_pool = scored.sort_values("activity_score", ascending=False).head(max(top_n * 6, 80))

    news_features = fetch_news_features(candidate_pool["ticker"].tolist(), lookback_hours=lookback_hours)
    rescored = apply_news_adjustments(scored, news_features)
    rescored = rescored.sort_values(["activity_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    rescored["selection_reason"] = rescored.apply(selection_reason, axis=1)
    rescored.attrs["missing_tickers"] = missing

    return rescored


@st.cache_data(ttl=120)
def get_single_ticker_activity(ticker: str, lookback_hours: int) -> pd.DataFrame:
    metrics = fetch_single_ticker_metric(ticker)
    if metrics.empty:
        return pd.DataFrame()
    scored = score_base_metrics(metrics)
    news_features = fetch_news_features([ticker], lookback_hours=lookback_hours)
    rescored = apply_news_adjustments(scored, news_features)
    rescored["selection_reason"] = rescored.apply(selection_reason, axis=1)
    return rescored


def main() -> None:
    st.set_page_config(page_title="SignalWeave Adhoc Scanner", layout="wide")
    st.title("SignalWeave: Adhoc Market Activity Scanner")
    st.caption("Run on demand to get top ticker picks, drivers, and post-run performance in one place.")

    if yf is None:
        st.error("Missing dependency: yfinance")
        st.code(
            f"pip install -r {APP_DIR / 'requirements.txt'}",
            language="bash",
        )
        st.caption("If that fails, run: pip install yfinance")
        st.stop()

    if "last_ranked" not in st.session_state:
        st.session_state["last_ranked"] = None
    if "last_top_n" not in st.session_state:
        st.session_state["last_top_n"] = 15
    if "last_universe_size" not in st.session_state:
        st.session_state["last_universe_size"] = 0
    if "last_run_at" not in st.session_state:
        st.session_state["last_run_at"] = None
    if "last_missing_tickers" not in st.session_state:
        st.session_state["last_missing_tickers"] = []

    with st.sidebar:
        st.subheader("Scan Controls")
        universe_mode = st.selectbox("Universe", ["S&P 500", "Custom tickers"], index=0)
        top_n = st.slider("Top picks", min_value=5, max_value=30, value=15, step=1)
        lookback_hours = st.slider("News lookback (hours)", min_value=6, max_value=48, value=24, step=6)

        custom_input = ""
        if universe_mode == "Custom tickers":
            custom_input = st.text_area("Tickers (comma-separated)", value="AAPL,MSFT,NVDA,TSLA,AMZN")

        run_clicked = st.button("Run Adhoc Scan", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running scan (market + news)..."):
            try:
                if universe_mode == "S&P 500":
                    universe = get_sp500_tickers()
                else:
                    universe = sorted(
                        {t.strip().upper().replace(".", "-") for t in custom_input.split(",") if t.strip()}
                    )
                    if not universe:
                        st.error("Enter at least one ticker for custom mode.")
                        st.stop()

                ranked = run_scan(universe, top_n=top_n, lookback_hours=lookback_hours)
                if ranked.empty:
                    st.error("No market data returned. Try again in a few minutes.")
                    st.stop()

                now = datetime.now(UTC).isoformat()
                top_df = ranked.head(top_n).copy()
                st.session_state["last_ranked"] = ranked
                st.session_state["last_top_n"] = top_n
                st.session_state["last_universe_size"] = len(universe)
                st.session_state["last_run_at"] = now
                st.session_state["last_missing_tickers"] = ranked.attrs.get("missing_tickers", [])

                run_record = {
                    "run_at_utc": now,
                    "universe": universe_mode,
                    "universe_size": len(universe),
                    "top_n": top_n,
                    "top_candidates": [
                        {
                            "ticker": r["ticker"],
                            "activity_score": int(r["activity_score"]),
                            "entry_price": float(r["price"]),
                            "selection_reason": r["selection_reason"],
                        }
                        for _, r in top_df.iterrows()
                    ],
                }
                save_run(run_record)

                st.success(f"Scan complete: {len(universe)} tickers processed at {now}.")

            except Exception as exc:
                st.error(f"Scan failed: {exc}")

    selected_ticker_for_notes = ""
    if st.session_state["last_ranked"] is not None:
        ranked = st.session_state["last_ranked"]
        top_n = int(st.session_state["last_top_n"])
        universe_size = int(st.session_state["last_universe_size"])
        run_at = st.session_state["last_run_at"]
        top_df = ranked.head(top_n).copy()

        if run_at:
            st.caption(f"Showing latest scan: {run_at}")
        missing_tickers = st.session_state.get("last_missing_tickers", [])
        if missing_tickers:
            preview = ", ".join(missing_tickers[:12])
            extra = "" if len(missing_tickers) <= 12 else f" (+{len(missing_tickers) - 12} more)"
            st.warning(
                f"{len(missing_tickers)} tickers were unavailable from the market data provider in this run. "
                f"Examples: {preview}{extra}"
            )

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Top Picks")
            st.dataframe(
                top_df[
                    [
                        "ticker",
                        "activity_score",
                        "price_move_pct",
                        "volume_ratio",
                        "mention_count",
                        "source_diversity",
                        "filing_flag",
                        "selection_reason",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.subheader("Run Snapshot")
            st.metric("Universe size", f"{universe_size}")
            avg_score = float(top_df["activity_score"].mean())
            avg_move = float(top_df["price_move_pct"].mean())
            st.metric("Avg top score", f"{avg_score:.1f}")
            st.metric("Avg top move", f"{avg_move:+.2f}%")

        st.subheader("Ticker Lookup")
        ticker_query = st.text_input(
            "Search ticker",
            value="",
            key="ticker_search_text",
            placeholder="Type ticker (e.g., NVDA) and press Enter",
        ).strip().upper()
        selected_ticker = ticker_query
        selected_ticker_for_notes = selected_ticker

        selected_row = pd.DataFrame()
        if selected_ticker:
            from_last_scan = ranked[ranked["ticker"] == selected_ticker]
            if not from_last_scan.empty:
                selected_row = from_last_scan.head(1).copy()
            else:
                with st.spinner(f"Fetching {selected_ticker} activity..."):
                    selected_row = get_single_ticker_activity(selected_ticker, lookback_hours=lookback_hours)

        if not selected_ticker:
            st.info("Enter a ticker symbol to view activity score and details.")
        elif selected_row.empty:
            st.warning(f"No data available for {selected_ticker}.")
        else:
            summary_cols = [
                "ticker",
                "activity_score",
                "price_move_pct",
                "volume_ratio",
                "mention_count",
                "source_diversity",
                "filing_flag",
                "selection_reason",
            ]
            st.dataframe(
                selected_row[summary_cols],
                use_container_width=True,
                hide_index=True,
                height="content",
            )

        st.subheader("Ticker Details")
        if not selected_ticker:
            st.info("Enter a ticker to load details.")
        else:
            timeframe = st.radio(
                "Timeframe",
                options=["1D", "5D", "1M", "3M", "6M", "1Y"],
                index=0,
                horizontal=True,
                key="detail_timeframe",
            )
            try:
                snap = get_ticker_detail_snapshot(selected_ticker, timeframe)
                if snap["price_value"] is None:
                    st.warning("No pricing data available for selected ticker.")
                else:
                    st.markdown(f"### {snap['name']}")
                    st.caption(f"{snap['exchange']}: {selected_ticker}")

                    change = snap.get("change")
                    change_pct = snap.get("change_pct")
                    price_text = fmt_money(snap["price_value"])
                    if change is not None and change_pct is not None:
                        sign = "+" if change >= 0 else ""
                        st.markdown(
                            f"## {price_text}\n\n"
                            f"<span style='color:{'#22c55e' if change >= 0 else '#f87171'};'>"
                            f"{sign}{change:.2f} ({sign}{change_pct:.2f}%)</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"## {price_text}")
                    st.caption(snap["price_label"])

                    if go is None:
                        st.line_chart(snap["chart"], x="Date", y="Close", use_container_width=True)
                    else:
                        chart_df = snap["chart"].copy()
                        chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
                        chart_df = chart_df.dropna(subset=["Date", "Close"])
                        series = chart_df["Close"].astype(float)
                        y_min = float(series.min())
                        y_max = float(series.max())
                        pad = (y_max - y_min) * 0.15 if y_max > y_min else max(1.0, abs(y_max) * 0.02)
                        line_color = "#22c55e" if (change or 0) >= 0 else "#f87171"

                        fig = go.Figure()
                        intraday_tickvals = None
                        intraday_ticktext = None
                        intraday_xrange = None
                        if timeframe == "1D":
                            intraday_df = chart_df.copy()
                            if not intraday_df.empty and intraday_df["Date"].dt.tz is None:
                                intraday_df["Date"] = intraday_df["Date"].dt.tz_localize("UTC")
                            if not intraday_df.empty:
                                intraday_df["DateET"] = intraday_df["Date"].dt.tz_convert("America/New_York")
                                trade_day = intraday_df["DateET"].iloc[-1].normalize()
                                intraday_tickvals = [
                                    trade_day + pd.Timedelta(hours=9, minutes=30),
                                    trade_day + pd.Timedelta(hours=12),
                                    trade_day + pd.Timedelta(hours=16),
                                    trade_day + pd.Timedelta(hours=20),
                                ]
                                intraday_ticktext = ["9:30AM", "12:00PM", "4:00PM", "8:00PM"]
                                intraday_xrange = [intraday_tickvals[0], intraday_tickvals[-1]]
                                intraday_df["DateETNaive"] = intraday_df["DateET"].dt.tz_localize(None)
                                intraday_tickvals = [t.tz_localize(None) for t in intraday_tickvals]
                                market_close = intraday_df["DateET"].dt.normalize() + pd.Timedelta(hours=16)
                                regular = intraday_df[intraday_df["DateET"] <= market_close]
                                after = intraday_df[intraday_df["DateET"] > market_close]

                                if not regular.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=regular["DateETNaive"],
                                            y=regular["Close"],
                                            mode="lines",
                                            line={"width": 3, "color": line_color},
                                            name=f"{selected_ticker} (regular)",
                                            hovertemplate="%{x|%I:%M %p}<br>$%{y:.2f}<extra></extra>",
                                            connectgaps=False,
                                        )
                                    )
                                if not after.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=after["DateETNaive"],
                                            y=after["Close"],
                                            mode="lines",
                                            line={"width": 3, "color": "rgba(148,163,184,0.85)"},
                                            name=f"{selected_ticker} (after-hours)",
                                            hovertemplate="%{x|%I:%M %p}<br>$%{y:.2f}<extra></extra>",
                                            connectgaps=False,
                                        )
                                    )
                            else:
                                fig.add_trace(
                                    go.Scatter(
                                        x=chart_df["Date"],
                                        y=chart_df["Close"],
                                        mode="lines",
                                        line={"width": 3, "color": line_color},
                                        name=selected_ticker,
                                        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
                                        connectgaps=False,
                                    )
                                )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=chart_df["Date"],
                                    y=chart_df["Close"],
                                    mode="lines",
                                    line={"width": 3, "color": line_color},
                                    fill="tozeroy",
                                    fillcolor="rgba(248,113,113,0.12)" if line_color == "#f87171" else "rgba(34,197,94,0.12)",
                                    name=selected_ticker,
                                    hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
                                    connectgaps=False,
                                )
                            )
                        prev_close = snap.get("prev_close")
                        if timeframe == "1D" and prev_close is not None:
                            fig.add_hline(
                                y=float(prev_close),
                                line_dash="dot",
                                line_color="rgba(148,163,184,0.8)",
                                annotation_text=f"Prev close {prev_close:.2f}",
                                annotation_position="top right",
                            )
                        fig.update_layout(
                            margin={"l": 20, "r": 20, "t": 10, "b": 20},
                            height=460,
                            xaxis_title=None,
                            yaxis_title=None,
                            showlegend=False,
                            hovermode="x",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        fig.update_yaxes(
                            range=[y_min - pad, y_max + pad],
                            rangemode="normal",
                            gridcolor="rgba(148,163,184,0.20)",
                            zeroline=False,
                            tickformat=".2f",
                        )
                        if timeframe == "1D" and intraday_tickvals is not None:
                            # Force exactly four visible ticks for intraday view.
                            fig.update_layout(
                                xaxis={
                                    "type": "date",
                                    "showgrid": True,
                                    "gridcolor": "rgba(148,163,184,0.15)",
                                    "tickmode": "array",
                                    "tickvals": intraday_tickvals,
                                    "ticktext": intraday_ticktext,
                                    "range": intraday_xrange,
                                    "ticks": "outside",
                                    "ticklabelmode": "instant",
                                    "rangeslider": {"visible": False},
                                }
                            )
                        else:
                            fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
                        st.plotly_chart(fig, use_container_width=True)

                    stats = snap.get("stats", {})
                    stats_rows = [
                        {"Metric": "Open", "Value": fmt_money(stats.get("Open"))},
                        {"Metric": "High", "Value": fmt_money(stats.get("High"))},
                        {"Metric": "Low", "Value": fmt_money(stats.get("Low"))},
                        {"Metric": "Prev close", "Value": fmt_money(stats.get("Prev close"))},
                        {"Metric": "Mkt cap", "Value": fmt_large_number(stats.get("Mkt cap"))},
                        {
                            "Metric": "P/E ratio",
                            "Value": "-"
                            if stats.get("P/E ratio") is None
                            else f"{float(stats.get('P/E ratio')):.2f}",
                        },
                        {"Metric": "52-wk high", "Value": fmt_money(stats.get("52-wk high"))},
                        {"Metric": "52-wk low", "Value": fmt_money(stats.get("52-wk low"))},
                        {"Metric": "Dividend yield", "Value": fmt_pct(stats.get("Dividend yield"))},
                        {"Metric": "Dividend rate", "Value": fmt_money(stats.get("Dividend rate"))},
                    ]
                    st.dataframe(
                        pd.DataFrame(stats_rows),
                        use_container_width=True,
                        hide_index=True,
                        height="content",
                    )
            except Exception as exc:
                st.warning(f"Unable to load ticker details: {exc}")

    st.divider()
    st.subheader("Research Notes")
    if not selected_ticker_for_notes and st.session_state["last_ranked"] is not None:
        fallback = st.session_state["last_ranked"].head(1)
        if not fallback.empty:
            selected_ticker_for_notes = str(fallback.iloc[0]["ticker"])

    if not selected_ticker_for_notes:
        st.info("Run a scan and select a ticker to load research notes.")
    else:
        st.caption(
            f"News signals for {selected_ticker_for_notes} that may explain activity score changes "
            f"and additional potential movers."
        )
        notes = get_ticker_research_notes(selected_ticker_for_notes, lookback_hours=lookback_hours, max_items=12)
        if not notes:
            st.info("No recent news found for this ticker.")
        else:
            for n in notes:
                ts = n["published_et"].strftime("%b %d, %I:%M %p ET")
                tag_text = " | ".join(n["tags"])
                if n["link"]:
                    st.markdown(f"- [{n['title']}]({n['link']})")
                else:
                    st.markdown(f"- {n['title']}")
                st.caption(f"{n['provider']} • {ts} • {tag_text}")

    st.divider()
    st.subheader("Performance Tracker")

    history = load_history()
    if not history:
        st.info("No previous runs yet. Run a scan to start tracking performance.")
        return

    options = [f"{h['run_at_utc']} | {h.get('universe', 'N/A')} | top {h.get('top_n', 0)}" for h in history]
    selected_label = st.selectbox("Pick a historical run", options=options, index=len(options) - 1)
    selected_idx = options.index(selected_label)
    selected_run = history[selected_idx]

    if st.button("Refresh Performance", use_container_width=False):
        with st.spinner("Updating current prices for selected run..."):
            perf = compute_run_performance(selected_run)
            if perf.empty:
                st.warning("Unable to compute performance for selected run.")
            else:
                st.dataframe(perf, use_container_width=True, hide_index=True)
                st.metric("Average return", f"{perf['return_pct'].mean():+.2f}%")


if __name__ == "__main__":
    main()
