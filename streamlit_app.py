#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from difflib import get_close_matches
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
SEC_RELEVANT_FORMS = {"8-K", "4", "10-Q", "10-K"}
SEC_LOOKBACK_DAYS_DEFAULT = 2
SEC_USER_AGENT = "SignalWeave/1.0 (contact: your-email@example.com)"
SEC_MAX_TICKERS_PER_SCAN = 75
SEC_REQUEST_THROTTLE_SEC = 0.12

# Tracks CIKs fetched in this process so throttle is only applied for first network fetch.
SEC_FETCHED_CIKS: set[str] = set()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(ttl=3600)
def get_sp500_universe_df() -> pd.DataFrame:
    errors: list[str] = []

    # Prefer a simple CSV endpoint first (typically more stable than HTML scraping).
    try:
        csv_url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/master/data/constituents.csv"
        )
        df = pd.read_csv(csv_url)
        if "Symbol" in df.columns and "Security" in df.columns:
            out = pd.DataFrame(
                {
                    "ticker": df["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
                    "company_name": df["Security"].astype(str).str.strip(),
                }
            )
            out = out[(out["ticker"] != "") & (out["company_name"] != "")]
            out = out.drop_duplicates(subset=["ticker"], keep="first").sort_values(["company_name", "ticker"])
            if not out.empty:
                return out.reset_index(drop=True)
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
        if "Symbol" in table.columns:
            security_col = "Security" if "Security" in table.columns else None
            if security_col is None:
                security_col = "GICS Sub-Industry" if "GICS Sub-Industry" in table.columns else None
            if security_col is None:
                security_col = "Symbol"

            out = pd.DataFrame(
                {
                    "ticker": table["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False),
                    "company_name": table[security_col].astype(str).str.strip(),
                }
            )
            out = out[(out["ticker"] != "") & (out["company_name"] != "")]
            out = out.drop_duplicates(subset=["ticker"], keep="first").sort_values(["company_name", "ticker"])
            if not out.empty:
                return out.reset_index(drop=True)
    except (HTTPError, URLError, Exception) as exc:
        errors.append(f"wikipedia: {exc}")

    detail = "; ".join(errors) if errors else "unknown source failure"
    raise RuntimeError(
        f"Unable to fetch S&P 500 constituents ({detail}). "
        "Use Custom tickers as a fallback."
    )


def get_sp500_tickers() -> list[str]:
    return get_sp500_universe_df()["ticker"].tolist()


def zscore(value: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return 0.0
    return (value - mean) / std


def clamp_0_100(value: float) -> int:
    return int(max(0, min(100, round(value))))


def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if s.empty:
        return s

    med = float(s.median())
    mad = float((s - med).abs().median())
    scale = 1.4826 * mad
    if scale <= 1e-9:
        std = float(s.std(ddof=0))
        scale = std if std > 1e-9 else 1.0
    z = (s - med) / scale
    return z.clip(lower=-4.0, upper=4.0)


def logistic_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    scaled = s.clip(lower=-8.0, upper=8.0)
    return 100.0 / (1.0 + (-scaled).map(math.exp))


def normalized_score_from_raw(raw: pd.Series) -> pd.Series:
    s = pd.to_numeric(raw, errors="coerce").astype(float)
    out = pd.Series(50.0, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out

    # Preferred normalization: percentile rank within current universe.
    if len(valid) >= 2 and valid.nunique() > 1:
        try:
            # Use min-rank so large tie groups at the bottom stay near 0 instead of mid-percentile.
            pct = (valid.rank(method="min") - 1) / (len(valid) - 1) * 100.0
            out.loc[valid.index] = pct
            return out
        except Exception:
            pass

    # Stable behavior for identical values across a universe.
    if len(valid) > 1 and valid.nunique() <= 1:
        out.loc[valid.index] = 50.0
        return out

    # Fallback for cases where percentile isn't meaningful (e.g., single ticker).
    out.loc[valid.index] = logistic_score(valid)
    return out


def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "market_activity_score" not in out.columns:
        out["market_activity_score"] = 50.0
    if "info_flow_score" not in out.columns:
        out["info_flow_score"] = 50.0
    if "activity_score" not in out.columns:
        out["activity_score"] = (0.65 * out["market_activity_score"] + 0.35 * out["info_flow_score"]).round().astype(int)
    if "filing_flag" not in out.columns:
        out["filing_flag"] = False
    if "filing_count" not in out.columns:
        out["filing_count"] = 0
    if "filing_types" not in out.columns:
        out["filing_types"] = [[] for _ in range(len(out))]
    if "latest_filing_time" not in out.columns:
        out["latest_filing_time"] = ""
    if "filing_links" not in out.columns:
        out["filing_links"] = [[] for _ in range(len(out))]
    if "filing_dates" not in out.columns:
        out["filing_dates"] = [[] for _ in range(len(out))]
    if "filings" not in out.columns:
        out["filings"] = [[] for _ in range(len(out))]
    return out


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


def _compute_metric_from_ohlcv(ticker: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()
    cols = {str(c).lower(): c for c in ohlcv.columns}
    close_col = cols.get("close")
    vol_col = cols.get("volume")
    if close_col is None or vol_col is None:
        return pd.DataFrame()

    close = pd.to_numeric(ohlcv[close_col], errors="coerce").dropna()
    volume = pd.to_numeric(ohlcv[vol_col], errors="coerce").dropna()
    if len(close) < 2 or len(volume) < 2:
        return pd.DataFrame()

    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    latest_volume = float(volume.iloc[-1])
    baseline_window = volume.iloc[-21:-1] if len(volume) >= 21 else volume.iloc[:-1]
    baseline_med = float(baseline_window.median()) if len(baseline_window) else latest_volume
    move_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0
    vol_ratio = latest_volume / baseline_med if baseline_med > 0 else 1.0

    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "price": latest_close,
                "price_move_pct": move_pct,
                "volume_ratio": vol_ratio,
            }
        ]
    )


def _download_yahoo_daily(tickers: list[str], retries: int = 3) -> pd.DataFrame:
    wait = 0.5
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period="2mo",
                interval="1d",
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = extract_latest_metrics(raw, tickers)
            if not df.empty:
                return df
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(wait)
            wait *= 2
    return pd.DataFrame()


def _fetch_stooq_single_ticker_metric(ticker: str) -> pd.DataFrame:
    # Stooq format expects "<symbol>.us" for US listings.
    symbol = ticker.lower().replace("-", ".")
    url = f"https://stooq.com/q/d/l/?s={symbol}.us&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    if df.empty or "Close" not in df.columns or str(df.iloc[0].get("Close", "")).lower() == "null":
        return pd.DataFrame()
    return _compute_metric_from_ohlcv(ticker, df)


def fetch_single_ticker_metric(ticker: str) -> pd.DataFrame:
    yahoo_df = _download_yahoo_daily([ticker], retries=3)
    if not yahoo_df.empty:
        return yahoo_df
    return _fetch_stooq_single_ticker_metric(ticker)


def score_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    abs_move = pd.to_numeric(out["price_move_pct"], errors="coerce").abs().fillna(0.0)
    vol_ratio = pd.to_numeric(out["volume_ratio"], errors="coerce").fillna(1.0)
    vol_excess = (vol_ratio - 1.0).clip(lower=0.0)

    move_z = robust_zscore(abs_move)
    vol_z = robust_zscore(vol_excess)
    combo_flag = ((abs_move >= 2.0) & (vol_ratio > 1.5)).astype(float)

    raw_market = (0.70 * move_z) + (0.30 * vol_z) + (0.35 * combo_flag)
    out["raw_market_activity"] = raw_market
    out["market_activity_score"] = normalized_score_from_raw(raw_market).round(1)

    # Placeholder until news metrics are merged.
    out["info_flow_score"] = 50.0
    out["activity_score"] = (0.65 * out["market_activity_score"] + 0.35 * out["info_flow_score"]).round().astype(int)
    return out


def _headline_sentiment(title: str) -> float:
    words = {w.strip(".,:;!?()[]{}\"'").lower() for w in title.split()}
    pos = len(words & SENT_POS)
    neg = len(words & SENT_NEG)
    return float(pos - neg)


@st.cache_data(ttl=24 * 3600)
def get_sec_ticker_cik_map() -> dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    req = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    out: dict[str, str] = {}
    if isinstance(payload, dict):
        for _, row in payload.items():
            if not isinstance(row, dict):
                continue
            t = str(row.get("ticker", "")).strip().upper()
            cik_num = row.get("cik_str")
            if not t or cik_num is None:
                continue
            try:
                cik = f"{int(cik_num):010d}"
            except Exception:
                continue
            out[t] = cik
    return out


@st.cache_data(ttl=6 * 3600)
def _get_sec_submissions_json_cached(cik_10: str) -> dict[str, Any]:
    url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
    req = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_sec_submissions_json(cik_10: str) -> dict[str, Any]:
    # Apply throttle only before first fetch for a CIK in this process.
    if cik_10 not in SEC_FETCHED_CIKS:
        time.sleep(SEC_REQUEST_THROTTLE_SEC)
    payload = _get_sec_submissions_json_cached(cik_10)
    SEC_FETCHED_CIKS.add(cik_10)
    return payload


def _default_sec_features() -> dict[str, Any]:
    return {
        "filing_flag": False,
        "filing_count": 0,
        "filing_types": [],
        "filing_dates": [],
        "latest_filing_time": "",
        "filing_links": [],
        "filings": [],
    }


@st.cache_data(ttl=3600)
def get_recent_sec_filings_for_ticker(ticker: str, lookback_days: int = SEC_LOOKBACK_DAYS_DEFAULT) -> dict[str, Any]:
    t = str(ticker).strip().upper()
    if not t:
        return _default_sec_features()
    try:
        cik_map = get_sec_ticker_cik_map()
    except Exception:
        return _default_sec_features()

    cik_10 = cik_map.get(t)
    if not cik_10:
        return _default_sec_features()

    try:
        sub = get_sec_submissions_json(cik_10)
        recent = (sub.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        filing_dates = recent.get("filingDate") or []
        accession_numbers = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []
    except Exception:
        return _default_sec_features()

    n = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_docs))
    if n <= 0:
        return _default_sec_features()

    cutoff = datetime.now(UTC).date() - timedelta(days=max(1, int(lookback_days)))
    selected: list[dict[str, str]] = []
    for i in range(n):
        form = str(forms[i] or "").strip().upper()
        if form not in SEC_RELEVANT_FORMS:
            continue
        date_str = str(filing_dates[i] or "").strip()
        try:
            filed_date = datetime.fromisoformat(date_str).date()
        except Exception:
            continue
        if filed_date < cutoff:
            continue

        acc = str(accession_numbers[i] or "").strip()
        doc = str(primary_docs[i] or "").strip()
        acc_nodash = acc.replace("-", "")
        try:
            cik_int = str(int(cik_10))
        except Exception:
            cik_int = cik_10.lstrip("0") or "0"
        index_link = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc}-index.html"
            if acc_nodash and acc
            else ""
        )
        primary_link = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}" if acc_nodash and doc else ""
        selected.append(
            {
                "form": form,
                "date": date_str,
                "link": index_link or primary_link,
                "index_link": index_link,
                "primary_link": primary_link,
            }
        )

    if not selected:
        return _default_sec_features()

    filing_types: list[str] = []
    for s in selected:
        if s["form"] not in filing_types:
            filing_types.append(s["form"])
    links = [s["link"] for s in selected if s.get("link")][:3]
    filing_dates = []
    for s in selected:
        d = s.get("date", "")
        if d and d not in filing_dates:
            filing_dates.append(d)
    latest = max((s["date"] for s in selected), default="")

    return {
        "filing_flag": True,
        "filing_count": len(selected),
        "filing_types": filing_types,
        "filing_dates": filing_dates[:5],
        "latest_filing_time": latest,
        "filing_links": links,
        "filings": selected[:10],
    }


def fetch_sec_features(tickers: list[str], lookback_days: int = SEC_LOOKBACK_DAYS_DEFAULT) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in tickers:
        try:
            out[t] = get_recent_sec_filings_for_ticker(t, lookback_days=lookback_days)
        except Exception:
            out[t] = _default_sec_features()
    return out


def choose_sec_tickers(
    scored: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    news_features: dict[str, dict[str, Any]],
    max_tickers: int = SEC_MAX_TICKERS_PER_SCAN,
) -> list[str]:
    max_tickers = max(1, int(max_tickers))
    chosen: list[str] = []
    seen: set[str] = set()

    # 1) Candidate pool first (already sorted by market_activity_score desc).
    for t in candidate_pool["ticker"].astype(str).tolist():
        if t not in seen:
            seen.add(t)
            chosen.append(t)
        if len(chosen) >= max_tickers:
            return chosen[:max_tickers]

    # 2) Any additional tickers with non-zero news mentions, prioritized by market score.
    mention_tickers = {
        str(t)
        for t, payload in news_features.items()
        if int(payload.get("mention_count", 0) or 0) > 0
    }
    if mention_tickers:
        ranked = scored.sort_values("market_activity_score", ascending=False)
        for t in ranked["ticker"].astype(str).tolist():
            if t in mention_tickers and t not in seen:
                seen.add(t)
                chosen.append(t)
            if len(chosen) >= max_tickers:
                break

    return chosen[:max_tickers]


def fetch_news_features(tickers: list[str], lookback_hours: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

    for ticker in tickers:
        mention_count = 0
        providers: set[str] = set()
        sent_values: list[float] = []

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

        sentiment_shift = 0.0
        if sent_values:
            sentiment_shift = max(sent_values) - min(sent_values)

        out[ticker] = {
            "mention_count": mention_count,
            "source_diversity": len(providers),
            "sentiment_shift": sentiment_shift,
        }

    return out


def apply_news_adjustments(
    df: pd.DataFrame,
    news: dict[str, dict[str, Any]],
    sec_features: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["mention_count"] = out["ticker"].map(lambda t: news.get(t, {}).get("mention_count", 0)).fillna(0)
    out["source_diversity"] = out["ticker"].map(lambda t: news.get(t, {}).get("source_diversity", 0)).fillna(0)
    out["sentiment_shift"] = out["ticker"].map(lambda t: float(news.get(t, {}).get("sentiment_shift", 0.0))).fillna(0.0)
    sec = sec_features or {}
    out["filing_flag"] = out["ticker"].map(lambda t: bool(sec.get(t, {}).get("filing_flag", False)))
    out["filing_count"] = out["ticker"].map(lambda t: int(sec.get(t, {}).get("filing_count", 0))).fillna(0).astype(int)
    out["filing_types"] = out["ticker"].map(lambda t: sec.get(t, {}).get("filing_types", []) or [])
    out["latest_filing_time"] = out["ticker"].map(lambda t: str(sec.get(t, {}).get("latest_filing_time", "")))
    out["filing_links"] = out["ticker"].map(lambda t: sec.get(t, {}).get("filing_links", []) or [])

    mention_log = out["mention_count"].map(lambda x: math.log1p(max(0.0, float(x))))
    source_log = out["source_diversity"].map(lambda x: math.log1p(max(0.0, float(x))))
    sent_abs = pd.to_numeric(out["sentiment_shift"], errors="coerce").abs().fillna(0.0)
    filing_num = out["filing_flag"].astype(float)
    filing_count_scaled = pd.to_numeric(out["filing_count"], errors="coerce").fillna(0.0).clip(lower=0, upper=3) / 3.0

    mention_z = robust_zscore(mention_log)
    source_z = robust_zscore(source_log)
    sent_z = robust_zscore(sent_abs)
    interaction = ((out["mention_count"] > 0) & (out["source_diversity"] > 0)).astype(float)

    raw_info = (
        (0.40 * mention_z)
        + (0.25 * source_z)
        + (0.20 * sent_z)
        + (0.20 * interaction)
        + (0.25 * filing_num)
        + (0.15 * filing_count_scaled)
    )
    out["raw_info_flow"] = raw_info
    out["info_flow_score"] = normalized_score_from_raw(raw_info).round(1)

    market_score = pd.to_numeric(out.get("market_activity_score", 50.0), errors="coerce").fillna(50.0)
    info_score = pd.to_numeric(out["info_flow_score"], errors="coerce").fillna(50.0)
    out["activity_score"] = (0.65 * market_score + 0.35 * info_score).round().astype(int)
    return out


def selection_reason(row: pd.Series) -> str:
    parts: list[str] = []

    if row.get("filing_flag", False):
        filing_count = int(row.get("filing_count", 0) or 0)
        filing_types = row.get("filing_types", []) or []
        types_label = ", ".join(str(x) for x in filing_types[:3]) if filing_types else "SEC filings"
        if filing_count > 0:
            parts.append(f"recent SEC filing activity ({filing_count}: {types_label})")
        else:
            parts.append("recent SEC filing activity")

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


def rank_all_stocks_matches(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = str(query or "").strip().upper()
    base = df.copy()
    if base.empty:
        return base
    if not q:
        return base.sort_values(["company_name", "ticker"]).reset_index(drop=True)

    ticker_u = base["ticker"].astype(str).str.upper()
    company_u = base["company_name"].astype(str).str.upper()

    # If query is an exact ticker, return only that ticker for a crisp experience.
    exact_ticker = base[ticker_u == q]
    if not exact_ticker.empty:
        return exact_ticker.sort_values(["company_name", "ticker"]).reset_index(drop=True)

    labels = (ticker_u + " " + company_u).tolist()

    fuzzy_hits: set[int] = set()
    if len(q) >= 2:
        fuzzy_labels = set(get_close_matches(q, labels, n=min(30, len(labels)), cutoff=0.75))
        fuzzy_hits = {i for i, label in enumerate(labels) if label in fuzzy_labels}

    rank_bucket: list[int | None] = []
    for i in range(len(base)):
        t = ticker_u.iloc[i]
        c = company_u.iloc[i]
        bucket: int | None = None
        if t == q:
            bucket = 0
        elif t.startswith(q):
            bucket = 1
        elif c.startswith(q):
            bucket = 2
        elif q in t:
            bucket = 3
        elif q in c:
            bucket = 4
        elif i in fuzzy_hits:
            bucket = 5
        rank_bucket.append(bucket)

    ranked = base.copy()
    ranked["_rank_bucket"] = rank_bucket
    ranked = ranked[ranked["_rank_bucket"].notna()].copy()
    if ranked.empty:
        return ranked.drop(columns=["_rank_bucket"], errors="ignore")
    ranked["_rank_bucket"] = ranked["_rank_bucket"].astype(int)
    ranked = ranked.sort_values(["_rank_bucket", "company_name", "ticker"]).reset_index(drop=True)
    return ranked.drop(columns=["_rank_bucket"], errors="ignore")


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
        in_window = published_utc >= cutoff

        tags: list[str] = []
        if in_window:
            tags.append("contributed to activity score window")
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


@st.cache_data(ttl=300)
def get_ticker_news_items(ticker: str, lookback_hours: int, max_items: int = 15) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception:
        news_items = []

    for n in news_items:
        ts = n.get("providerPublishTime")
        if ts is None:
            continue
        published = datetime.fromtimestamp(ts, tz=UTC)
        if published < cutoff:
            continue
        title = str(n.get("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "headline": title,
                "snippet": str(n.get("summary") or n.get("snippet") or "").strip(),
                "source": str(n.get("publisher") or n.get("provider") or "Unknown").strip(),
                "time_utc": published.isoformat(),
                "url": str(n.get("link") or "").strip(),
            }
        )

    items.sort(key=lambda x: x["time_utc"], reverse=True)
    return items[:max_items]


def build_codex_ticker_prompt(bundle: dict[str, Any]) -> str:
    return (
        "You are a market surveillance research analyst. Use ONLY the evidence JSON below. "
        "Do not use external sources or assumptions beyond the provided evidence.\n\n"
        f"Evidence JSON:\n{json.dumps(bundle, ensure_ascii=True, indent=2)}\n\n"
        "Task:\n"
        "1) Classify signal_category as one of: EARLY_CATALYST, CONFIRMED_MOMENTUM, NO_SIGNAL.\n"
        "2) Provide recommended_action as one of: BUY, SELL, HOLD, NO_ACTION (paper trading context only).\n"
        "3) Provide confidence as an integer 0-100.\n"
        "4) List catalysts.\n"
        "5) List evidence items (concise).\n"
        "6) List contradictions / uncertainty factors.\n"
        "7) Give 1-3 sentence reasoning.\n"
        "8) Provide time_horizon (e.g., intraday, 1-3 days, 1-2 weeks).\n\n"
        "Output STRICT JSON only in this schema:\n"
        "{\n"
        '  "ticker": "STRING",\n'
        '  "signal_category": "EARLY_CATALYST|CONFIRMED_MOMENTUM|NO_SIGNAL",\n'
        '  "recommended_action": "BUY|SELL|HOLD|NO_ACTION",\n'
        '  "confidence": 0,\n'
        '  "time_horizon": "STRING",\n'
        '  "catalysts": ["STRING"],\n'
        '  "evidence": ["STRING"],\n'
        '  "contradictions": ["STRING"],\n'
        '  "reasoning": "STRING"\n'
        "}"
    )


def run_scan(universe: list[str], top_n: int, lookback_hours: int) -> pd.DataFrame:
    chunk_size = 30
    parts: list[pd.DataFrame] = []

    for i in range(0, len(universe), chunk_size):
        chunk = universe[i : i + chunk_size]
        chunk_metrics = _download_yahoo_daily(chunk, retries=3)
        if not chunk_metrics.empty:
            parts.append(chunk_metrics)
        time.sleep(0.1)

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
            time.sleep(0.05)
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
    candidate_pool = scored.sort_values("market_activity_score", ascending=False).head(max(top_n * 6, 80))

    news_features = fetch_news_features(candidate_pool["ticker"].tolist(), lookback_hours=lookback_hours)
    sec_targets = choose_sec_tickers(
        scored=scored,
        candidate_pool=candidate_pool,
        news_features=news_features,
        max_tickers=SEC_MAX_TICKERS_PER_SCAN,
    )
    sec_features = fetch_sec_features(sec_targets, lookback_days=SEC_LOOKBACK_DAYS_DEFAULT)
    rescored = apply_news_adjustments(scored, news_features, sec_features=sec_features)
    rescored = rescored.sort_values(["activity_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    rescored["selection_reason"] = rescored.apply(selection_reason, axis=1)
    rescored.attrs["missing_tickers"] = missing
    rescored.attrs["sec_queried_count"] = len(sec_targets)

    return rescored


@st.cache_data(ttl=120)
def get_single_ticker_activity(ticker: str, lookback_hours: int) -> pd.DataFrame:
    metrics = fetch_single_ticker_metric(ticker)
    if metrics.empty:
        return pd.DataFrame()
    scored = score_base_metrics(metrics)
    news_features = fetch_news_features([ticker], lookback_hours=lookback_hours)
    sec_features = fetch_sec_features([ticker], lookback_days=SEC_LOOKBACK_DAYS_DEFAULT)
    rescored = apply_news_adjustments(scored, news_features, sec_features=sec_features)
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
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = ""
    if "pending_all_stocks_search" not in st.session_state:
        st.session_state["pending_all_stocks_search"] = ""
    if "last_table_selected_ticker" not in st.session_state:
        st.session_state["last_table_selected_ticker"] = ""

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
                            "market_activity_score": float(r.get("market_activity_score", 50.0)),
                            "info_flow_score": float(r.get("info_flow_score", 50.0)),
                            "filing_flag": bool(r.get("filing_flag", False)),
                            "filing_count": int(r.get("filing_count", 0)),
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
        ranked = ensure_score_columns(st.session_state["last_ranked"])
        st.session_state["last_ranked"] = ranked
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

        st.subheader("All US Stocks (S&P 500)")
        pending_search_raw = str(st.session_state.get("pending_all_stocks_search", ""))
        if pending_search_raw == "__CLEAR__":
            st.session_state["all_stocks_search"] = ""
            st.session_state["pending_all_stocks_search"] = ""
        else:
            pending_search = pending_search_raw.strip().upper()
            if pending_search:
                st.session_state["all_stocks_search"] = pending_search
                st.session_state["pending_all_stocks_search"] = ""
        all_stocks_query = st.text_input(
            "Search all stocks (ticker or company name)",
            value="",
            key="all_stocks_search",
            placeholder="e.g., NVDA or NVIDIA",
        ).strip().upper()
        try:
            universe_df = get_sp500_universe_df()
            score_cols = ranked[
                [
                    "ticker",
                    "activity_score",
                    "market_activity_score",
                    "info_flow_score",
                    "filing_flag",
                    "filing_count",
                    "price_move_pct",
                    "volume_ratio",
                    "mention_count",
                    "source_diversity",
                ]
            ].copy()
            all_stocks = universe_df.merge(score_cols, on="ticker", how="left")
            all_stocks = all_stocks.sort_values(["company_name", "ticker"]).reset_index(drop=True)

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                only_movers = st.checkbox("Only movers", value=False, key="flt_only_movers")
            with f2:
                only_volume_spikes = st.checkbox("Only volume spikes", value=False, key="flt_only_volume_spikes")
            with f3:
                only_news_active = st.checkbox("Only news active", value=False, key="flt_only_news_active")
            with f4:
                only_sec_filings = st.checkbox("Only SEC filings", value=False, key="flt_only_sec_filings")

            if only_movers:
                all_stocks = all_stocks[pd.to_numeric(all_stocks["price_move_pct"], errors="coerce").abs() >= 2.0]
            if only_volume_spikes:
                all_stocks = all_stocks[pd.to_numeric(all_stocks["volume_ratio"], errors="coerce") >= 1.5]
            if only_news_active:
                mentions = pd.to_numeric(all_stocks["mention_count"], errors="coerce").fillna(0)
                sources = pd.to_numeric(all_stocks["source_diversity"], errors="coerce").fillna(0)
                all_stocks = all_stocks[(mentions > 0) | (sources > 0)]
            if only_sec_filings:
                all_stocks = all_stocks[all_stocks["filing_flag"] == True]

            all_stocks = rank_all_stocks_matches(all_stocks, all_stocks_query)

            display_cols = [
                "company_name",
                "ticker",
                "activity_score",
                "market_activity_score",
                "info_flow_score",
                "filing_flag",
                "filing_count",
            ]
            show_extra_cols = st.checkbox(
                "Show extra activity columns (price/volume/news)",
                value=True,
                key="all_stocks_show_extra_cols",
            )
            if show_extra_cols:
                display_cols.extend(["price_move_pct", "volume_ratio", "mention_count", "source_diversity"])
            table_df = all_stocks[display_cols].reset_index(drop=True)

            current_selected = str(st.session_state.get("selected_ticker", "")).upper()
            if len(table_df) == 1:
                auto_ticker = str(table_df.iloc[0]["ticker"]).upper()
                if auto_ticker and auto_ticker != current_selected:
                    st.session_state["selected_ticker"] = auto_ticker
                    st.session_state["last_table_selected_ticker"] = auto_ticker
                    st.session_state["pending_all_stocks_search"] = auto_ticker
                single_df = table_df.copy()
                single_df.insert(0, "select", [True])
                st.data_editor(
                    single_df,
                    use_container_width=True,
                    hide_index=True,
                    height="content",
                    key="all_stocks_editor_single",
                    column_config={"select": st.column_config.CheckboxColumn("Select")},
                    disabled=list(single_df.columns),
                )
            else:
                editor_df = table_df.copy()
                editor_df.insert(0, "select", editor_df["ticker"].astype(str).str.upper() == current_selected)
                edited_df = st.data_editor(
                    editor_df,
                    use_container_width=True,
                    hide_index=True,
                    height=320,
                    key="all_stocks_editor",
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                    },
                    disabled=[c for c in editor_df.columns if c != "select"],
                )

                checked = edited_df[edited_df["select"] == True]
                if not checked.empty:
                    picked_ticker = str(checked.iloc[0]["ticker"]).upper()
                    if picked_ticker != current_selected:
                        st.session_state["selected_ticker"] = picked_ticker
                        st.session_state["pending_all_stocks_search"] = picked_ticker
                        st.session_state["last_table_selected_ticker"] = picked_ticker
                        st.rerun()
                else:
                    if current_selected:
                        st.session_state["selected_ticker"] = ""
                        st.session_state["last_table_selected_ticker"] = ""
                        st.session_state["pending_all_stocks_search"] = "__CLEAR__"
                        st.rerun()

            if st.session_state.get("selected_ticker"):
                st.caption(f"Selected from table: {st.session_state['selected_ticker']}")
        except Exception as exc:
            st.warning(f"Unable to load S&P 500 name list: {exc}")

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.subheader("Top Picks")
            st.dataframe(
                top_df[
                    [
                        "ticker",
                        "activity_score",
                        "market_activity_score",
                        "info_flow_score",
                        "price_move_pct",
                        "volume_ratio",
                        "mention_count",
                        "source_diversity",
                        "filing_flag",
                        "filing_count",
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
        with st.expander("Score Components (Top Picks)"):
            st.dataframe(
                top_df[["ticker", "market_activity_score", "info_flow_score", "activity_score"]],
                use_container_width=True,
                hide_index=True,
                height="content",
            )

        st.subheader("Ticker Lookup")
        selected_ticker = str(st.session_state.get("selected_ticker", "")).upper()
        if selected_ticker:
            st.session_state["selected_ticker"] = selected_ticker
        selected_ticker_for_notes = selected_ticker

        selected_row = pd.DataFrame()
        if selected_ticker:
            from_last_scan = ranked[ranked["ticker"] == selected_ticker]
            if not from_last_scan.empty:
                selected_row = ensure_score_columns(from_last_scan.head(1).copy())
            else:
                with st.spinner(f"Fetching {selected_ticker} activity..."):
                    selected_row = ensure_score_columns(
                        get_single_ticker_activity(selected_ticker, lookback_hours=lookback_hours)
                    )

        if not selected_ticker:
            st.info("Select a ticker in 'All US Stocks (S&P 500)' to view activity score and details.")
        elif selected_row.empty:
            st.warning(f"No data available for {selected_ticker}.")
        else:
            st.caption(f"Showing ticker: {selected_ticker}")
            summary_cols = [
                "ticker",
                "activity_score",
                "market_activity_score",
                "info_flow_score",
                "price_move_pct",
                "volume_ratio",
                "mention_count",
                "source_diversity",
                "filing_flag",
                "filing_count",
                "selection_reason",
            ]
            st.dataframe(
                selected_row[summary_cols],
                use_container_width=True,
                hide_index=True,
                height="content",
            )

            with st.expander("Evidence preview", expanded=True):
                recent_news = get_ticker_news_items(selected_ticker, lookback_hours=lookback_hours, max_items=3)
                if not recent_news:
                    st.caption("No recent headlines available.")
                else:
                    for i, item in enumerate(recent_news[:3], start=1):
                        headline = item.get("headline", "")
                        source = item.get("source", "Unknown")
                        ts = item.get("time_utc", "")
                        snippet = item.get("snippet", "")
                        url = item.get("url", "")
                        if url:
                            st.markdown(f"{i}. [{headline}]({url})")
                        else:
                            st.markdown(f"{i}. {headline}")
                        if snippet:
                            st.caption(f"{source} • {ts} • {snippet[:220]}")
                        else:
                            st.caption(f"{source} • {ts}")

                latest_type = "-"
                latest_date = "-"
                if "filings" in selected_row.columns and not selected_row["filings"].empty:
                    filings = selected_row.iloc[0].get("filings", []) or []
                    if filings:
                        latest = filings[0]
                        latest_type = str(latest.get("form", "-"))
                        latest_date = str(latest.get("date", "-"))
                elif bool(selected_row.iloc[0].get("filing_flag", False)):
                    types = selected_row.iloc[0].get("filing_types", []) or []
                    dates = selected_row.iloc[0].get("filing_dates", []) or []
                    latest_type = str(types[0]) if types else "-"
                    latest_date = str(dates[0]) if dates else str(selected_row.iloc[0].get("latest_filing_time", "-"))
                st.caption(f"Most recent filing: {latest_type} on {latest_date}")

        st.subheader("Ticker Details")
        if not selected_ticker:
            st.info("Select a ticker in 'All US Stocks (S&P 500)' to load details.")
        else:
            try:
                timeframe = str(st.session_state.get("detail_timeframe", "1D"))
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
                    timeframe = st.radio(
                        "Timeframe",
                        options=["1D", "5D", "1M", "3M", "6M", "1Y"],
                        index=["1D", "5D", "1M", "3M", "6M", "1Y"].index(timeframe)
                        if timeframe in {"1D", "5D", "1M", "3M", "6M", "1Y"}
                        else 0,
                        horizontal=True,
                        key="detail_timeframe",
                    )
                    # Fetch again so chart reflects the current radio value while keeping control below price.
                    snap = get_ticker_detail_snapshot(selected_ticker, timeframe)

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

                    st.subheader("SEC Filings (recent)")
                    # Always do an on-demand SEC pull for the selected ticker so details are current,
                    # even if this ticker was outside the bounded SEC scan set.
                    sec_data = get_recent_sec_filings_for_ticker(
                        selected_ticker, lookback_days=SEC_LOOKBACK_DAYS_DEFAULT
                    )

                    if not sec_data.get("filing_flag", False):
                        st.caption(
                            f"No 8-K / 4 / 10-Q / 10-K filings in the last {SEC_LOOKBACK_DAYS_DEFAULT} days."
                        )
                    else:
                        st.caption(
                            f"Found {int(sec_data.get('filing_count', 0))} filing(s). "
                            f"Latest: {sec_data.get('latest_filing_time', '-')}"
                        )
                        types_text = ", ".join(sec_data.get("filing_types", []) or [])
                        if types_text:
                            st.write(f"Types: {types_text}")
                        links = sec_data.get("filing_links", []) or []
                        if links:
                            for idx, link in enumerate(links[:3], start=1):
                                st.markdown(f"- [SEC filing {idx}]({link})")

                    st.subheader("On-Demand Analysis")
                    analyze_clicked = st.button(
                        "Analyze this ticker now (paper research)",
                        key=f"analyze_ticker_btn_{selected_ticker}",
                    )
                    if analyze_clicked:
                        row_metrics = {}
                        if not selected_row.empty:
                            row0 = selected_row.iloc[0]
                            row_metrics = {
                                "move_pct_1d": float(row0.get("price_move_pct", 0.0) or 0.0),
                                "volume_ratio": float(row0.get("volume_ratio", 1.0) or 1.0),
                                "mention_count": int(row0.get("mention_count", 0) or 0),
                                "source_diversity": int(row0.get("source_diversity", 0) or 0),
                                "sentiment_shift": float(row0.get("sentiment_shift", 0.0) or 0.0),
                                "filing_flag": bool(row0.get("filing_flag", False)),
                                "filing_count": int(row0.get("filing_count", 0) or 0),
                            }
                        else:
                            row_metrics = {
                                "move_pct_1d": 0.0,
                                "volume_ratio": 1.0,
                                "mention_count": 0,
                                "source_diversity": 0,
                                "sentiment_shift": 0.0,
                                "filing_flag": bool(sec_data.get("filing_flag", False)),
                                "filing_count": int(sec_data.get("filing_count", 0) or 0),
                            }

                        news_bundle = get_ticker_news_items(
                            selected_ticker,
                            lookback_hours=lookback_hours,
                            max_items=15,
                        )
                        sec_bundle = {
                            "filing_flag": bool(sec_data.get("filing_flag", False)),
                            "filing_count": int(sec_data.get("filing_count", 0) or 0),
                            "filing_types": sec_data.get("filing_types", []) or [],
                            "filing_dates": sec_data.get("filing_dates", []) or [],
                            "latest_filing_time": str(sec_data.get("latest_filing_time", "") or ""),
                            "filing_links": (sec_data.get("filing_links", []) or [])[:3],
                            "filings": (sec_data.get("filings", []) or [])[:10],
                        }
                        evidence_bundle = {
                            "ticker": selected_ticker,
                            "as_of_utc": datetime.now(UTC).isoformat(),
                            "metrics_today": row_metrics,
                            "yahoo_news_items": news_bundle,
                            "sec_edgar_recent_filings": sec_bundle,
                        }
                        codex_prompt = build_codex_ticker_prompt(evidence_bundle)

                        st.markdown("Evidence bundle JSON")
                        st.code(json.dumps(evidence_bundle, ensure_ascii=True, indent=2), language="json")
                        st.markdown("Codex-ready prompt")
                        st.text_area(
                            "Copy this prompt into Codex",
                            value=codex_prompt,
                            height=360,
                            key=f"codex_prompt_{selected_ticker}",
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
