# SignalWeave Adhoc Scanner

One-click, adhoc US equity scanner with built-in ranking, reason tags, and performance tracking.

## Run

```bash
cd /Users/varmakammili/Documents/GitHub/SignalWeave
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run /Users/varmakammili/Documents/GitHub/SignalWeave/streamlit_app.py
```

## What it does

- Runs an on-demand scan for S&P 500 (or custom tickers).
- Computes activity scores from price move, relative volume, headline flow, source diversity, filing hints, and sentiment shift.
- Shows top picks with a direct selection reason.
- Saves each run and lets you refresh how that run's picks performed since selection.

## Data notes

- Price/volume and headlines are fetched at runtime via Yahoo Finance.
- Filing flag is inferred from filing-related headline keywords.
