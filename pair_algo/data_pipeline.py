"""
data_pipeline.py

Pulls OHLCV price history from the Schwab API via schwab-py into a clean
pandas DataFrame, with explicit start/end date support so you can carve
out walk-forward train/test windows.
"""

import pandas as pd
from datetime import datetime


def _candles_to_df(resp_json: dict) -> pd.DataFrame:
    """Convert a schwab-py price-history response body into a tidy DataFrame."""
    candles = resp_json.get("candles", [])
    if not candles:
        raise ValueError(f"No candles returned. Full response: {resp_json}")

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_daily_history(
    client,
    symbol: str,
    start: datetime = None,
    end: datetime = None,
    years: int = None,
) -> pd.DataFrame:
    """
    Pull daily OHLCV bars for `symbol`.

    Pass either:
      - start/end as datetime objects for an explicit range (needed for
        walk-forward splits), or
      - years=N to just grab the last N years (quick/simple case).
    """
    if start is not None and end is not None:
        resp = client.get_price_history_every_day(
            symbol, start_datetime=start, end_datetime=end
        )
    else:
        resp = client.get_price_history_every_day(symbol)

    if resp.status_code != 200:
        raise RuntimeError(
            f"get_price_history_every_day failed ({resp.status_code}): {resp.text}"
        )
    df = _candles_to_df(resp.json())

    if years is not None and start is None:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df[df.index >= cutoff]

    return df


def get_latest_quote(client, symbol: str) -> dict:
    resp = client.get_quote(symbol)
    if resp.status_code != 200:
        raise RuntimeError(f"get_quote failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data.get(symbol, data)


if __name__ == "__main__":
    from schwab.auth import easy_client

    c = easy_client(
        api_key="YOUR_KEY",
        app_secret="YOUR_SECRET",
        callback_url="https://127.0.0.1:8182",
        token_path="/tmp/token.json",
    )

    # Pull the full 2015-2024 span once; walk-forward slicing happens later
    # in train_walkforward.py from this single cached file.
    df = fetch_daily_history(
        c, "SPY", start=datetime(2015, 1, 1), end=datetime(2024, 1, 1)
    )
    df.to_csv("spy_daily_2015_2024.csv")
    print(f"Saved {len(df)} rows to spy_daily_2015_2024.csv")