"""
data_pipeline.py

Pulls OHLCV price history from the Schwab API via schwab-py and returns/saves
a clean pandas DataFrame. Designed to feed a Gymnasium trading environment.

Usage:
    from data_pipeline import fetch_daily_history, fetch_minute_history

    df = fetch_daily_history(client, "SPY", years=2)
    df.to_csv("spy_daily.csv")
"""

import pandas as pd
from datetime import datetime


def _candles_to_df(resp_json: dict) -> pd.DataFrame:
    """
    Convert a schwab-py price-history response body into a tidy DataFrame
    indexed by timestamp with columns: open, high, low, close, volume.
    """
    candles = resp_json.get("candles", [])
    if not candles:
        raise ValueError(
            f"No candles returned. Full response: {resp_json}"
        )

    df = pd.DataFrame(candles)
    # Schwab returns 'datetime' as epoch milliseconds
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_daily_history(client, symbol: str, years: int = 2) -> pd.DataFrame:
    """
    Pull daily OHLCV bars for `symbol` going back `years` years.
    Raises RuntimeError with the response body if the call fails.
    """
    resp = client.get_price_history_every_day(symbol)
    if resp.status_code != 200:
        raise RuntimeError(
            f"get_price_history_every_day failed ({resp.status_code}): {resp.text}"
        )
    df = _candles_to_df(resp.json())

    # Trim to the requested lookback window (Schwab's default range varies)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df.index >= cutoff]
    return df


def fetch_minute_history(client, symbol: str) -> pd.DataFrame:
    """
    Pull intraday 1-minute OHLCV bars for `symbol`.
    Note: Schwab typically only returns recent days of minute-level data
    (not years of history) — good for live/recent-session data, not
    long backtests.
    """
    resp = client.get_price_history_every_minute(symbol)
    if resp.status_code != 200:
        raise RuntimeError(
            f"get_price_history_every_minute failed ({resp.status_code}): {resp.text}"
        )
    return _candles_to_df(resp.json())


def get_latest_quote(client, symbol: str) -> dict:
    """
    Return the parsed quote dict for a single symbol (not just status code).
    """
    resp = client.get_quote(symbol)
    if resp.status_code != 200:
        raise RuntimeError(f"get_quote failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    # Schwab nests the quote under the symbol key
    return data.get(symbol, data)


if __name__ == "__main__":
    # Quick manual test — assumes you already built `c` via easy_client elsewhere
    # and are running this in the same auth'd session/venv.
    from schwab.auth import easy_client

 
    c = easy_client(
    api_key='RNz4Esx0WFVNTlGiP57GpJGO4XqLoAONtqC3dnrrG1MaXgJQ',
    app_secret = '5SNiAxva0QjrwnGbcN0LEsQn33TBBj4OmjmPTutx3yw09UPtDDCebtBSAAnn3JzS',
    callback_url='https://127.0.0.1:8182',
    token_path='/tmp/token.json')


    df = fetch_daily_history(c, "SPY", years=2)
    print(df.tail())
    df.to_csv("spy_daily.csv")
    print(f"Saved {len(df)} rows to spy_daily.csv")