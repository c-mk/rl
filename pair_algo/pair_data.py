"""
pair_data.py

Fetches aligned daily history for two symbols (e.g. SPY, QQQ) and merges
them into a single DataFrame indexed by date, keeping only dates both
symbols traded (inner join — handles any holiday/listing mismatches).
"""

import pandas as pd
from data_pipeline import fetch_daily_history


def fetch_pair_history(client, symbol_a: str, symbol_b: str, start, end) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: close_a, close_b (renamed from the
    two symbols), aligned on shared trading dates.
    """
    df_a = fetch_daily_history(client, symbol_a, start=start, end=end)[["close"]]
    df_b = fetch_daily_history(client, symbol_b, start=start, end=end)[["close"]]

    df_a.columns = [f"close_{symbol_a.lower()}"]
    df_b.columns = [f"close_{symbol_b.lower()}"]

    merged = df_a.join(df_b, how="inner").dropna()
    return merged


if __name__ == "__main__":
    from datetime import datetime
    from schwab.auth import easy_client
    
    c = easy_client(
    api_key='RNz4Esx0WFVNTlGiP57GpJGO4XqLoAONtqC3dnrrG1MaXgJQ',
    app_secret = '5SNiAxva0QjrwnGbcN0LEsQn33TBBj4OmjmPTutx3yw09UPtDDCebtBSAAnn3JzS',
    callback_url='https://127.0.0.1:8182',
    token_path='/tmp/token.json')

    df = fetch_pair_history(c, "SPY", "QQQ", datetime(2015, 1, 1), datetime(2024, 1, 1))
    df.to_csv("spy_qqq_2015_2024.csv")
    print(f"Saved {len(df)} aligned rows to spy_qqq_2015_2024.csv")
    print(df.head())