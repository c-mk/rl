"""
train_walkforward.py

Walk-forward training + out-of-sample evaluation, two folds:

  Fold 1: train on data BEFORE 2020-01-01, test on 2020-01-01 to 2022-01-01
  Fold 2: train on data BEFORE 2022-01-01, test on 2022-01-01 to 2024-01-01

This is the standard walk-forward setup: each fold's test window is data
the agent never saw during that fold's training, and fold 2's training
set grows to include fold 1's test period (as if you retrained your
strategy once new data became available). This avoids the lookahead bias
you'd get from a single random train/test split on time-series data.

For each fold, we:
  1. Train a PPO agent on the training window (reward = differential Sharpe)
  2. Run the trained agent through the test window with a deterministic
     (greedy) policy
  3. Compute the ACTUAL Sharpe ratio on the test window's realized returns
     — this, not the training reward, is the number that matters

Usage:
    python train_walkforward.py --csv spy_daily_2015_2024.csv
"""

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from trading_env import TradingEnv
from sharpe_utils import compute_sharpe


def make_env_fn(df):
    return lambda: TradingEnv(df, window_size=10)


def run_fold(train_df, test_df, fold_name, timesteps=50_000, seed=0):
    print(f"\n{'='*60}\n{fold_name}\n{'='*60}")
    print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")

    # --- Train ---
    train_env = make_vec_env(make_env_fn(train_df), n_envs=1, seed=seed)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)

    # --- Evaluate out-of-sample on the test window (greedy/deterministic) ---
    test_env = TradingEnv(test_df, window_size=10)
    obs, info = test_env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))

    test_sharpe = compute_sharpe(test_env.strategy_returns)
    total_return = test_env.portfolio_value / test_env.initial_balance - 1

    print(f"\n--- {fold_name} OUT-OF-SAMPLE RESULTS ---")
    print(f"Sharpe ratio:   {test_sharpe:.3f}")
    print(f"Total return:   {total_return*100:.2f}%")
    print(f"Final value:    ${test_env.portfolio_value:,.2f}")

    return {
        "fold": fold_name,
        "sharpe": test_sharpe,
        "total_return": total_return,
        "final_value": test_env.portfolio_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to daily OHLCV CSV (from data_pipeline.py)")
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    df = df.sort_index()

    fold1_train = df[df.index < "2020-01-01"]
    fold1_test = df[(df.index >= "2020-01-01") & (df.index < "2022-01-01")]

    fold2_train = df[df.index < "2022-01-01"]
    fold2_test = df[(df.index >= "2022-01-01") & (df.index < "2024-01-01")]

    for name, d in [("Fold 1 train", fold1_train), ("Fold 1 test", fold1_test),
                    ("Fold 2 train", fold2_train), ("Fold 2 test", fold2_test)]:
        if len(d) < 30:
            raise ValueError(
                f"{name} has only {len(d)} rows — need more history in your CSV. "
                f"Re-run data_pipeline.py with an earlier start date."
            )

    results = []
    results.append(run_fold(fold1_train, fold1_test, "FOLD 1 (train <2020, test 2020-2022)", args.timesteps))
    results.append(run_fold(fold2_train, fold2_test, "FOLD 2 (train <2022, test 2022-2024)", args.timesteps))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        print(f"{r['fold']}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']*100:.2f}%")
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    print(f"\nAverage out-of-sample Sharpe across folds: {avg_sharpe:.3f}")


if __name__ == "__main__":
    main()