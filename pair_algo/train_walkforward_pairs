"""
train_walkforward_pairs.py

Same two-fold walk-forward structure as train_walkforward.py, adapted for
the SPY/QQQ pairs environment:

  Fold 1: train <2020,  test 2020-2022
  Fold 2: train <2022,  test 2022-2024

Critically: the hedge ratio (beta) is re-estimated from scratch on EACH
fold's training window only, then held fixed while evaluating that fold's
test window. This mirrors how you'd actually deploy this — periodically
re-fit the hedge ratio on trailing data, then trade forward with it fixed
until the next re-fit.

Usage:
    python train_walkforward_pairs.py --csv spy_qqq_2015_2024.csv
"""

import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pair_env import PairTradingEnv, estimate_hedge_ratio
from sharpe_utils import compute_sharpe

COL_A = "close_spy"
COL_B = "close_qqq"


def make_env_fn(df, beta):
    return lambda: PairTradingEnv(df, COL_A, COL_B, hedge_ratio=beta, window_size=20)


def run_fold(train_df, test_df, fold_name, timesteps=50_000, seed=0):
    print(f"\n{'='*60}\n{fold_name}\n{'='*60}")
    print(f"Train: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows)")
    print(f"Test:  {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")

    beta = estimate_hedge_ratio(train_df, COL_A, COL_B)
    print(f"Hedge ratio (beta), fit on train only: {beta:.4f}")

    train_env = make_vec_env(make_env_fn(train_df, beta), n_envs=1, seed=seed)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)

    test_env = PairTradingEnv(test_df, COL_A, COL_B, hedge_ratio=beta, window_size=20)
    obs, info = test_env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(int(action))

    test_sharpe = compute_sharpe(test_env.strategy_returns)
    total_return = test_env.portfolio_value / 10_000.0 - 1

    print(f"\n--- {fold_name} OUT-OF-SAMPLE RESULTS ---")
    print(f"Hedge ratio used: {beta:.4f}")
    print(f"Sharpe ratio:     {test_sharpe:.3f}")
    print(f"Total return:     {total_return*100:.2f}%")
    print(f"Final value:      ${test_env.portfolio_value:,.2f}")

    return {"fold": fold_name, "sharpe": test_sharpe, "total_return": total_return, "beta": beta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to aligned SPY/QQQ CSV (from pair_data.py)")
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()

    fold1_train = df[df.index < "2020-01-01"]
    fold1_test = df[(df.index >= "2020-01-01") & (df.index < "2022-01-01")]
    fold2_train = df[df.index < "2022-01-01"]
    fold2_test = df[(df.index >= "2022-01-01") & (df.index < "2024-01-01")]

    for name, d in [("Fold 1 train", fold1_train), ("Fold 1 test", fold1_test),
                    ("Fold 2 train", fold2_train), ("Fold 2 test", fold2_test)]:
        if len(d) < 60:
            raise ValueError(f"{name} has only {len(d)} rows — need more history in your CSV.")

    results = []
    results.append(run_fold(fold1_train, fold1_test, "FOLD 1 (train <2020, test 2020-2022)", args.timesteps))
    results.append(run_fold(fold2_train, fold2_test, "FOLD 2 (train <2022, test 2022-2024)", args.timesteps))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in results:
        print(f"{r['fold']}: Sharpe={r['sharpe']:.3f}, Return={r['total_return']*100:.2f}%, beta={r['beta']:.3f}")
    print(f"\nAverage out-of-sample Sharpe: {np.mean([r['sharpe'] for r in results]):.3f}")


if __name__ == "__main__":
    main()