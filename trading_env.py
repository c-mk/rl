"""
trading_env.py

A minimal Gymnasium environment for single-symbol trading over a historical
OHLCV DataFrame (e.g. produced by data_pipeline.fetch_daily_history).

This is intentionally simple — a starting skeleton, not a finished strategy:
  - Discrete action space: 0 = hold, 1 = buy (go long), 2 = sell (flat/short-close)
  - Observation: a rolling window of past returns + current position
  - Reward: change in portfolio value each step, minus a transaction cost
    when a trade is made

Usage:
    import pandas as pd
    from trading_env import TradingEnv

    df = pd.read_csv("spy_daily.csv", index_col=0, parse_dates=True)
    env = TradingEnv(df, window_size=10)

    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.0005,  # 5 bps per trade, adjust as needed
    ):
        super().__init__()
        assert "close" in df.columns, "DataFrame must have a 'close' column"

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Precompute simple returns for the observation window
        self.returns = self.df["close"].pct_change().fillna(0.0).values

        # Actions: 0 = hold, 1 = go long (buy), 2 = go flat (sell)
        self.action_space = spaces.Discrete(3)

        # Observation: window of past returns + current position (0 or 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size + 1,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0 = flat, 1 = long
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance

    def _get_obs(self) -> np.ndarray:
        window = self.returns[self.current_step - self.window_size : self.current_step]
        obs = np.append(window, float(self.position)).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "price": self.df["close"].iloc[self.current_step],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        price = self.df["close"].iloc[self.current_step]
        reward = 0.0

        # Execute action
        if action == 1 and self.position == 0:  # open long
            self.position = 1
            self.entry_price = price
            reward -= self.transaction_cost * self.portfolio_value
        elif action == 2 and self.position == 1:  # close long
            pnl = (price - self.entry_price) / self.entry_price
            self.portfolio_value *= (1 + pnl)
            reward += pnl * self.portfolio_value
            reward -= self.transaction_cost * self.portfolio_value
            self.position = 0
            self.entry_price = 0.0
        # action == 0 (hold) or invalid transitions: no-op

        # Mark-to-market unrealized PnL each step while holding
        if self.position == 1:
            next_price = (
                self.df["close"].iloc[self.current_step + 1]
                if self.current_step + 1 < len(self.df)
                else price
            )
            step_return = (next_price - price) / price
            reward += step_return * self.portfolio_value

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        info = self._get_info()
        print(
            f"Step {info['step']:>5} | Price {info['price']:.2f} | "
            f"Position {info['position']} | Portfolio ${info['portfolio_value']:.2f}"
        )


if __name__ == "__main__":
    # Smoke test with synthetic data so this runs without any API calls
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    prices = 400 + np.cumsum(np.random.randn(300))
    df = pd.DataFrame({"close": prices}, index=dates)

    env = TradingEnv(df, window_size=10)
    obs, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")