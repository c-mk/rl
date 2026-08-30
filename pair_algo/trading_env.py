"""
trading_env.py

Gymnasium environment for single-symbol trading, trained to maximize
risk-adjusted return (Sharpe) rather than raw PnL.

Key difference from a plain PnL-reward env: at each step, the strategy's
realized return for that step is fed into a DifferentialSharpe tracker,
and the AGENT'S REWARD is the differential Sharpe ratio value — not the
dollar PnL. This trains the policy to prefer steady, low-variance gains
over volatile/lucky ones, which is what Sharpe rewards.

The env also logs the raw per-step strategy returns into
`self.strategy_returns`, so you can compute the actual (non-differential)
Sharpe ratio over a full episode afterward with sharpe_utils.compute_sharpe
— that's your real evaluation metric; the DSR is only the training signal.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from sharpe_utils import DifferentialSharpe


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_balance: float = 10_000.0,
        transaction_cost: float = 0.0005,
        dsr_eta: float = 0.01,
    ):
        super().__init__()
        assert "close" in df.columns, "DataFrame must have a 'close' column"

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.dsr_eta = dsr_eta

        self.price_returns = self.df["close"].pct_change().fillna(0.0).values

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
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
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.dsr = DifferentialSharpe(eta=self.dsr_eta)
        self.strategy_returns = []  # raw per-step returns, for post-hoc Sharpe

    def _get_obs(self) -> np.ndarray:
        window = self.price_returns[
            self.current_step - self.window_size : self.current_step
        ]
        return np.append(window, float(self.position)).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
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
        step_pnl_return = 0.0  # this step's realized strategy return

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            step_pnl_return -= self.transaction_cost

        elif action == 2 and self.position == 1:
            trade_return = (price - self.entry_price) / self.entry_price
            step_pnl_return += trade_return - self.transaction_cost
            self.position = 0
            self.entry_price = 0.0

        # Mark-to-market while holding
        if self.position == 1 and self.current_step + 1 < len(self.df):
            next_price = self.df["close"].iloc[self.current_step + 1]
            step_pnl_return += (next_price - price) / price

        # Update portfolio value and record the raw return for Sharpe eval
        self.portfolio_value *= (1 + step_pnl_return)
        self.strategy_returns.append(step_pnl_return)

        # Reward = differential Sharpe ratio of this step's return,
        # NOT the raw return itself. This is what points the agent
        # toward risk-adjusted performance.
        reward = self.dsr.update(step_pnl_return)

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
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    prices = 400 + np.cumsum(np.random.randn(300))
    df = pd.DataFrame({"close": prices}, index=dates)

    env = TradingEnv(df, window_size=10)
    obs, info = env.reset()
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    from sharpe_utils import compute_sharpe
    sharpe = compute_sharpe(env.strategy_returns)
    print(f"Episode Sharpe (random policy, synthetic data): {sharpe:.3f}")
    print(f"Final portfolio value: ${info['portfolio_value']:.2f}")