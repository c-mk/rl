"""
pair_env.py

Pairs-trading environment for two cointegrated-ish instruments (e.g. SPY/QQQ).

State variable: rolling z-score of the log-price spread
    spread_t = log(price_a_t) - beta * log(price_b_t)
where beta (the hedge ratio) is estimated via OLS regression of
log(price_a) on log(price_b) — and MUST be fit only on training data,
then held fixed and applied to test data (passed in as a constructor arg),
otherwise you leak future information into the backtest.

Actions:
    0 = hold
    1 = enter LONG spread  (long A, short B) — bet the spread will rise
    2 = enter SHORT spread (short A, long B) — bet the spread will fall
    3 = flatten (close whatever position is open)

Reward: differential Sharpe ratio of the pair's daily return, same
mechanism as the single-symbol trading_env.py.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from sharpe_utils import DifferentialSharpe


def estimate_hedge_ratio(train_df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """
    OLS hedge ratio: log(price_a) = alpha + beta * log(price_b) + eps
    Fit ONLY on training data, then reused as a fixed constant on test data.
    """
    log_a = np.log(train_df[col_a].values)
    log_b = np.log(train_df[col_b].values)
    beta, alpha = np.polyfit(log_b, log_a, deg=1)
    return float(beta)


class PairTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        hedge_ratio: float,
        window_size: int = 20,
        transaction_cost: float = 0.0005,
        dsr_eta: float = 0.01,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.col_a = col_a
        self.col_b = col_b
        self.hedge_ratio = hedge_ratio
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.dsr_eta = dsr_eta

        log_a = np.log(self.df[col_a].values)
        log_b = np.log(self.df[col_b].values)
        self.spread = log_a - hedge_ratio * log_b

        # Rolling z-score of the spread — the core state variable
        spread_series = pd.Series(self.spread)
        roll_mean = spread_series.rolling(window_size).mean()
        roll_std = spread_series.rolling(window_size).std()
        self.zscore = ((spread_series - roll_mean) / roll_std).fillna(0.0).values

        # Per-step simple returns of each leg (for computing pair PnL)
        self.returns_a = self.df[col_a].pct_change().fillna(0.0).values
        self.returns_b = self.df[col_b].pct_change().fillna(0.0).values

        self.action_space = spaces.Discrete(4)  # hold, long spread, short spread, flatten
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size + 1,),  # window of z-scores + current position
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.position = 0  # -1 short spread, 0 flat, 1 long spread
        self.portfolio_value = 10_000.0
        self.dsr = DifferentialSharpe(eta=self.dsr_eta)
        self.strategy_returns = []

    def _get_obs(self) -> np.ndarray:
        window = self.zscore[self.current_step - self.window_size : self.current_step]
        return np.append(window, float(self.position)).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self.current_step,
            "position": self.position,
            "zscore": self.zscore[self.current_step],
            "portfolio_value": self.portfolio_value,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        step_return = 0.0
        cost = 0.0

        if action == 1 and self.position != 1:      # enter/flip to long spread
            cost = self.transaction_cost * 2 * abs(1 - self.position)  # two legs
            self.position = 1
        elif action == 2 and self.position != -1:    # enter/flip to short spread
            cost = self.transaction_cost * 2 * abs(-1 - self.position)
            self.position = -1
        elif action == 3 and self.position != 0:     # flatten
            cost = self.transaction_cost * 2 * abs(self.position)
            self.position = 0
        # action == 0 (hold): no change

        # Next-day leg returns realize this step's pair PnL if holding a position
        if self.current_step + 1 < len(self.df) and self.position != 0:
            r_a = self.returns_a[self.current_step + 1]
            r_b = self.returns_b[self.current_step + 1]
            # long spread = long A, short B -> profits when A outperforms B
            step_return = self.position * (r_a - self.hedge_ratio * r_b)

        step_return -= cost
        self.portfolio_value *= (1 + step_return)
        self.strategy_returns.append(step_return)

        reward = self.dsr.update(step_return)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        info = self._get_info()
        print(
            f"Step {info['step']:>5} | Z {info['zscore']:.2f} | "
            f"Position {info['position']:>2} | Portfolio ${info['portfolio_value']:.2f}"
        )


if __name__ == "__main__":
    # Smoke test with two synthetic, mildly cointegrated series
    np.random.seed(0)
    n = 500
    common_factor = np.cumsum(np.random.randn(n) * 0.01)
    noise_a = np.cumsum(np.random.randn(n) * 0.002)
    noise_b = np.cumsum(np.random.randn(n) * 0.002)

    price_a = 400 * np.exp(common_factor + noise_a)
    price_b = 350 * np.exp(common_factor + noise_b)

    df = pd.DataFrame({"close_spy": price_a, "close_qqq": price_b})
    beta = estimate_hedge_ratio(df.iloc[:300], "close_spy", "close_qqq")
    print(f"Estimated hedge ratio (beta): {beta:.3f}")

    env = PairTradingEnv(df, "close_spy", "close_qqq", hedge_ratio=beta, window_size=20)
    obs, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    from sharpe_utils import compute_sharpe
    sharpe = compute_sharpe(env.strategy_returns)
    print(f"Episode Sharpe (random policy, synthetic pair): {sharpe:.3f}")
    print(f"Final portfolio value: ${env.portfolio_value:,.2f}")