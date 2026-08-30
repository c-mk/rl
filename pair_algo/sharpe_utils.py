"""
sharpe_utils.py

Two things live here:

1. compute_sharpe() — the standard annualized Sharpe ratio, used to EVALUATE
   a strategy after the fact on a return series (e.g. an out-of-sample test
   fold). This is your reporting metric.

2. DifferentialSharpe — a running, step-wise approximation of the Sharpe
   ratio's marginal change (Moody & Saffell, 1998). This is fed to the RL
   agent as its REWARD at each step, so the agent is trained to directly
   improve risk-adjusted return rather than raw PnL.

Why two different things? Sharpe ratio itself is only defined over a window
of returns (mean/std need a batch). An RL agent needs a reward every single
step. The differential Sharpe ratio is the standard trick for turning Sharpe
into a step-wise signal: it's the derivative of the Sharpe ratio with respect
to adding one more return to a running average, so maximizing the sum of
per-step DSR values is approximately equivalent to maximizing the final
Sharpe ratio.
"""

import numpy as np


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """
    Standard annualized Sharpe ratio over a series of period returns.

    returns: array of per-period simple returns (e.g. daily returns)
    periods_per_year: 252 for daily equity data
    risk_free: per-period risk-free rate to subtract (default 0 for simplicity)

    Returns 0.0 if there's no variance (avoids div-by-zero on flat/empty series).
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free
    mean = excess.mean()
    std = excess.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float((mean / std) * np.sqrt(periods_per_year))


class DifferentialSharpe:
    """
    Running differential Sharpe ratio, used as a step-wise RL reward.

    Maintains exponentially-weighted running estimates of the first and
    second moment of returns (A_t, B_t), and on each new return computes
    the marginal contribution to the Sharpe ratio.

    eta controls the decay rate of the running averages — smaller eta means
    longer memory (slower-adapting Sharpe estimate), larger eta adapts
    faster to recent returns. 0.01-0.05 is a reasonable starting range for
    daily bars.
    """

    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0  # running estimate of E[return]
        self.B = 0.0  # running estimate of E[return^2]
        self._initialized = False

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self._initialized = False

    def update(self, r: float) -> float:
        """
        Feed in this step's return `r`, get back the differential Sharpe
        ratio reward for this step.
        """
        if not self._initialized:
            # First observation just seeds the running stats; no meaningful
            # reward signal yet.
            self.A = r
            self.B = r ** 2
            self._initialized = True
            return 0.0

        delta_A = r - self.A
        delta_B = r ** 2 - self.B

        denom = (self.B - self.A ** 2) ** 1.5
        if denom <= 1e-8 or np.isnan(denom):
            dsr = 0.0
        else:
            # Moody & Saffell (1998), eq. for D_t
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / denom

        # Update running moments AFTER computing the reward from the
        # pre-update values (this is what makes it "differential")
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return float(dsr)