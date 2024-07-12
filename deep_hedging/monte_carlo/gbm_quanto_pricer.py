from typing import Callable

import numpy as np

from deep_hedging.monte_carlo import MonteCarloPricer
from deep_hedging.config import GlobalConfig


class GBMQuantoPricer(MonteCarloPricer):
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        random_seed: [int, None] = None,
    ):
        super().__init__(payoff_function, random_seed)

    def get_paths(
        self,
        spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], np.array],
        dividends_fn: Callable[[float], np.array],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ) -> np.array:
        days_till_maturity = int(round(GlobalConfig.TRADING_DAYS * time_till_maturity))

        if n_paths is None:
            n_paths = GlobalConfig.MONTE_CARLO_PATHS

        n_stocks = len(spot)

        time = np.linspace(0, time_till_maturity, days_till_maturity)
        d_time = time[1] - time[0]

        drift = []
        vol_scaling = []
        for t in time:
            var_covar = var_covar_fn(t)
            dividends = dividends_fn(t)
            dividends = np.concatenate([dividends, np.zeros(len(dividends))], axis=0)
            drift.append(
                [
                    (risk_free_rate_fn(t) - dividends - 0.5 * np.diag(var_covar))
                    * d_time
                ]
            )
            if len(spot) == 1:
                vol_scaling.append(np.sqrt(np.diag(var_covar)))
            else:
                vol_scaling.append(np.linalg.cholesky(var_covar))

        drift = np.array(drift).reshape(1, len(time), n_stocks, 1)
        vol_scaling = np.array(vol_scaling).reshape(1, len(time), n_stocks, n_stocks)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        diffusion = (
            vol_scaling
            @ np.random.normal(0, 1, size=(n_paths, len(time), n_stocks, 1))
            * np.sqrt(d_time)
        )
        paths = np.exp(drift + diffusion)
        paths = np.insert(paths, 0, np.array(spot).reshape(1, 1, -1, 1), axis=1)
        paths = np.cumprod(paths, axis=1).squeeze(3)

        base_paths = paths[:, :, :n_stocks // 2]
        modifying_paths = paths[:, :, n_stocks // 2:]

        return base_paths * modifying_paths
