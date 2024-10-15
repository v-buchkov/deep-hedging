from typing import Callable

import numpy as np

from deep_hedging.monte_carlo.spot.monte_carlo_simulator import MonteCarloSimulator

from deep_hedging.config import GlobalConfig


class GBMSimulator(MonteCarloSimulator):
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
        risk_free_rate_fn: Callable[[np.array], np.array],
        dividends_fn: Callable[[np.array], np.array],
        var_covar_fn: Callable[[np.array], np.array],
        n_paths: [int, None] = None,
    ) -> np.array:
        days_till_maturity = int(round(GlobalConfig.TRADING_DAYS * time_till_maturity))

        if n_paths is None:
            n_paths = GlobalConfig.MONTE_CARLO_PATHS

        n_stocks = len(spot)

        time = np.linspace(0, time_till_maturity, days_till_maturity)
        d_time = time[1] - time[0]

        var_covar = var_covar_fn(time)
        if n_stocks == 1:
            vols = var_covar[:, np.newaxis]
        else:
            vols = np.diagonal(var_covar, axis1=1, axis2=2)
        drift = (risk_free_rate_fn(time) - dividends_fn(time) - 0.5 * vols) * d_time
        drift = np.array(drift).reshape(1, len(time), n_stocks, 1)

        if n_stocks == 1:
            vol_scaling = np.sqrt(vols)
        else:
            vol_scaling = np.linalg.cholesky(var_covar)
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

        return paths
