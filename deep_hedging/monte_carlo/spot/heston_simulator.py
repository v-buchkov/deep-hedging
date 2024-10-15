from typing import Callable

import numpy as np

from deep_hedging.monte_carlo.spot.monte_carlo_simulator import MonteCarloSimulator

from deep_hedging.config import GlobalConfig


class HestonSimulator(MonteCarloSimulator):
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        random_seed: [int, None] = None,
    ):
        super().__init__(payoff_function, random_seed)

    def get_paths(
        self,
        current_spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
        random_seed: [int, None] = None,
    ) -> np.array:
        days_till_maturity = int(round(GlobalConfig.TRADING_DAYS * time_till_maturity))

        if n_paths is None:
            n_paths = GlobalConfig.MONTE_CARLO_PATHS

        t = days_till_maturity
        n_stocks = len(current_spot)

        time = np.linspace(0, t / GlobalConfig.TRADING_DAYS, t)
        d_time = time[1] - time[0]

        drift = []
        cholesky = []
        for t in time:
            var_covar = var_covar_fn(t)
            drift.append(
                [
                    (
                        risk_free_rate_fn(t)
                        - dividends_fn(t)
                        - 0.5 * np.diag(var_covar) ** 2
                    )
                    * d_time
                ]
            )
            cholesky.append(np.linalg.cholesky(var_covar))

        drift = np.array(drift).reshape(1, len(time), n_stocks, 1)
        cholesky = np.array(cholesky).reshape(1, len(time), n_stocks, n_stocks)

        np.random.seed(random_seed)
        diffusion = (
            cholesky
            @ np.random.normal(0, 1, size=(n_paths, len(time), n_stocks, 1))
            * np.sqrt(d_time)
        )
        paths = np.exp(drift + diffusion)
        paths = np.insert(paths, 0, np.array(current_spot).reshape(1, 1, -1, 1), axis=1)
        paths = np.cumprod(paths, axis=1).squeeze(3)

        return paths
