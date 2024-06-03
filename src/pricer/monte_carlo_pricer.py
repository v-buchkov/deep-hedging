from typing import Callable

import numpy as np

RANDOM_SEED = 12


class MonteCarloPricer:
    PATHS = 20_000
    TRADING_DAYS: int = 252

    def __init__(
            self,
            payoff_function: Callable[[np.array], float],
            random_seed: [int, None] = RANDOM_SEED
    ):
        self.payoff_function = payoff_function
        self.random_seed = random_seed

    def _geometric_brownian_motion(
            self,
            current_spot: list[float],
            days_till_maturity: int,
            risk_free_rate_fn: Callable[[float], float],
            dividends_fn: Callable[[float], float],
            var_covar_fn: Callable[[float], np.array],
            n_paths: [int, None] = None,
            *args,
            **kwargs
    ) -> np.array:

        if n_paths is None:
            n_paths = self.PATHS

        t = days_till_maturity
        n_stocks = len(current_spot)

        time = np.linspace(0, t / self.TRADING_DAYS, t)
        d_time = time[1] - time[0]

        drift = []
        cholesky = []
        for t in time:
            var_covar = var_covar_fn(t)
            drift.append([(risk_free_rate_fn(t) - dividends_fn(t) - 0.5 * np.diag(var_covar) ** 2) * d_time])
            cholesky.append(np.linalg.cholesky(var_covar))

        drift = np.array(drift).reshape(1, len(time), n_stocks, 1)
        cholesky = np.array(cholesky).reshape(1, len(time), n_stocks, n_stocks)

        np.random.seed(self.random_seed)
        diffusion = cholesky @ np.random.normal(0, 1, size=(n_paths, len(time), n_stocks, 1)) * np.sqrt(d_time)
        paths = np.exp(drift + diffusion)
        paths = np.insert(paths, 0, np.array(current_spot).reshape(1, 1, -1, 1), axis=1)
        paths = np.cumprod(paths, axis=1).squeeze(3)

        return paths

    def get_paths(
            self,
            current_spot: list[float],
            time_till_maturity: float,
            risk_free_rate_fn: Callable[[float], float],
            dividends_fn: Callable[[float], float],
            var_covar_fn: Callable[[float], np.array],
            n_paths: [int, None] = None
    ) -> np.array:
        return self._geometric_brownian_motion(
            current_spot=current_spot,
            days_till_maturity=int(round(self.TRADING_DAYS * time_till_maturity)),
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths
        )

    def get_future_value(
            self,
            current_spot: list[float],
            time_till_maturity: float,
            risk_free_rate_fn: Callable[[float], float],
            dividends_fn: Callable[[float], float],
            var_covar_fn: Callable[[float], np.array],
            n_paths: [int, None] = None
    ) -> float:
        random_paths = self._geometric_brownian_motion(
            current_spot=current_spot,
            days_till_maturity=int(round(self.TRADING_DAYS * time_till_maturity)),
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths
        )

        instrument_payoffs = self.payoff_function(random_paths)

        return np.mean(instrument_payoffs)
