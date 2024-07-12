from typing import Callable

import numpy as np

from deep_hedging.monte_carlo import MonteCarloPricer
from deep_hedging.underlyings import Underlyings
from deep_hedging.config import GlobalConfig


class GBMQuantoPricer(MonteCarloPricer):
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        base_underlyings: Underlyings,
        modifying_underlyings: Underlyings,
        random_seed: [int, None] = None,
    ):
        super().__init__(payoff_function, random_seed)
        self.base_underlyings = base_underlyings
        self.modifying_underlyings = modifying_underlyings

    def get_paths(
        self,
        spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[str, float], np.array],
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
        base_ccys = [ticker.currency for ticker in self.base_underlyings.tickers]
        fx_ccys = [ticker.currency for ticker in self.modifying_underlyings.tickers]
        for t in time:
            rf_rates = [risk_free_rate_fn(t, ccy) for ccy in base_ccys]
            rf_rates += [-risk_free_rate_fn(t, ccy) for ccy in fx_ccys]

            var_covar = var_covar_fn(t)

            dividends = dividends_fn(t)
            if len(dividends) < var_covar.shape[0]:
                dividends = np.concatenate(
                    [dividends, np.zeros(var_covar.shape[0] - len(dividends))], axis=0
                )

            drift.append([(rf_rates - dividends - 0.5 * np.diag(var_covar)) * d_time])

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

        base_paths = paths[:, :, : len(self.base_underlyings)]
        modifying_paths = {
            ticker.currency: paths[:, :, len(self.base_underlyings) + i]
            for i, ticker in enumerate(self.modifying_underlyings.tickers)
        }

        for i, ticker in enumerate(self.base_underlyings.tickers):
            base_paths[:, :, i] *= modifying_paths[ticker.currency]

        return base_paths
