from typing import Callable

import numpy as np

from deep_hedging.monte_carlo.bid_ask import BidAskSimulator

from deep_hedging.config.global_config import GlobalConfig


class WhiteNoiseBidAskSimulator(BidAskSimulator):
    def __init__(
        self,
        random_seed: [int, None] = None,
    ):
        super().__init__(random_seed=random_seed)

    def get_paths(
            self,
            bid_asks: list[float],
            time_till_maturity: float,
            var_covar_fn: Callable[[np.array], np.array],
            noise: np.array = None,
            n_paths: int = GlobalConfig.MONTE_CARLO_PATHS,
            *args,
            **kwargs
    ) -> np.array:

        days_till_maturity = int(round(GlobalConfig.TRADING_DAYS * time_till_maturity))

        if n_paths is None:
            n_paths = GlobalConfig.MONTE_CARLO_PATHS

        n_stocks = len(bid_asks)

        time = np.linspace(0, time_till_maturity, days_till_maturity)
        d_time = time[1] - time[0]

        var_covar = var_covar_fn(time)
        if n_stocks == 1:
            vols = var_covar[:, np.newaxis]
        else:
            vols = np.diagonal(var_covar, axis1=1, axis2=2)

        if n_stocks == 1:
            vol_scaling = np.sqrt(vols)
        else:
            vol_scaling = np.linalg.cholesky(var_covar)
        vol_scaling = np.array(vol_scaling).reshape(1, len(time), n_stocks, n_stocks)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if noise is None:
            noise = np.random.normal(0, 1, size=(n_paths, len(time), n_stocks, 1))

        paths = vol_scaling @ noise * np.sqrt(d_time)
        paths = np.where(paths > 0, paths, 0)
        paths = np.insert(paths, 0, np.array(bid_asks).reshape(1, 1, -1, 1), axis=1)
        paths = np.cumsum(paths, axis=1).squeeze(3)

        return paths
