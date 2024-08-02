from typing import Callable

import numpy as np

from deep_hedging.monte_carlo.spot.monte_carlo_simulator import MonteCarloSimulator
from deep_hedging.monte_carlo.rates.interest_rate_simulator import InterestRateSimulator
from deep_hedging.monte_carlo.bid_ask.bid_ask_simulator import BidAskSimulator
from deep_hedging.monte_carlo.volatility.volatility_simulator import VolatilitySimulator

from deep_hedging.config import GlobalConfig


class CorrelatedHestonSimulator(MonteCarloSimulator):
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        volatility_simulator: VolatilitySimulator,
        rates_simulator: InterestRateSimulator,
        bid_ask_simulator: BidAskSimulator,
        random_seed: [int, None] = None,
    ):
        super().__init__(payoff_function, random_seed)
        self.volatility_simulator = volatility_simulator
        self.rates_simulator = rates_simulator
        self.bid_ask_simulator = bid_ask_simulator

    def get_paths(
        self,
        spot: list[float],
        bid_ask_spread: list[float],
        rf_rate: float,
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[np.array], np.array],
        dividends_fn: Callable[[np.array], np.array],
        corr_fn: Callable[[np.array], np.array],
        n_paths: [int, None] = None,
        *args,
        **kwargs
    ) -> tuple[np.array, np.array]:
        days_till_maturity = int(round(GlobalConfig.TRADING_DAYS * time_till_maturity))

        if n_paths is None:
            n_paths = GlobalConfig.MONTE_CARLO_PATHS

        n_stocks = len(spot)
        noise_size = 3 * n_stocks + 1  # n_stocks + n_bid_asks = n_stocks + n_vols = n_stocks + rates

        time = np.linspace(0, time_till_maturity, days_till_maturity)
        d_time = time[1] - time[0]

        var_covar = np.array(corr_fn(time))
        vol_scaling = np.linalg.cholesky(var_covar)
        vol_scaling = np.array(vol_scaling).reshape(1, len(time), noise_size, noise_size)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        noise = vol_scaling @ np.random.normal(0, 1, size=(n_paths, len(time), noise_size, 1))

        spot_noise = noise[:, :, :n_stocks, :]
        vol_noise = noise[:, :, n_stocks:2 * n_stocks, :]
        bid_ask_noise = noise[:, :, 2 * n_stocks:3 * n_stocks, :]
        rates_noise = noise[:, :, -1, :]

        volatility = self.volatility_simulator.get_paths(noise=vol_noise)

        drift = (risk_free_rate_fn(time) - dividends_fn(time) - 0.5 * volatility) * d_time
        drift = np.array(drift).reshape(1, len(time), n_stocks, 1)
        diffusion = volatility @ spot_noise * np.sqrt(d_time)

        paths = np.exp(drift + diffusion)
        paths = np.insert(paths, 0, np.array(spot).reshape(1, 1, -1, 1), axis=1)
        paths = np.cumprod(paths, axis=1).squeeze(3)

        bid_ask = self.bid_ask_simulator.get_paths(
            noise=bid_ask_noise,
        )

        bids = paths - bid_ask
        asks = paths + bid_ask

        spot_paths = np.stack([bids, asks])

        rates = self.rates_simulator.simulate(
            r0=rf_rate,
            terms=time.tolist(),
            noise=rates_noise,
            n_paths=n_paths,
        )

        return spot_paths, rates
