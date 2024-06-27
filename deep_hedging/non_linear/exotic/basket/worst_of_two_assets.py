import datetime as dt

import numpy as np

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.market_data.market_data import MarketData
from deep_hedging.non_linear.exotic.exotic_option import ExoticOption


class WorstOfCallTwoAssets(ExoticOption):
    def __init__(
        self,
        underlyings: MarketData,
        yield_curve: YieldCurve,
        strike_level: float,
        barrier_level: float,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
        )

        self.barrier_level = barrier_level

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.all(spot_paths[:, -1] >= self.barrier_level, axis=1))
        returns = self.strike_level - spot_paths[:, -1].min(axis=1)
        returns[indices] = 0

        return returns
