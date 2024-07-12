import datetime as dt

import numpy as np

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.non_linear.monte_carlo_option import MonteCarloOption


class WorstOfCall(MonteCarloOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        strike_level: float,
        start_date: dt.datetime,
        end_date: dt.datetime,
        random_seed: int = None,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
        )

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.any(spot_paths[:, -1] <= self.strike_level, axis=1))
        returns = spot_paths[:, -1].min(axis=1) - self.strike_level
        returns[indices] = 0

        return returns


class WorstOfPut(MonteCarloOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        initial_spot: [float, list[float]],
        strike_level: float,
        start_date: dt.datetime,
        end_date: dt.datetime,
        random_seed: int = None,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            initial_spot=initial_spot,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
        )

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.all(spot_paths[:, -1] >= self.strike_level, axis=1))
        returns = self.strike_level - spot_paths[:, -1].min(axis=1)
        returns[indices] = 0

        return returns
