import datetime as dt

import numpy as np

from deep_hedging.base.frequency import Frequency
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.non_linear.monte_carlo_option import MonteCarloOption
from deep_hedging.utils.fixing_dates import get_periods_indices


class WorstOfDigitalMemoryCall(MonteCarloOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        strike_level: float,
        digital_coupon: float,
        frequency: Frequency,
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
        self.digital_coupon = digital_coupon
        self.frequency = frequency

    def payoff(self, spot_paths: np.array) -> np.array:
        periods, observation_days = get_periods_indices(
            self.time_till_maturity, self.frequency.value
        )
        coupons = np.where(
            np.all(spot_paths[:, observation_days, :] >= self.strike_level, axis=2),
            self.digital_coupon,
            0,
        )

        for i in range(coupons.shape[1]):
            coupons[:, i] *= i + 1

        payments = coupons.copy()
        payments[:, 1:] = np.diff(coupons, axis=1)

        payments = np.where(payments > 0, payments, 0)
        payments = self.apply_discounting(payments, observation_days)

        return payments.sum(axis=1)


class WorstOfDigitalMemoryPut(MonteCarloOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        initial_spot: [float, list[float]],
        strike_level: float,
        digital_coupon: float,
        frequency: Frequency,
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
        self.digital_coupon = digital_coupon
        self.frequency = frequency

    def payoff(self, spot_paths: np.array) -> np.array:
        periods, observation_days = get_periods_indices(
            self.time_till_maturity, self.frequency.value
        )
        coupons = np.where(
            np.all(spot_paths[:, observation_days, :] <= self.strike_level, axis=2),
            self.digital_coupon,
            0,
        )

        for i in range(coupons.shape[1]):
            coupons[:, i] *= i + 1

        payments = coupons.copy()
        payments[:, 1:] = np.diff(coupons, axis=1)

        payments = np.where(payments > 0, payments, 0)
        payments = self.apply_discounting(payments, observation_days)

        return payments.sum(axis=1)
