import datetime as dt

import numpy as np

from deep_hedging.base import Instrument
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin


class Forward(FixedMaturityMixin, Instrument):
    def __init__(
            self,
            yield_curve: YieldCurve,
            start_date: dt.datetime,
            end_date: dt.datetime,
            strike: [float, None] = None
    ):
        super().__init__(yield_curve=yield_curve, start_date=start_date, end_date=end_date)

        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date

        if strike is None:
            self.strike = float(self.get_strike()[0])
        else:
            self.strike = strike

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        return 0

    def pv_coupons(self) -> float:
        return 0

    def _get_discount_factors(self, times: [np.array, None] = None) -> np.array:
        if times is None:
            return self.discount_factor(
                rate=-self.yield_curve.get_rate(self.time_till_maturity), term=self.time_till_maturity
            )
        else:
            return self.discount_factor(
                rate=-self.yield_curve.get_rate(times), term=times
            )

    def get_strike(self, spot_price: np.array = np.array([1.]), times: [np.array, None] = None) -> np.array:
        return spot_price * self._get_discount_factors(times)

    def price(self, spot: np.array = np.array([1.]), times: [np.array, None] = None) -> float:
        strikes = self.get_strike(spot, times)
        intrinsic_value = strikes - self.strike
        return intrinsic_value * self._get_discount_factors(times)

    def payoff(self, spot: np.array) -> float:
        assert spot.ndim > 1, f"If the array consists of one spot-ref only, use .reshape(1, -1)"
        return spot[:, -1] - self.strike

    def __repr__(self):
        instrument_str = f"Forward:\n"
        instrument_str += f"* Term = {round(self.time_till_maturity, 2)} years\n"
        instrument_str += f"* Strike = {round(self.strike * 100, 4)}%\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}"
        return instrument_str
