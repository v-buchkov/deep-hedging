import datetime as dt

import numpy as np

from deep_hedging.base import Instrument
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin


class FXForward(FixedMaturityMixin, Instrument):
    def __init__(
        self,
        yield_curve_quote: YieldCurve,
        yield_curve_base: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
        strike: [float, None] = None,
    ):
        super().__init__(
            yield_curve=yield_curve_quote, start_date=start_date, end_date=end_date
        )

        self.yield_curve_quote = yield_curve_quote
        self.yield_curve_base = yield_curve_base

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

    def get_strike(
        self, spot_price: np.array = np.array([1.0]), days: [np.array, None] = None
    ) -> np.array:
        if days is None:
            days = self.days_till_maturity
        return (
            spot_price
            * self.yield_curve_quote.fv_discount_factors(days)
            / self.yield_curve_base.fv_discount_factors(days)
        )

    def price(
        self, spot: np.array = np.array([1.0]), days: [np.array, None] = None
    ) -> float:
        if days is None:
            days = self.days_till_maturity

        strikes = self.get_strike(spot, days)
        intrinsic_value = strikes - self.strike
        price = (
            intrinsic_value
            * self.yield_curve_quote.pv_discount_factors(days)
            / self.yield_curve_base.pv_discount_factors(days)
        )
        if len(price) == 1:
            return price.item()
        else:
            return price

    def payoff(self, spot: np.array = np.array([[1.0]])) -> float:
        if isinstance(spot, float) or isinstance(spot, int):
            return spot - self.strike
        assert (
            spot.ndim > 1
        ), f"If the array consists of one path only, use .reshape(1, -1).\nIf each path consists of one value only, use .reshape(-1, 1)."
        return spot[:, -1] - self.strike

    def __repr__(self):
        instrument_str = f"FXForward:\n"
        instrument_str += f"* Buy = {self.yield_curve_base.currency.name}, Sell = {self.yield_curve_quote.currency.name}\n"
        instrument_str += f"* Term = {round(self.time_till_maturity, 2)} years\n"
        instrument_str += f"* Strike = {round(self.strike * 100, 4)}%\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}"
        return instrument_str
