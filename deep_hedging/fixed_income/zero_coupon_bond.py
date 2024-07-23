import datetime as dt

import numpy as np

from deep_hedging.base.instrument import Instrument
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.utils.annuity import annuity_factor


class ZeroCouponBond(FixedMaturityMixin, Instrument):
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ):
        super().__init__(
            yield_curve=yield_curve, start_date=start_date, end_date=end_date
        )

        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date

    def pv_coupons(self) -> float:
        return 1 - self.price()

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        if frequency > 0:
            annual_rate = self.yield_curve(self.days_till_maturity)
            return self.pv_coupons() / annuity_factor(
                annual_rate=annual_rate,
                frequency=frequency,
                till_maturity=self.time_till_maturity,
            )
        return 0

    def price(self, spot_paths: np.array = np.array([1.0])):
        return self.yield_curve.to_present_value(
            self.payoff(spot_paths=spot_paths), self.days_till_maturity
        )

    def payoff(self, spot_paths: np.array = np.array([1.0])) -> float:
        return 1

    @property
    def ytm(self):
        return self.yield_curve(self.days_till_maturity)

    def __repr__(self):
        instrument_str = f"ZeroCouponBond:\n"
        if self.yield_curve.currency is not None:
            instrument_str += f"* CCY = {self.yield_curve.currency}\n"
        instrument_str += f"* Term = {round(self.time_till_maturity, 2)} years\n"
        instrument_str += f"* YTM = {round(self.ytm * 100, 2)}%\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}"
        return instrument_str
