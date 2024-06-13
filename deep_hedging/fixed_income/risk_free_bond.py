import datetime as dt

import numpy as np

from deep_hedging.base.instrument import Instrument
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.utils.annuity import annuity_factor
from deep_hedging.config.global_config import GlobalConfig


class RiskFreeBond(Instrument):
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ):
        super().__init__()
        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date

        self.time_till_maturity = (
            self.end_date - self.start_date
        ).days / GlobalConfig.CALENDAR_DAYS

    def pv_coupons(self) -> float:
        return 1 - self.price()

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        if frequency > 0:
            annual_rate = self.yield_curve.get_rate(self.time_till_maturity)
            return self.pv_coupons() / annuity_factor(
                annual_rate=annual_rate,
                frequency=frequency,
                till_maturity=self.time_till_maturity,
            )
        return 0

    def price(self):
        return self.yield_curve.get_discount_factor(self.time_till_maturity)

    def payoff(self, spot_paths: np.array) -> float:
        return 1

    def __repr__(self):
        instrument_str = f"RiskFreeBond:\n"
        instrument_str += f"* Term = {round(self.time_till_maturity, 2)} years\n"
        instrument_str += f"* YTM = {round(self.yield_curve.get_rate(self.time_till_maturity) * 100, 2)}%\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}\n"
        return instrument_str
