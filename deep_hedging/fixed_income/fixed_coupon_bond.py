import datetime as dt

import numpy as np

from deep_hedging.base.instrument import Instrument
from deep_hedging.base.frequency import Frequency
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.fixed_income.zero_coupon_bond import ZeroCouponBond

from deep_hedging.config.global_config import GlobalConfig


class FixedCouponBond(FixedMaturityMixin, Instrument):
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
        fixed_coupon: float,
        frequency: Frequency,
    ):
        super().__init__(
            yield_curve=yield_curve,
            start_date=start_date,
            end_date=end_date,
            fixed_coupon=fixed_coupon,
            frequency=frequency,
        )

        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_coupon = fixed_coupon
        self.frequency = frequency

        self.portfolio = self._create_portfolio()

    def _create_portfolio(self):
        portfolio = ZeroCouponBond(
            yield_curve=self.yield_curve,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        point = self.frequency.value
        while point < self.time_till_maturity + self.frequency.value:
            portfolio += (
                ZeroCouponBond(
                    yield_curve=self.yield_curve,
                    start_date=self.start_date,
                    end_date=self.start_date
                    + dt.timedelta(days=int(round(point * GlobalConfig.CALENDAR_DAYS))),
                )
                * self.fixed_coupon
            )
            point += self.frequency.value

        return portfolio

    def pv_coupons(self) -> float:
        return self.portfolio.pv_coupons()

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        return self.fixed_coupon

    def price(self):
        return self.portfolio.price()

    def payoff(self, spot_paths: np.array) -> float:
        return self.portfolio.payoff(spot_paths)

    def __repr__(self):
        return self.portfolio.__repr__()
