import datetime as dt

import numpy as np

from src.curve.yield_curve import YieldCurve
from src.market_data.market_data import MarketData
from src.option.exotic._exotic_option import ExoticOption
from src.utils.annuity import annuity_factor


class WorstOfBarrierPut(ExoticOption):
    def __init__(
            self,
            underlyings: MarketData,
            yield_curve: YieldCurve,
            strike_level: float,
            barrier_level: float,
            start_date: dt.datetime,
            end_date: dt.datetime
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date
        )

        self.barrier_level = barrier_level

    def pv_coupons(self) -> float:
        return self.price()

    def coupon(self, frequency: float = 0., commission: float = 0., *args, **kwargs) -> float:
        if frequency > 0:
            annual_rate = self.yield_curve.get_rate(self.time_till_maturity)
            return (self.pv_coupons() - commission) / annuity_factor(annual_rate=annual_rate, frequency=frequency,
                                                                     till_maturity=self.time_till_maturity)
        return 0.

    def __repr__(self):
        instrument_str = f"WorstOfBarrierPut:\n"
        underlyings = "\n".join([f"-> {stock}" for stock in self.underlyings.tickers])
        instrument_str += underlyings
        instrument_str += f"* Strike = {self.strike_level * 100}\n"
        instrument_str += f"* Barrier = {self.barrier_level * 100}\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}\n"
        return instrument_str

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.all(spot_paths[:, -1] >= self.barrier_level, axis=1))
        returns = self.strike_level - spot_paths[:, -1].min(axis=1)
        returns[indices] = 0

        return returns
