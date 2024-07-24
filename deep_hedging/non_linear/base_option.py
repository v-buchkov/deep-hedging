import abc
import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

from deep_hedging.base.instrument import Instrument
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.utils import annuity_factor


class BaseOption(FixedMaturityMixin, Instrument):
    def __init__(
        self,
        yield_curve: YieldCurve,
        strike_level: float,
        start_date: dt.datetime,
        end_date: dt.datetime,
        underlyings: Underlyings = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            yield_curve=yield_curve,
            start_date=start_date,
            end_date=end_date,
        )

        self.underlyings = underlyings
        self.yield_curve = yield_curve
        self.strike_level = strike_level
        self.start_date = start_date
        self.end_date = end_date

    # TODO: non-constant term + call to self.strike
    def volatility_surface(self, term: np.array) -> np.array:
        if isinstance(term, float) or isinstance(term, int):
            return self.underlyings.get_var_covar()
        return np.array([self.underlyings.get_var_covar()] * len(term))

    # TODO: non-constant term
    def dividends(self, term: np.array) -> np.array:
        if isinstance(term, float) or isinstance(term, int):
            return self.underlyings.get_dividends()
        return np.array([self.underlyings.get_dividends()] * len(term))

    def pv_coupons(self) -> float:
        return self.price()

    def coupon(
        self, frequency: float = 0.0, commission: float = 0.0, *args, **kwargs
    ) -> float:
        if frequency > 0:
            annual_rate = self.yield_curve(self.days_till_maturity)
            return self.pv_coupons() / annuity_factor(
                annual_rate=annual_rate,
                frequency=frequency,
                till_maturity=self.time_till_maturity,
            )
        return 0.0

    # TODO: add normalization decorator
    @abc.abstractmethod
    def delta(
        self, spot_change: float = 0.01, spot: np.array = np.array([1.0])
    ) -> np.array:
        raise NotImplementedError

    # TODO: add normalization decorator
    @abc.abstractmethod
    def gamma(
        self, spot_change: float = 0.005, spot: np.array = np.array([1.0])
    ) -> np.array:
        raise NotImplementedError

    @abc.abstractmethod
    def vega(
        self, vol_change: float = 0.01, spot: np.array = np.array([1.0])
    ) -> np.array:
        raise NotImplementedError

    @abc.abstractmethod
    def theta(self, time_change: float = 1 / 365, spot: np.array = np.array([1.0])):
        raise NotImplementedError

    @abc.abstractmethod
    def rho(self, rate_change: float = 0.25 / 100, spot: np.array = np.array([1.0])):
        raise NotImplementedError

    def __repr__(self):
        instrument_str = f"{self.__class__.__name__}:\n"
        instrument_str += "* Underlyings:\n"
        underlyings = "\n".join([f"-> {stock}" for stock in self.underlyings.tickers])
        instrument_str += underlyings
        if hasattr(self, "strike_level"):
            instrument_str += f"\n* Strike = {self.strike_level * 100}\n"
        if hasattr(self, "barrier_level"):
            instrument_str += f"* Barrier = {self.barrier_level * 100}\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}\n"
        return instrument_str

    @abc.abstractmethod
    def price(self, spot: [float, np.array, None] = None) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def payoff(self, spot_paths: np.array) -> float:
        raise NotImplementedError
