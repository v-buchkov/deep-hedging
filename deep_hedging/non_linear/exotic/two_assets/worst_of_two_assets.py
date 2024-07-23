import datetime as dt

import numpy as np
from scipy.stats import multivariate_normal

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.non_linear.exotic.two_assets import TwoAssetsExchange


class WorstOfCallTwoAssets(BaseOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        strike_level: float,
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

    @staticmethod
    def _bivariate_cdf(upper_limit1: float, upper_limit2: float, correlation: float):
        diag = np.identity(2)
        corr = np.array([[1, correlation], [correlation, 1]])
        var_covar = diag @ corr @ diag
        dist = multivariate_normal(mean=None, cov=var_covar)
        return dist.cdf(np.array([upper_limit1, upper_limit2]))

    @staticmethod
    def discount_factor(rate: float, term: float) -> float:
        return np.exp(-rate * term)

    def _closed_out_price(self, spot_start: np.array) -> float:
        """Using notation from (Stulz, 1982)"""
        self.strike_level = self.strike_level + 1e-12
        v, h = spot_start

        tau = self.time_till_maturity

        var_covar = self.volatility_surface(tau)
        vol_v, vol_h = np.sqrt(np.diag(var_covar))
        corr = var_covar[0, 1] / (vol_v * vol_h)
        vol_joint = np.sqrt(vol_v**2 + vol_h**2 - 2 * corr * vol_v * vol_h)

        rate = self.yield_curve.rate(tau)

        gamma1 = (np.log(h / self.strike_level) + (rate - vol_h**2 / 2) * tau) / (
            vol_h * np.sqrt(tau)
        )
        gamma2 = (np.log(v / self.strike_level) + (rate - vol_v**2 / 2) * tau) / (
            vol_v * np.sqrt(tau)
        )

        cdf1 = self._bivariate_cdf(
            upper_limit1=gamma1 + vol_h * np.sqrt(tau),
            upper_limit2=(
                np.log(v / h) - vol_joint**2 / 2 * tau
            )  # beware: not * np.sqrt(tau), but * tau
            / (vol_joint * np.sqrt(tau)),
            correlation=(corr * vol_v - vol_h) / vol_joint,
        )
        cdf2 = self._bivariate_cdf(
            upper_limit1=gamma2 + vol_v * np.sqrt(tau),
            upper_limit2=(
                np.log(h / v) - vol_joint**2 / 2 * tau
            )  # beware: not * np.sqrt(tau), but * tau
            / (vol_joint * np.sqrt(tau)),
            correlation=(corr * vol_h - vol_v) / vol_joint,
        )
        cdf_below = self._bivariate_cdf(
            upper_limit1=gamma1, upper_limit2=gamma2, correlation=corr
        )

        discount_factor = self.discount_factor(rate, tau)

        return h * cdf1 + v * cdf2 - self.strike_level * discount_factor * cdf_below

    def price(self, spot_start: np.array = np.array([1.0, 1.0])) -> float:
        assert len(spot_start) == 2, "This experiment is valid for 2 assets only!"
        if self.strike_level == 0.0:
            spot1, _ = spot_start
            exchange_opt = TwoAssetsExchange(
                underlyings=self.underlyings,
                yield_curve=self.yield_curve,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            return spot1 - exchange_opt.price(spot_start)
        else:
            return self._closed_out_price(spot_start=spot_start)

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.any(spot_paths[:, -1] <= self.strike_level, axis=1))
        returns = spot_paths[:, -1].min(axis=1) - self.strike_level
        returns[indices] = 0

        return returns


class WorstOfPutTwoAssets(BaseOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        strike_level: float,
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

    @staticmethod
    def _bivariate_cdf(upper_limit1: float, upper_limit2: float, correlation: float):
        diag = np.identity(2)
        corr = np.array([[1, correlation], [correlation, 1]])
        var_covar = diag @ corr @ diag
        dist = multivariate_normal(mean=None, cov=var_covar)
        return dist.cdf(np.array([upper_limit1, upper_limit2]))

    @staticmethod
    def discount_factor(rate: float, term: float) -> float:
        return np.exp(-rate * term)

    def price(self, spot_start: np.array = np.array([1.0, 1.0])) -> float:
        assert len(spot_start) == 2, "This experiment is valid for 2 assets only!"

        tau = self.time_till_maturity
        rate = self.yield_curve.rate(tau)
        discount_factor = self.discount_factor(rate, tau)

        worst_of_spot = WorstOfCallTwoAssets(
            underlyings=self.underlyings,
            yield_curve=self.yield_curve,
            strike_level=0.0,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        worst_of_call = WorstOfCallTwoAssets(
            underlyings=self.underlyings,
            yield_curve=self.yield_curve,
            strike_level=self.strike_level,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        return (
            self.strike_level * discount_factor
            - worst_of_spot.price(spot_start)
            + worst_of_call.price(spot_start)
        )

    def payoff(self, spot_paths: np.array) -> np.array:
        indices = np.where(np.all(spot_paths[:, -1] >= self.strike_level, axis=1))
        returns = self.strike_level - spot_paths[:, -1].min(axis=1)
        returns[indices] = 0

        return returns
