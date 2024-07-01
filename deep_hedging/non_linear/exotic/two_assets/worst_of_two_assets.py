import datetime as dt

import numpy as np
from scipy.stats import multivariate_normal

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.market_data.underlyings import Underlyings
from deep_hedging.non_linear.monte_carlo_option import MonteCarloOption
from deep_hedging.non_linear.exotic.two_assets import TwoAssetsExchange


class WorstOfCallTwoAssets(MonteCarloOption):
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

    def _closed_out_price(self, spot_start: [float, list[float], None] = None) -> float:
        assert (
            isinstance(spot_start, list) and len(spot_start) == 2
        ), "This experiment is valid for 2 assets only!"
        spot1, spot2 = spot_start

        tau = self.time_till_maturity

        var_covar = self.volatility_surface(tau)
        vol1, vol2 = np.sqrt(np.diag(var_covar))
        corr = var_covar[0, 1] / (vol1 * vol2)
        vol_joint = np.sqrt(vol1**2 + vol2**2 - 2 * corr * vol1 * vol2)

        rate = self.yield_curve.get_rate(tau)

        gamma1 = (np.log(spot1 / self.strike_level) + (rate - vol1**2 / 2) * tau) / (
            vol1 * np.sqrt(tau)
        )
        gamma2 = (np.log(spot2 / self.strike_level) + (rate - vol2**2 / 2) * tau) / (
            vol2 * np.sqrt(tau)
        )

        cdf1 = self._bivariate_cdf(
            upper_limit1=gamma1 + vol1 * np.sqrt(tau),
            upper_limit2=(np.log(spot2 / spot1) - vol_joint**2 / 2 * np.sqrt(tau))
            / (vol_joint * np.sqrt(tau)),
            correlation=(corr * vol2 - vol1) / vol_joint,
        )
        cdf2 = self._bivariate_cdf(
            upper_limit1=gamma2 + vol2 * np.sqrt(tau),
            upper_limit2=(np.log(spot1 / spot2) - vol_joint**2 / 2 * np.sqrt(tau))
            / (vol_joint * np.sqrt(tau)),
            correlation=(corr * vol1 - vol2) / vol_joint,
        )
        cdf_below = self._bivariate_cdf(
            upper_limit1=gamma1, upper_limit2=gamma2, correlation=corr
        )

        discount_factor = self.discount_factor(rate, tau)

        return (
            spot1 * cdf1
            + spot2 * cdf2
            - self.strike_level * discount_factor * cdf_below
        )

    def price(self, spot_start: [float, list[float], None] = None) -> float:
        assert (
            isinstance(spot_start, list) and len(spot_start) == 2
        ), "This experiment is valid for 2 assets only!"
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


class WorstOfPutTwoAssets(MonteCarloOption):
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

    def price(self, spot_start: [float, list[float], None] = None) -> float:
        assert (
            isinstance(spot_start, list) and len(spot_start) == 2
        ), "This experiment is valid for 2 assets only!"

        tau = self.time_till_maturity
        rate = self.yield_curve.get_rate(tau)
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
