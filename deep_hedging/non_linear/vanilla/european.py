import datetime as dt

import numpy as np
import scipy.stats as scs

from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.market_data.underlyings import Underlyings


class EuropeanCall(BaseOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        initial_spot: [float, np.array],
        strike_level: [float, np.array],
        start_date: dt.datetime,
        end_date: dt.datetime,
        *args,
        **kwargs,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            initial_spot=initial_spot,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
        )

        self.implied_vol = np.sqrt(np.diag(self.underlyings.get_var_covar()))

    def payoff(self, spot: float) -> float:
        return np.maximum(spot - self.strike_level, 0)

    def price(self, spot: [float, np.array, None] = None) -> float:
        if spot is None:
            spot = np.array([1.0])

        rf_rate = self.yield_curve.get_rate(self.time_till_maturity)

        d1 = (
            np.log(spot / self.strike_level)
            + (rf_rate + self.implied_vol**2 / 2) * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))
        d2 = d1 - self.implied_vol * np.sqrt(self.time_till_maturity)

        cdf_d1 = scs.norm.cdf(d1)
        cdf_d2 = scs.norm.cdf(d2)

        call_price = spot * cdf_d1 - cdf_d2 * self.strike_level * np.exp(
            -rf_rate * self.time_till_maturity
        )

        return call_price

    def delta(
        self, spot_change: float = 0.01, spot: np.array = np.array([1.0])
    ) -> np.array:
        if spot is None:
            spot = np.array([1.0])

        d1 = (
            np.log(spot / self.strike_level)
            + (
                self.yield_curve.get_rate(self.time_till_maturity)
                + self.implied_vol**2 / 2
            )
            * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))

        cdf_d1 = scs.norm.cdf(d1)

        return cdf_d1

    def gamma(self, spot_change: float = 0.005, spot: np.array = np.array([1.0])):
        if spot is None:
            spot = np.array([1.0])

        d1 = (
            np.log(spot / self.strike_level)
            + (
                self.yield_curve.get_rate(self.time_till_maturity)
                + self.implied_vol**2 / 2
            )
            * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))

        pdf_d1 = scs.norm.pdf(d1)

        return pdf_d1 / (spot * self.implied_vol * self.time_till_maturity)

    def vega(self, vol_change: float = 0.01, spot: np.array = np.array([1.0])):
        d1 = (
            np.log(spot / self.strike_level)
            + (
                self.yield_curve.get_rate(self.time_till_maturity)
                + self.implied_vol**2 / 2
            )
            * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))

        pdf_d1 = scs.norm.pdf(d1)

        return spot * np.sqrt(self.time_till_maturity) * pdf_d1

    def theta(self, time_change: float = 1 / 365, spot: np.array = np.array([1.0])):
        rf_rate = self.yield_curve.get_rate(self.time_till_maturity)

        d1 = (
            np.log(spot / self.strike_level)
            + (rf_rate + self.implied_vol**2 / 2) * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))
        d2 = d1 - self.implied_vol * np.sqrt(self.time_till_maturity)

        cdf_d2 = scs.norm.cdf(d2)

        pdf_d1 = scs.norm.pdf(d1)

        return (
            -(spot * pdf_d1 * self.implied_vol) / (2 * np.sqrt(self.time_till_maturity))
            - rf_rate
            * self.strike_level
            * np.exp(-rf_rate * self.time_till_maturity)
            * cdf_d2
        )

    def rho(self, rate_change: float = 0.25 / 100, spot: np.array = np.array([1.0])):
        rf_rate = self.yield_curve.get_rate(self.time_till_maturity)

        d1 = (
            np.log(spot / self.strike_level)
            + (rf_rate + self.implied_vol**2 / 2) * self.time_till_maturity
        ) / (self.implied_vol * np.sqrt(self.time_till_maturity))
        d2 = d1 - self.implied_vol * np.sqrt(self.time_till_maturity)

        cdf_d2 = scs.norm.cdf(d2)

        return (
            self.strike_level
            * self.time_till_maturity
            * np.exp(-rf_rate * self.time_till_maturity)
            * cdf_d2
        )


class EuropeanPut(BaseOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        initial_spot: [float, np.array],
        strike_level: [float, np.array],
        start_date: dt.datetime,
        end_date: dt.datetime,
        *args,
        **kwargs,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            initial_spot=initial_spot,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
        )

        self.european_call = EuropeanCall(
            underlyings=underlyings,
            yield_curve=yield_curve,
            initial_spot=initial_spot,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
        )

    def payoff(self, spot: float) -> float:
        return np.maximum(self.strike_level - spot, 0)

    def price(self, spot: [float, np.array, None] = None) -> float:
        return (
            self.european_call.price(spot=spot)
            - spot
            + self.strike_level
            * np.exp(
                -self.yield_curve.get_rate(self.time_till_maturity)
                * self.time_till_maturity
            )
        )

    def delta(
        self, spot_change: float = 0.01, spot: np.array = np.array([1.0])
    ) -> np.array:
        return self.european_call.delta(spot_change=spot_change, spot=spot) - 1

    def gamma(self, spot_change: float = 0.005, spot: np.array = np.array([1.0])):
        return self.european_call.gamma(spot_change=spot_change, spot=spot)

    def vega(self, vol_change: float = 0.01, spot: np.array = np.array([1.0])):
        return self.european_call.vega(vol_change=vol_change, spot=spot)

    def theta(self, time_change: float = 1 / 365, spot: np.array = np.array([1.0])):
        rf_rate = self.yield_curve.get_rate(self.time_till_maturity)
        # Assuming that [dS/dt = 0]
        return self.european_call.theta(
            time_change=time_change, spot=spot
        ) - self.strike_level * rf_rate * np.exp(-rf_rate * self.time_till_maturity)

    def rho(self, rate_change: float = 0.25 / 100, spot: np.array = np.array([1.0])):
        return self.european_call.rho(
            rate_change=rate_change, spot=spot
        ) - self.strike_level * self.time_till_maturity * np.exp(
            -self.yield_curve.get_rate(self.time_till_maturity)
            * self.time_till_maturity
        )
