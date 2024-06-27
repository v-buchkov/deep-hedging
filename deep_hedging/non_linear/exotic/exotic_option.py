import abc
import datetime as dt
from functools import lru_cache

import numpy as np

from deep_hedging.base.instrument import Instrument
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.market_data.market_data import MarketData
from deep_hedging.monte_carlo.monte_carlo_pricer import MonteCarloPricer
from deep_hedging.config.global_config import GlobalConfig


class ExoticOption(Instrument):
    def __init__(
        self,
        underlyings: MarketData,
        yield_curve: YieldCurve,
        strike_level: [float, list[float]],
        start_date: dt.datetime,
        end_date: dt.datetime,
        *args,
        **kwargs
    ):
        super().__init__()

        self.underlyings = underlyings
        self.yield_curve = yield_curve
        self.strike_level = strike_level
        self.start_date = start_date
        self.end_date = end_date

        self.time_till_maturity = (
            self.end_date - self.start_date
        ).days / GlobalConfig.CALENDAR_DAYS
        self._price = None

        self._mc_pricer = MonteCarloPricer(self.payoff)

    # TODO: non-constant term
    @lru_cache(maxsize=None)
    def _volatility_surface(self, term: float) -> np.array:
        return self.underlyings.get_var_covar()

    # TODO: non-constant term
    @lru_cache(maxsize=None)
    def _dividends(self, term: float) -> np.array:
        return self.underlyings.get_dividends()

    def pv_coupons(self) -> float:
        return self.price()

    def coupon(
            self, frequency: float = 0.0, commission: float = 0.0, *args, **kwargs
    ) -> float:
        if frequency > 0:
            annual_rate = self.yield_curve.get_rate(self.time_till_maturity)
            return (self.pv_coupons() - commission) / annuity_factor(
                annual_rate=annual_rate,
                frequency=frequency,
                till_maturity=self.time_till_maturity,
            )
        return 0.0

    def delta(
        self, spot_change: float = 0.01, spot_start: [list[float], None] = None
    ) -> np.array:
        n_stocks = len(self.underlyings)
        spot_up = np.exp(spot_change)
        spot_down = np.exp(-spot_change)

        delta = []
        for i in range(n_stocks):
            if spot_start is None:
                x_up, x_down = [1] * n_stocks, [1] * n_stocks
            else:
                x_up, x_down = spot_start.copy(), spot_start.copy()

            x_up[i] *= spot_up
            x_down[i] *= spot_down

            price_up = self.price(x_up)
            price_down = self.price(x_down)

            delta.append((price_up - price_down) / (spot_up - spot_down))

        return np.array(delta)

    def gamma(
            self, spot_change: float = 0.005, spot_start: [list[float], None] = None
    ) -> np.array:
        n_stocks = len(self.underlyings)
        spot_up = np.exp(spot_change)
        spot_down = np.exp(-spot_change)

        gamma = []
        for i in range(n_stocks):
            if spot_start is None:
                x_up, x_down = [1] * n_stocks, [1] * n_stocks
            else:
                x_up, x_down = spot_start.copy(), spot_start.copy()

            x_up[i] *= spot_up
            x_down[i] *= spot_down

            delta_up = self.delta(spot_start=x_up)
            delta_down = self.delta(spot_start=x_down)

            gamma.append((delta_up - delta_down) / (spot_up - spot_down))

        return np.array(gamma)

    def vega(
        self, vol_change: float = 0.01, spot_start: [list[float], None] = None
    ) -> np.array:
        n_stocks = len(self.underlyings)
        diagonal = np.diag(np.sqrt(np.diag(self.underlyings.get_var_covar())))
        corr = self.underlyings.get_corr()
        price_down = self.price()

        vega = []
        for i in range(n_stocks):
            diag = diagonal.copy()
            diag[i][i] += vol_change
            new_var_covar = diag @ corr @ diag

            future_value_new = self._mc_pricer.get_future_value(
                current_spot=spot_start
                if spot_start is not None
                else [1.0] * len(self.underlyings),
                time_till_maturity=self.time_till_maturity,
                risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
                dividends_fn=self._dividends,
                var_covar_fn=lambda term: new_var_covar,
            )
            price_up = future_value_new * self.yield_curve.get_discount_factor(
                self.time_till_maturity
            )

            vega.append((price_up - price_down) / vol_change)

        return np.array(vega)

    def correlation_sensitivity(
        self, corr_change: float = 0.01, spot_start: [list[float], None] = None
    ) -> np.array:
        n_stocks = len(self.underlyings)
        diagonal = np.diag(np.sqrt(np.diag(self.underlyings.get_var_covar())))
        correlation = self.underlyings.get_corr()
        price_down = self.price()

        vega = []
        for i in range(n_stocks - 1):
            corr = correlation.copy()
            corr[i][i + 1] += corr_change
            corr[i + 1][i] += corr_change
            new_var_covar = diagonal @ corr @ diagonal

            future_value_new = self._mc_pricer.get_future_value(
                current_spot=spot_start
                if spot_start is not None
                else [1.0] * len(self.underlyings),
                time_till_maturity=self.time_till_maturity,
                risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
                dividends_fn=self._dividends,
                var_covar_fn=lambda term: new_var_covar,
            )
            price_up = future_value_new * self.yield_curve.get_discount_factor(
                self.time_till_maturity
            )

            vega.append((price_up - price_down) / corr_change)

        return np.array(vega)

    def price(self, spot_start: [float, list[float], None] = None) -> float:
        future_value = self._mc_pricer.get_future_value(
            current_spot=spot_start
            if spot_start is not None
            else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self._dividends,
            var_covar_fn=self._volatility_surface,
        )
        return future_value * self.yield_curve.get_discount_factor(
            self.time_till_maturity
        )

    def get_paths(self, spot_start: [float, list[float], None] = None) -> np.array:
        return self._mc_pricer.get_paths(
            current_spot=spot_start
            if spot_start is not None
            else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self._dividends,
            var_covar_fn=self._volatility_surface,
        )

    def __repr__(self):
        instrument_str = f"{self.__class__.__name__}:\n"
        underlyings = "\n".join([f"-> {stock}" for stock in self.underlyings.tickers])
        instrument_str += underlyings
        instrument_str += f"* Strike = {self.strike_level * 100}\n"
        if hasattr(self, "barrier_level"):
            instrument_str += f"* Barrier = {self.barrier_level * 100}\n"
        instrument_str += f"* Start Date = {self.start_date}\n"
        instrument_str += f"* End Date = {self.end_date}\n"
        return instrument_str

    @abc.abstractmethod
    def payoff(self, spot_paths: np.array) -> float:
        raise NotImplementedError
