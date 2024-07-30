import abc
import datetime as dt

import numpy as np

from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.monte_carlo.spot.gbm_simulator import GBMSimulator


class MonteCarloOption(BaseOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        strike_level: float,
        start_date: dt.datetime,
        end_date: dt.datetime,
        random_seed: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            strike_level=strike_level,
            start_date=start_date,
            end_date=end_date,
        )

        self._mc_pricer = GBMSimulator(self.payoff, random_seed=random_seed)

    def delta(
        self, spot_change: float = 0.01, spot: np.array = np.array([1.0])
    ) -> np.array:
        n_stocks = len(self.underlyings)
        spot_up = np.exp(spot_change)
        spot_down = np.exp(-spot_change)

        delta = []
        for i in range(n_stocks):
            if spot is None:
                x_up, x_down = [1] * n_stocks, [1] * n_stocks
            else:
                x_up, x_down = spot.copy(), spot.copy()

            x_up[i] *= spot_up
            x_down[i] *= spot_down

            price_up = self.price(x_up)
            price_down = self.price(x_down)

            delta.append((price_up - price_down) / (spot_up - spot_down))

        return np.array(delta)

    def gamma(
        self, spot_change: float = 0.005, spot: np.array = np.array([1.0])
    ) -> np.array:
        n_stocks = len(self.underlyings)
        spot_up = np.exp(spot_change)
        spot_down = np.exp(-spot_change)

        gamma = []
        for i in range(n_stocks):
            if spot is None:
                x_up, x_down = [1] * n_stocks, [1] * n_stocks
            else:
                x_up, x_down = spot.copy(), spot.copy()

            x_up[i] *= spot_up
            x_down[i] *= spot_down

            delta_up = self.delta(spot=x_up)
            delta_down = self.delta(spot=x_down)

            gamma.append((delta_up - delta_down) / (spot_up - spot_down))

        return np.array(gamma)

    def vega(
        self, vol_change: float = 0.01, spot: np.array = np.array([1.0])
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

            price_up = self.yield_curve.to_present_value(
                self._mc_pricer.future_value(
                    spot=spot if spot is not None else [1.0] * len(self.underlyings),
                    time_till_maturity=self.time_till_maturity,
                    risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
                    dividends_fn=self.dividends,
                    var_covar_fn=lambda term: new_var_covar,
                ),
                self.days_till_maturity,
            )

            vega.append((price_up - price_down) / vol_change)

        return np.array(vega)

    def correlation_sensitivity(
        self, corr_change: float = 0.01, spot: np.array = np.array([1.0])
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

            price_up = self.yield_curve.to_present_value(
                self._mc_pricer.future_value(
                    spot=spot if spot is not None else [1.0] * len(self.underlyings),
                    time_till_maturity=self.time_till_maturity,
                    risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
                    dividends_fn=self.dividends,
                    var_covar_fn=lambda term: new_var_covar,
                ),
                self.days_till_maturity,
            )

            vega.append((price_up - price_down) / corr_change)

        return np.array(vega)

    def price(self, spot: [float, np.array, None] = None) -> float:
        fv = self._mc_pricer.future_value(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self.dividends,
            var_covar_fn=self.volatility_surface,
        )
        return self.yield_curve.to_present_value(fv, self.days_till_maturity)

    def std(self, spot: [float, np.array, None] = None) -> float:
        return self._mc_pricer.std(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self.dividends,
            var_covar_fn=self.volatility_surface,
        )

    def apply_discounting(
        self, payments: np.array, observation_days: np.array
    ) -> np.array:
        dfs = self.yield_curve.fv_discount_factors(
            self.days_till_maturity - observation_days
        )
        return payments * dfs

    @abc.abstractmethod
    def payoff(self, spot_paths: np.array) -> float:
        raise NotImplementedError
