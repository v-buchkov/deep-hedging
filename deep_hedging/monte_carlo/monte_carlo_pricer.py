import abc
from typing import Callable

import numpy as np

from deep_hedging.utils.linalg import corr_matrix_from_cov


class MonteCarloPricer:
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        random_seed: [int, None] = None,
    ):
        self.payoff_function = payoff_function
        self.random_seed = random_seed

    def delta(
        self,
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        spot_change: float = 0.01,
        spot_start: [list[float], None] = None,
        n_paths: [int, None] = None,
    ) -> np.array:
        n_stocks = len(var_covar_fn(time_till_maturity).shape[0])

        delta = []
        for i in range(n_stocks):
            if spot_start is None:
                x_up, x_down = [1] * n_stocks, [1] * n_stocks
            else:
                x_up, x_down = spot_start.copy(), spot_start.copy()

            x_up[i] += spot_change
            x_down[i] -= spot_change

            price_down = self.price(
                current_spot=x_down,
                time_till_maturity=time_till_maturity,
                risk_free_rate_fn=risk_free_rate_fn,
                dividends_fn=dividends_fn,
                var_covar_fn=var_covar_fn,
                n_paths=n_paths,
            )

            price_up = self.price(
                current_spot=x_up,
                time_till_maturity=time_till_maturity,
                risk_free_rate_fn=risk_free_rate_fn,
                dividends_fn=dividends_fn,
                var_covar_fn=var_covar_fn,
                n_paths=n_paths,
            )

            delta.append((price_up - price_down) / spot_change)

        return np.array(delta)

    def gamma(
        self,
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        spot_change: float = 0.01,
        spot_start: [list[float], None] = None,
        n_paths: [int, None] = None,
    ) -> np.array:
        n_stocks = len(var_covar_fn(time_till_maturity).shape[0])

        if spot_start is None:
            x_down = [1] * n_stocks
        else:
            x_down = spot_start.copy()

        delta_down = self.delta(
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            spot_start=x_down,
            n_paths=n_paths,
        )

        gamma = []
        for i in range(n_stocks):
            x_up = x_down.copy()

            x_up[i] += spot_change

            delta_up = self.delta(
                time_till_maturity=time_till_maturity,
                risk_free_rate_fn=risk_free_rate_fn,
                dividends_fn=dividends_fn,
                var_covar_fn=var_covar_fn,
                spot_start=x_up,
                n_paths=n_paths,
            )

            gamma.append((delta_up - delta_down) / spot_change)

        return np.array(gamma)

    def vega(
        self,
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        vol_change: float = 0.01,
        spot_start: [list[float], None] = None,
        n_paths: [int, None] = None,
    ) -> np.array:
        n_stocks = len(var_covar_fn(time_till_maturity).shape[0])

        var_covar = var_covar_fn(time_till_maturity)
        diagonal = np.diag(np.sqrt(np.diag(var_covar)))
        corr = corr_matrix_from_cov(var_covar)
        self.price(
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths,
        )

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

    @abc.abstractmethod
    def get_paths(
        self,
        current_spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ) -> np.array:
        raise NotImplementedError

    def _future_value(
        self,
        current_spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ) -> float:
        random_paths = self.get_paths(
            current_spot=current_spot,
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths,
        )

        instrument_payoffs = self.payoff_function(random_paths)
        return np.mean(instrument_payoffs)

    def price(
        self,
        current_spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ):
        fv = self._future_value(
            current_spot=current_spot,
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths,
        )
        return fv * np.exp(-time_till_maturity * risk_free_rate_fn(time_till_maturity))
