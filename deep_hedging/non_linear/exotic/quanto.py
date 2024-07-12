from functools import lru_cache

import numpy as np
import pandas as pd

from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.curve.yield_curve import YieldCurves
from deep_hedging.monte_carlo.gbm_quanto_pricer import GBMQuantoPricer


class QuantoOption:
    def __init__(
        self,
        option: BaseOption,
        modifying_underlyings: Underlyings,
        yield_curves: YieldCurves,
        random_seed: int = None,
    ):
        self.option = option
        self.modifying_underlyings = modifying_underlyings
        self.yield_curves = yield_curves

        data = pd.merge_asof(
            left=option.underlyings.data,
            right=modifying_underlyings.data,
            right_index=True,
            left_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("1 min"),
        )
        self.underlyings = Underlyings(
            tickers=self.option.underlyings.tickers + modifying_underlyings.tickers,
            start=self.option.underlyings.start,
            end=self.option.underlyings.end,
            data=data,
        )

        self._quanto_pricer = GBMQuantoPricer(
            self.option.payoff,
            option.underlyings,
            modifying_underlyings,
            random_seed=random_seed,
        )

    # TODO: non-constant vol
    def volatility_surface(self, t: np.array) -> np.array:
        return self.underlyings.get_var_covar()

    @lru_cache
    def risk_free_rate(self, t: np.array, currency: str = None) -> np.array:
        if currency is None:
            currency = self.option.yield_curve.currency.value
        return self.yield_curves[currency].get_instant_fwd_rate(t)

    @lru_cache
    def dividends(self, t: np.array) -> np.array:
        dividends = self.option.dividends(t)
        fx_rates = np.array(
            [-self.option.yield_curve.get_instant_fwd_rate(t)]
            * len(self.modifying_underlyings)
        )
        return np.concatenate([dividends, fx_rates], axis=0)

    def price(self, spot: [float, np.array, None] = None) -> float:
        return self._quanto_pricer.price(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.option.time_till_maturity,
            risk_free_rate_fn=self.risk_free_rate,
            dividends_fn=self.dividends,
            var_covar_fn=self.volatility_surface,
        )
