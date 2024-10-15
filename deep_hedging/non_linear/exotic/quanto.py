import numpy as np
import pandas as pd

from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.curve.yield_curve import YieldCurves
from deep_hedging.monte_carlo.spot.gbm_quanto_simulator import GBMQuantoSimulator


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

        self._quanto_pricer = GBMQuantoSimulator(
            self.option.payoff,
            option.underlyings,
            modifying_underlyings,
            random_seed=random_seed,
        )

    # TODO: non-constant vol
    def volatility_surface(self, terms: np.array) -> np.array:
        return np.array([self.underlyings.get_var_covar()] * len(terms))

    def risk_free_rate(self, terms: np.array, currency: str = None) -> np.array:
        if currency is None:
            currency = self.option.yield_curve.currency.value
        return self.yield_curves[currency].get_instant_fwd_rate(terms)

    def dividends(self, t: np.array) -> np.array:
        dividends = self.option.dividends(t)
        fx_rates = np.array(
            [-self.option.yield_curve.get_instant_fwd_rate(t)]
            * len(self.modifying_underlyings)
        ).reshape(len(t), len(self.modifying_underlyings))
        return np.concatenate([dividends, fx_rates], axis=1)

    def price(self, spot: [float, np.array, None] = None) -> float:
        fv = self._quanto_pricer.future_value(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.option.time_till_maturity,
            risk_free_rate_fn=self.risk_free_rate,
            dividends_fn=self.dividends,
            var_covar_fn=self.volatility_surface,
        )

        return self.option.yield_curve.to_present_value(
            fv, self.option.days_till_maturity
        )
