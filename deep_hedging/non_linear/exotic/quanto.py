import datetime as dt

import numpy as np

from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.non_linear.monte_carlo_option import MonteCarloOption
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.curve.yield_curve import YieldCurve


class QuantoOption:
    def __init__(
        self,
        option: BaseOption,
        modifying_underlying: Underlyings,
        yield_curve: YieldCurve,
    ):
        self.option = option
        self.modifying_underlying = modifying_underlying
        self.yield_curve = yield_curve

    def price(self, spot: [float, np.array, None] = None) -> float:
        return self._mc_pricer.price(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self._dividends,
            var_covar_fn=self.volatility_surface,
        )

    def get_base_paths(self, spot: [float, np.array, None] = None) -> np.array:
        return self._mc_pricer.get_paths(
            spot=spot if spot is not None else [1.0] * len(self.underlyings),
            time_till_maturity=self.time_till_maturity,
            risk_free_rate_fn=self.yield_curve.get_instant_fwd_rate,
            dividends_fn=self._dividends,
            var_covar_fn=self.volatility_surface,
        )

    def payoff(self, spot_paths: np.array) -> float:
        return self.option.payoff * spot_paths
