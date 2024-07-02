import datetime as dt

import numpy as np
from scipy.stats import norm

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.underlyings.underlyings import Underlyings
from deep_hedging.non_linear.monte_carlo_option import MonteCarloOption


class TwoAssetsExchange(MonteCarloOption):
    def __init__(
        self,
        underlyings: Underlyings,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ):
        super().__init__(
            underlyings=underlyings,
            yield_curve=yield_curve,
            strike_level=0.0,
            start_date=start_date,
            end_date=end_date,
        )

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

        d1 = (np.log(spot1 / spot2) + vol_joint**2 / 2 * tau) / (
            vol_joint * np.sqrt(tau)
        )
        d2 = d1 - vol_joint * np.sqrt(tau)

        return spot1 * norm.cdf(d1) - spot2 * norm.cdf(d2)

    def price(self, spot_start: [float, list[float], None] = None) -> float:
        assert (
            isinstance(spot_start, list) and len(spot_start) == 2
        ), "This experiment is valid for 2 assets only!"
        return self._closed_out_price(spot_start=spot_start)

    def payoff(self, spot_paths: np.array) -> np.array:
        return np.max(spot_paths[:, 0] - spot_paths[:, 1], 0)
