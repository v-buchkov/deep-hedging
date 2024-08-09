import numpy as np
import pandas as pd

from deep_hedging.monte_carlo.volatility.vasicek_volatility_simulator import VasicekVolatilitySimulator
from deep_hedging.base.volatility_surface import VolatilitySurface


class HullWhiteVolatilitySimulator(VasicekVolatilitySimulator):
    def __init__(
            self,
            random_seed: [int, None] = None,
    ) -> None:
        super().__init__(random_seed=random_seed)

    def calibrate_mu(self, volatility_surface: VolatilitySurface) -> None:
        t_old = yield_curve.instant_fwd_rate.index[0]
        fwd_rate_old = yield_curve.instant_fwd_rate.iloc[0, 0]
        mu = []
        for t, fwd_rate in yield_curve.instant_fwd_rate.iloc[1:, :].itertuples():
            dfwd_rate = (fwd_rate - fwd_rate_old) / (t - t_old)

            # TODO: check that not constant
            mu_bootstrapped = (
                    dfwd_rate
                    + self.lambda_(t) * fwd_rate
                    + self.sigma**2
                    / (2 * self.lambda_(t))
                    * (1 - np.exp(-2 * self.lambda_(t) * t))
            )
            # According to https://quant.stackexchange.com/questions/38739/how-to-get-set-the-theta-function-in-the-hull-white-model-to-replicate-the-curre
            # , we estimate theta(t) * kappa (under "Slightly rewriting your SDE") => need to divide by kappa <=> divide by lambda in this notation
            mu_bootstrapped /= self.lambda_(t)
            mu.append(mu_bootstrapped)

            fwd_rate_old = fwd_rate
            t_old = t

        self._mu_curve = (
            pd.DataFrame(
                mu,
                index=yield_curve.instant_fwd_rate.index[1:],
                columns=[self.MU_COLUMN],
            )
            .reset_index()
            .to_numpy()
        )

    def fit(self, **kwargs) -> None:
        self.fit_regression(kwargs.get("data"))
        self.calibrate_mu(kwargs.get("yield_curve"))
