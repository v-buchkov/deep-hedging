import numpy as np
import pandas as pd

from deep_hedging.monte_carlo.rates.vasicek_simulator import VasicekSimulator
from deep_hedging.curve.yield_curve import YieldCurve


class HullWhiteSimulator(VasicekSimulator):
    MU_COLUMN = "mu"

    def __init__(
        self,
        target_column: str = "close",
        random_seed: [int, None] = None,
    ) -> None:
        super().__init__(target_column=target_column, random_seed=random_seed)

        self._mu_curve = None

    def mu_(self, t: float) -> float:
        if self._mu_curve is None:
            raise ValueError(
                "Model is not calibrated to the market yet! Call .calibrate_mu(yield_curve) first."
            )
        index = np.absolute(self._mu_curve[:, 0] - t).argmin()
        return self._mu_curve[index, 1]

    def calibrate_mu(self, yield_curve: YieldCurve) -> None:
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
