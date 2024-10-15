import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from deep_hedging.monte_carlo.volatility.volatility_simulator import VolatilitySimulator

from deep_hedging.config.global_config import GlobalConfig


class VasicekVolatilitySimulator(VolatilitySimulator):
    def __init__(
        self,
        rolling_days: int = GlobalConfig.VOLATILITY_ROLLING_DAYS,
        target_column: str = "close",
        random_seed: [int, None] = None,
    ) -> None:
        super().__init__(random_seed=random_seed)

        self.rolling_days = rolling_days
        self.target_column = target_column

        self.random_seed = random_seed

        self.model = None
        self.sigma = None
        self.dt = None
        self._r2_fitted = None

    def _create_lags(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        data["close_t-1"] = data[self.target_column].shift(1)
        return data.iloc[1:], ["close_t-1"]

    @property
    def r2_score(self) -> float:
        if self._r2_fitted is None:
            raise ValueError("Regression is not fitted yet!")
        return self._r2_fitted

    @staticmethod
    def _resample_to_daily(data: pd.DataFrame) -> pd.DataFrame:
        return data.resample("1D").ffill()

    def lambda_(self, *args, **kwargs) -> float:
        if self.model is None:
            raise ValueError("Regression is not fitted yet!")
        return (1 - self.model.coef_[0]) / self.dt

    def mu_(self, *args, **kwargs) -> float:
        if self.model is None:
            raise ValueError("Regression is not fitted yet!")
        return self.model.intercept_ / (1 - self.model.coef_[0])

    def fit_regression(self, data: pd.DataFrame) -> None:
        data = self._resample_to_daily(data)
        data = data.rolling(window=self.rolling_days, min_periods=1).std()
        data.dropna(inplace=True)
        data, lag_columns = self._create_lags(data)

        features = data[lag_columns].to_numpy().reshape(-1, 1)
        y = data[self.target_column].to_numpy()

        # Linreg by analytical solution => no need to fix random seed
        self.model = LinearRegression()
        self.model.fit(features, y)

        self.dt = (
            data.index.to_series() - data.index.to_series().shift(1)
        ).dt.days.min() / GlobalConfig.CALENDAR_DAYS
        self.sigma = np.std(y) * np.sqrt(self.dt)
        self._r2_fitted = self.model.score(features, y)

    def get_paths(self, vol_start: list[float], terms: np.array, noise: np.array = None, n_paths: int = GlobalConfig.MONTE_CARLO_PATHS, *args, **kwargs) -> np.array:
        if self.model is None:
            raise ValueError("Regression is not fitted yet!")

        noise = np.random.normal(size=(n_paths, len(terms))) if noise is None else noise

        vols = [np.array([vol_start] * n_paths).reshape(n_paths, 1, len(vol_start))]
        t_old = terms[0]
        for i, t in enumerate(terms[1:]):
            dvol_drift = self.lambda_(t) * (self.mu_(t) - vols[-1]) * (t - t_old)
            dvol_diffusion = self.sigma * np.sqrt(vols[-1] * (t - t_old)) * noise[:, i]

            vols.append(vols[-1] + dvol_drift + dvol_diffusion)
            t_old = t

        vols = np.concatenate(vols, axis=1)
        vols = np.where(vols > 0, vols, 0)

        return vols

    def fit(self, *args, **kwargs) -> None:
        if "data" in kwargs.keys():
            self.fit_regression(kwargs.get("data"))
        elif len(args) > 0:
            self.fit_regression(args[0])
        else:
            raise ValueError("No arguments provided.")
