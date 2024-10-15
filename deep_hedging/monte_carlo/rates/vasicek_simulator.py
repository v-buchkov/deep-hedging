import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from deep_hedging.monte_carlo.rates.interest_rate_simulator import InterestRateSimulator


class VasicekSimulator(InterestRateSimulator):
    def __init__(
        self,
        target_column: str = "close",
        random_seed: [int, None] = None,
    ) -> None:
        super().__init__(target_column)

        self.target_column = target_column

        self.random_seed = random_seed

        self.dt = None
        self._r2_fitted = None

    def _create_lags(
        self, data: pd.DataFrame, n_lags: int
    ) -> tuple[pd.DataFrame, list[str]]:
        # lag_columns = []
        # for lag in range(1, n_lags + 1):
        #     column_name = f"{self.target_column}_t-{lag}"
        #     data[column_name] = tonia_df[self.target_column].shift(lag)
        #     lag_columns.append(column_name)
        # return data.iloc[n_lags:, :], lag_columns
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

    def fit_regression(self, data: pd.DataFrame, n_lags: int = 1) -> None:
        assert n_lags == 1, "Only one lag can be fitted right now."

        data = self._resample_to_daily(data)
        data, lag_columns = self._create_lags(data, n_lags=n_lags)

        X = data[lag_columns].to_numpy()
        y = data[self.target_column].to_numpy()

        if n_lags == 1:
            X = X.reshape(-1, 1)

        # Linreg by analytical solution => no need to fix random seed
        self.model = LinearRegression()
        self.model.fit(X, y)

        self.dt = (
            data.index.to_series() - data.index.to_series().shift(1)
        ).dt.days.min() / self.CALENDAR_DAYS
        self.sigma = np.std(y) * np.sqrt(self.dt)
        self._r2_fitted = self.model.score(X, y)

    def get_simulation_drift(self, r0: float, terms: list[float]) -> np.array:
        if self.model is None:
            raise ValueError("Regression is not fitted yet!")
        rates = [r0]
        t_old = terms[0]
        for t in terms[1:]:
            dr = self.lambda_(t) * (self.mu_(t) - rates[-1]) * (t - t_old)
            rates.append(rates[-1] + dr)
            t_old = t
        return np.array(rates[1:])

    def fit(self, *args, **kwargs) -> None:
        if "data" in kwargs.keys():
            self.fit_regression(kwargs.get("data"))
        elif len(args) > 0:
            self.fit_regression(args[0])
        else:
            raise ValueError("No arguments provided.")
