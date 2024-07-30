from abc import abstractmethod

import numpy as np
import pandas as pd

from deep_hedging.utils.plot_rates import plot_rates


class InterestRateSimulator:
    CALENDAR_DAYS = 365
    DEFAULT_TEST_LENGTH = 0.25

    def __init__(
            self,
            target_column: str = "close",
            n_paths: int = 100,
            random_seed: [int, None] = None,
    ) -> None:
        self.target_column = target_column
        self.n_paths = n_paths
        self.random_seed = random_seed

        self.model = None
        self._sigma = None

    @abstractmethod
    def fit_regression(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def get_simulation_drift(self, r0: float, terms: list[float]) -> np.array:
        pass

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        pass

    @property
    def sigma(self) -> float:
        if self._sigma is None:
            raise ValueError("Model error is not calculated yet!")
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        self._sigma = sigma

    def simulate(self, r0: float, terms: list[float]) -> np.array:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        paths = []
        for _ in range(self.n_paths):
            path = [r0]
            for _ in terms:
                path.append(
                    (
                            self.get_simulation_drift(path[-1], terms=tuple([1] * 2))
                            + np.random.normal(scale=self.sigma)
                    ).tolist()[0]
                )
            paths.append(path[1:])
        return np.array(paths)

    def run_simulation(self, *args, **kwargs) -> None:
        if "data" in kwargs.keys():
            data = kwargs.get("data")
        elif len(args) > 0:
            data = args[0]
        else:
            raise ValueError("No arguments provided.")

        if "test_length" in kwargs.keys():
            test_length = kwargs.get("test_length")
        elif len(args) > 1:
            test_length = args[1]
        else:
            test_length = self.DEFAULT_TEST_LENGTH

        self.fit(*args, **kwargs)

        start_date = data.index[-1]
        dates = pd.date_range(
            start=start_date, periods=int(test_length * data.shape[0]), freq="1D"
        )

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        simulated = self.simulate(
            r0=data[self.target_column].iloc[-1],
            terms=tuple(
                ((dates.shift(1) - dates).days / self.CALENDAR_DAYS)
                .to_series()
                .cumsum()
                .to_list()
            ),
        )
        simulated = pd.DataFrame(
            simulated.T, columns=[self.target_column] * simulated.shape[0]
        )

        simulated["index"] = dates
        simulated.set_index("index", inplace=True)

        plot_rates(data[self.target_column], simulated)