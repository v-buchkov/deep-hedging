import numpy as np


class VolatilitySmile:
    def __init__(self, strikes: np.array, volatilities: np.array):
        self.strikes = strikes
        self.volatilities = volatilities

        self._data_dict = {}
        for i, strike in enumerate(strikes):
            self._data_dict[strike] = volatilities[i]

    def __getitem__(self, strike: float) -> float:
        return self._data_dict[strike]


class VolatilitySurface:
    def __init__(self, times: np.array, strikes: np.array, volatilities: np.array):
        self.times = times
        self.strikes = strikes
        self.volatilities = volatilities

        self._data_dict = {}
        for time in times:
            self._data_dict[time] = VolatilitySmile(self.strikes[time], self.volatilities[time, :].squeeze(0))

    def __getitem__(self, time: float, strike: float) -> float:
        return self._data_dict[time][strike]
