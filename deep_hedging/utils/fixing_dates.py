from enum import Enum

import numpy as np

from deep_hedging.config.global_config import GlobalConfig


def get_periods_indices(till_maturity: float, freq: [float, Enum]) -> tuple[np.array, np.array]:
    if isinstance(freq, Enum):
        freq = freq.value

    points = np.linspace(
        0.0, till_maturity, int(round(till_maturity * GlobalConfig.TRADING_DAYS))
    )
    periods = []
    indices = []
    if freq != 0:
        for f in np.arange(freq, till_maturity, freq):
            idx = (np.abs(points - f)).argmin()
            periods.append(f)
            indices.append(idx)
    indices.append(len(points))
    return np.array(periods), np.array(indices)


def get_indices(till_maturity: float, freq: [float, Enum]) -> np.array:
    if isinstance(freq, Enum):
        freq = freq.value

    points = np.linspace(
        0.0, till_maturity, int(round(till_maturity * GlobalConfig.TRADING_DAYS))
    )
    periods = []
    indices = []
    if freq != 0:
        for f in np.arange(freq, till_maturity, freq):
            idx = (np.abs(points - f)).argmin()
            periods.append(f)
            indices.append(idx)
    indices.append(len(points))
    return np.array(indices)


def get_annual_indices(till_maturity: float, freq: [float, Enum]) -> np.array:
    if isinstance(freq, Enum):
        freq = freq.value

    points = np.linspace(
        freq, till_maturity, int(round(till_maturity / freq))
    )
    return points
