import datetime as dt
from enum import Enum

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from deep_hedging.config.global_config import GlobalConfig


def get_periods_indices(
    till_maturity: float, freq: [float, Enum]
) -> tuple[np.array, np.array]:
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
    return np.array(periods), np.array(indices).astype(int)


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
    return np.array(indices).astype(int)


def get_annual_indices(till_maturity: float, freq: [float, Enum]) -> np.array:
    if isinstance(freq, Enum):
        freq = freq.value

    points = np.linspace(freq, till_maturity, int(round(till_maturity / freq)))
    return points


def generate_schedule(
    start: dt.datetime, end: dt.datetime, freq: Enum
) -> list[dt.datetime]:
    if isinstance(freq, Enum):
        freq = freq.value

    till_maturity = (end - start).days
    points = np.linspace(
        0.0,
        till_maturity,
        int(round(till_maturity / GlobalConfig.CALENDAR_DAYS / freq)) + 1,
    )

    dates = []
    for point in points:
        dates.append(start + dt.timedelta(days=int(point)) + 0 * BDay())

    return dates


def days_from_schedule(schedule: list[dt.datetime]) -> list[int]:
    return (pd.Series(schedule) - pd.Series(schedule).shift(1)).iloc[1:].dt.days.values
