import numpy as np

from deep_hedging.config.global_config import GlobalConfig


def get_periods_indices(till_maturity: float, freq: float) -> np.array:
    periods = np.linspace(
        0.0, till_maturity, int(round(till_maturity * GlobalConfig.TRADING_DAYS))
    )
    indices = []
    if freq != 0:
        for f in np.arange(freq, till_maturity, freq):
            idx = (np.abs(periods - f)).argmin()
            indices.append(idx)
    indices.append(len(periods))
    return np.array(indices)
