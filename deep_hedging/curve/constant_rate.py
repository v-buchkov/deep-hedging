import numpy as np

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.config.global_config import GlobalConfig


class ConstantRateCurve(YieldCurve):
    def __init__(
        self,
        rate: float,
        initial_terms: np.array = np.linspace(
            1 / GlobalConfig.CALENDAR_DAYS,
            GlobalConfig.YEARS_IN_CURVE,
            GlobalConfig.N_POINTS_CURVE,
        ),
    ) -> None:
        self.rate = rate

        super().__init__(initial_terms)

    def get_rates(self, terms: list[float]) -> np.array:
        return np.array([self.rate] * len(terms))
