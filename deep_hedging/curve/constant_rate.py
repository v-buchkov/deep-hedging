import numpy as np

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.base.frequency import Frequency
from deep_hedging.config.global_config import GlobalConfig


class ConstantRateCurve(YieldCurve):
    def __init__(
        self,
        constant_rate: float,
        currency: str = None,
        create_curve_only: bool = False,
        compounding_frequency: Frequency = Frequency.CONTINUOUS,
        initial_terms: np.array = np.linspace(
            1 / GlobalConfig.CALENDAR_DAYS,
            GlobalConfig.YEARS_IN_CURVE,
            GlobalConfig.N_POINTS_CURVE,
        ),
    ) -> None:
        self.fixed_rate = constant_rate

        super().__init__(
            initial_terms=initial_terms,
            currency=currency,
            compounding_frequency=compounding_frequency,
            create_curve_only=create_curve_only,
        )

    def get_rates(self, terms: list[float]) -> np.array:
        return np.array([self.fixed_rate] * len(terms))
