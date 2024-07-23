import numpy as np

from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.base.frequency import Frequency
from deep_hedging.config.global_config import GlobalConfig


class NelsonSiegelCurve(YieldCurve):
    def __init__(
        self,
        b0: float,
        b1: float,
        b2: float,
        tau: float,
        currency: str = None,
        create_curve_only: bool = False,
        compounding_frequency: Frequency = Frequency.CONTINUOUS,
        initial_terms: np.array = np.linspace(
            1 / GlobalConfig.CALENDAR_DAYS,
            GlobalConfig.YEARS_IN_CURVE,
            GlobalConfig.N_POINTS_CURVE,
        ),
    ) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.tau = tau

        super().__init__(
            initial_terms=initial_terms,
            currency=currency,
            compounding_frequency=compounding_frequency,
            create_curve_only=create_curve_only,
        )

    def get_rates(self, terms: list[float]) -> np.array:
        terms = np.array(terms)
        rates = (
            self.b0
            + (self.b1 + self.b2) * self.tau / terms * (1 - np.exp(-terms / self.tau))
            - self.b2 * np.exp(-terms / self.tau)
        )
        return rates / 100
