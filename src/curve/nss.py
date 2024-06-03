from yield_curve import YieldCurve

import numpy as np


class NelsonSiegelCurve(YieldCurve):
    def __init__(self, b0: float, b1: float, b2: float, tau: float,
                 initial_terms: np.array = np.linspace(1 / 365, 25., 100)) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.tau = tau

        super().__init__(initial_terms)

    def get_rates(self, terms: list[float]) -> np.array:
        terms = np.array(terms)
        rates = self.b0 + (self.b1 + self.b2) * self.tau / terms * (1 - np.exp(-terms / self.tau)) - self.b2 * np.exp(
            -terms / self.tau)
        return rates / 100
