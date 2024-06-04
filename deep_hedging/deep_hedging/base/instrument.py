import abc

import numpy as np

from .position_side import PositionSide
from .structured_note import StructuredNote


class Instrument:
    CALENDAR_DAYS: int = 365

    def __init__(self):
        pass

    @staticmethod
    def discount_factor(rate: float, term: float) -> float:
        return np.exp(-rate * term)

    @abc.abstractmethod
    def coupon(self, frequency: float = 0., *args, **kwargs) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def pv_coupons(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def payoff(self, spot: [np.array, float]) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        return StructuredNote([(PositionSide.LONG, self), (PositionSide.LONG, other)])

    def __sub__(self, other):
        return StructuredNote([(PositionSide.LONG, self), (PositionSide.SHORT, other)])
