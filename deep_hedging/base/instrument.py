import abc

import numpy as np

from .position_side import PositionSide


class Instrument:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def discount_factor(rate: float, term: float) -> float:
        return np.exp(-rate * term)

    @abc.abstractmethod
    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
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


class StructuredNote:
    def __init__(
        self, instruments: [list[tuple[PositionSide, Instrument]], None] = None
    ):
        if instruments is not None:
            self.instruments = instruments
        else:
            self.instruments = []

    def coupon(
        self, frequency: float = 0.0, commission: float = 0.0, *args, **kwargs
    ) -> float:
        return sum(
            [
                instrument.coupon(frequency, commission)
                for _, instrument in self.instruments
            ]
        )

    def __add__(self, other: Instrument):
        return self.instruments.append((PositionSide.LONG, other))

    def __sub__(self, other: Instrument):
        return self.instruments.append((PositionSide.SHORT, other))

    # def price(self) -> float:
    #     return sum([side.value * instrument.price() + instrument.pv_coupons() for side, instrument in self.instruments])

    def payoff(self, spot_paths: np.array) -> float:
        return sum(
            [
                side.value * instrument.payoff(spot_paths)
                for side, instrument in self.instruments
            ]
        )

    def __repr__(self):
        sp_str = f"StructuredNote of:\n"
        for side, instrument in self.instruments:
            sp_str += f"* {side} -> {instrument}\n"
        return sp_str

    def __str__(self):
        return self.__repr__()
