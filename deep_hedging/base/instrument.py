import abc
from typing import Union

import numpy as np

from deep_hedging.base.position import Position, PositionSide


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
        return StructuredNote(
            [(Position(PositionSide.LONG), self), (Position(PositionSide.LONG), other)]
        )

    def __sub__(self, other):
        return StructuredNote(
            [(Position(PositionSide.LONG), self), (Position(PositionSide.SHORT), other)]
        )

    def __mul__(self, notional):
        return StructuredNote([(Position(PositionSide.LONG, size=notional), self)])


class StructuredNote:
    def __init__(self, instruments: [list[tuple[Position, Instrument]], None] = None):
        if instruments is not None:
            self.instruments = instruments
        else:
            self.instruments = []

    def __add__(self, other: Instrument):
        if isinstance(other, StructuredNote):
            self.instruments.extend(other.instruments)
        else:
            self.instruments.append((Position(PositionSide.LONG), other))
        return self

    def __sub__(self, other: Instrument):
        if isinstance(other, StructuredNote):
            self.instruments.extend([(position.invert(), instr) for position, instr in other.instruments])
        else:
            self.instruments.append((Position(PositionSide.SHORT), other))
        return self

    def __mul__(self, notional: Union[float, int]):
        for position, instrument in self.instruments:
            position.size *= notional
        return self

    def coupon(
        self, frequency: float = 0.0, commission: float = 0.0, *args, **kwargs
    ) -> float:
        return sum(
            [
                position.size
                * position.side.value
                * instrument.coupon(frequency, commission)
                for position, instrument in self.instruments
            ]
        )

    def price(self) -> float:
        return sum(
            [
                position.size * position.side.value * instrument.price()
                for position, instrument in self.instruments
            ]
        )

    def pv_coupons(self) -> float:
        return sum(
            [
                position.size * position.side.value * instrument.pv_coupons()
                for position, instrument in self.instruments
            ]
        )

    def payoff(self, spot_paths: np.array) -> float:
        return sum(
            [
                position.size * position.side.value * instrument.payoff(spot_paths)
                for position, instrument in self.instruments
            ]
        )

    def __repr__(self):
        sp_str = f"StructuredNote of:\n"
        for i, (position, instrument) in enumerate(self.instruments):
            sp_str += f"{i + 1}. {position.side.name} {round(position.size, 4)} units of {instrument}.\n\n"
        return sp_str

    def __str__(self):
        return self.__repr__()
