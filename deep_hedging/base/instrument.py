import abc
from typing import Union
from copy import deepcopy

import numpy as np

from deep_hedging.base.position import Position, PositionSide


class Instrument:
    def __init__(self, *args, **kwargs):
        pass

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

    def __mul__(self, notional: Union[float, int]):
        if notional >= 0:
            return StructuredNote([(Position(PositionSide.LONG, size=notional), self)])
        else:
            return StructuredNote([(Position(PositionSide.SHORT, size=notional), self)])

    def __truediv__(self, notional: Union[float, int]):
        if notional >= 0:
            return StructuredNote(
                [(Position(PositionSide.LONG, size=1 / notional), self)]
            )
        else:
            return StructuredNote(
                [(Position(PositionSide.SHORT, size=1 / notional), self)]
            )


class StructuredNote:
    def __init__(
        self,
        instruments: [list[tuple[Position, Instrument]], None] = None,
        *args,
        **kwargs,
    ):
        self.instruments = []
        if instruments is not None:
            for position, instrument in instruments:
                if isinstance(instrument, StructuredNote):
                    sub_instruments = []
                    for internal_pos, internal_inst in instrument.instruments:
                        internal_pos *= position.size
                        sub_instruments.append((internal_pos, internal_inst))
                    self.instruments.extend(sub_instruments)
                else:
                    self.instruments.append((position, instrument))

        self._current = 0
        self._days_till_maturity = None

        self._scaled = 1

    def __add__(self, other: Instrument):
        if isinstance(other, StructuredNote):
            self.instruments.extend(other.instruments)
        else:
            self.instruments.append((Position(PositionSide.LONG), other))
        return self

    def __sub__(self, other: Instrument):
        if isinstance(other, StructuredNote):
            self.instruments.extend(
                [(position.invert(), instr) for position, instr in other.instruments]
            )
        else:
            self.instruments.append((Position(PositionSide.SHORT), other))
        return self

    def __mul__(self, notional: Union[float, int]):
        self._scaled *= notional
        for position, instrument in self.instruments:
            position.size *= notional
        return self

    def __truediv__(self, notional: Union[float, int]):
        self._scaled /= notional
        for position, instrument in self.instruments:
            position.size /= notional
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

    @property
    def days_till_maturity(self) -> int:
        if self._days_till_maturity is None:
            days = []
            for position, instrument in self.instruments:
                if hasattr(instrument, "days_till_maturity"):
                    days.append(instrument.days_till_maturity)
                else:
                    self._days_till_maturity = -1
                    return self._days_till_maturity
            self._days_till_maturity = max(days)
        return self._days_till_maturity

    def payoff(self, spot: np.array) -> np.array:
        assert (
            spot.ndim > 1
        ), f"If the array consists of one path only, use .reshape(1, -1).\nIf each path consists of one value only, use .reshape(-1, 1)."
        payoff = np.zeros((spot.shape[0], spot.shape[1] - 1))
        for position, instrument in self.instruments:
            if hasattr(instrument, "days_till_maturity"):
                days = instrument.days_till_maturity
            else:
                days = spot.shape[1] - 1
            payoff[:, days - 1] += (
                position.size
                * position.side.value
                * instrument.payoff(spot[:, : days + 1])
            )
        return payoff

    def __repr__(self):
        sp_str = f"StructuredNote of:\n"
        for i, (position, instrument) in enumerate(self.instruments):
            sp_str += f"{i + 1}. {position.side.name} {round(position.size, 4)} units of {instrument}.\n\n"
        return sp_str

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, item):
        return self.instruments[item]

    def __iter__(self):
        return iter(self.instruments)

    def __next__(self):
        return next(self)
