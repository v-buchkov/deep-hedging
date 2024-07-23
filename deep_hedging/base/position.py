from dataclasses import dataclass
from enum import Enum
from typing import Union


class PositionSide(Enum):
    LONG = 1
    SHORT = -1

    def invert(self):
        if self == PositionSide.LONG:
            return PositionSide.SHORT
        else:
            return PositionSide.LONG


@dataclass
class Position:
    side: PositionSide
    size: Union[float, int] = 1.0

    def __mul__(self, other_size: Union[float, int]):
        new_size = self.size * other_size
        if new_size >= 0:
            return Position(PositionSide.LONG, new_size)
        else:
            return Position(PositionSide.SHORT, -new_size)

    def __truediv__(self, other_size: Union[float, int]):
        new_size = self.size / other_size
        if new_size >= 0:
            return Position(PositionSide.LONG, new_size)
        else:
            return Position(PositionSide.SHORT, -new_size)
