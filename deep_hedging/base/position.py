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
