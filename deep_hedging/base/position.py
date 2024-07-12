from dataclasses import dataclass
from enum import Enum
from typing import Union


class PositionSide(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    side: PositionSide
    size: Union[float, int] = 1.0
