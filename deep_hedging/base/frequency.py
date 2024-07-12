from enum import Enum


class Frequency(Enum):
    DAILY = 1 / 365
    WEEKLY = 1 / 52
    MONTHLY = 1 / 12
    QUARTERLY = 1 / 4
    SEMIANNUALLY = 1 / 2
    ANNUALLY = 1.0
