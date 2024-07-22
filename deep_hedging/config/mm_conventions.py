from dataclasses import dataclass

import numpy as np

from deep_hedging.base.currency import Currency

from deep_hedging.config.global_config import GlobalConfig


class DiscountTerm:
    def __init__(self, numerator: [int, str] = "ACT", denominator: [int, str] = "ACT"):
        self.numerator = numerator
        self.denominator = denominator

    def __call__(self, days: int) -> float:
        return days / self.denominator


@dataclass
class DiscountingConventions:
    _dict = {
        Currency.RUB: DiscountTerm("ACT", 365),
        Currency.USD: DiscountTerm("ACT", 360),
        Currency.EUR: DiscountTerm("ACT", 360),
    }

    def __getitem__(self, item) -> DiscountTerm:
        if item is None:
            return DiscountTerm("ACT", GlobalConfig.CALENDAR_DAYS)
        if item not in self._dict:
            raise KeyError(f"Currency unsupported: {item}")
        return self._dict[item]
