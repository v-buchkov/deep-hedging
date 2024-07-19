from dataclasses import dataclass

from deep_hedging.base.currency import Currency

from deep_hedging.config.global_config import GlobalConfig


class DiscountingConvention:
    def __init__(self, numerator: [int, str] = "ACT", denominator: [int, str] = "ACT"):
        self.numerator = numerator
        self.denominator = denominator

    def __call__(self, days: int) -> float:
        return days / self.denominator


@dataclass
class DiscountingConventions:
    _dict = {
        Currency.RUB: DiscountingConvention("ACT", 365),
        Currency.USD: DiscountingConvention("ACT", 360),
        Currency.EUR: DiscountingConvention("ACT", 360),
    }

    def __getitem__(self, item) -> DiscountingConvention:
        if item is None:
            return DiscountingConvention("ACT", GlobalConfig.CALENDAR_DAYS)
        if item not in self._dict:
            raise KeyError(f"Currency unsupported: {item}")
        return self._dict[item]
