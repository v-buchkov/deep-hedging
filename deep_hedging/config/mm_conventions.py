from deep_hedging.base.currency import Currency


class DiscountingConvention:
    def __init__(self, numerator: [int, str] = "ACT", denominator: [int, str] = "ACT"):
        self.numerator = numerator
        self.denominator = denominator

    def __call__(self, days: int) -> float:
        return days / self.denominator


class DiscountingConventions:
    def __init__(self):
        self.dict = {
            Currency.RUB: DiscountingConvention("ACT", 365),
            Currency.USD: DiscountingConvention("ACT", 360),
            Currency.EUR: DiscountingConvention("ACT", 360),
        }

    def __getitem__(self, currency: Currency) -> DiscountingConvention:
        return self.dict[currency]
