import numpy as np

from deep_hedging.base import Instrument


class ForwardOld(Instrument):
    def __init__(
        self, rates_difference: float, spot_price: float, term: float, *args, **kwargs
    ):
        super().__init__()
        self.rates_difference = rates_difference
        self.spot_price = spot_price
        self.term = term

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        return 0

    def pv_coupons(self) -> float:
        return 0

    def get_strike(self, spot_price: [float, None] = None) -> float:
        if spot_price is None:
            spot_price = self.spot_price
        return spot_price * self.discount_factor(
            rate=-self.rates_difference, term=self.term
        )

    @property
    def strike(self) -> float:
        return self.get_strike()

    def payoff(self, spot: [float, np.array]) -> float:
        return spot[:, -1] - self.strike

    def __repr__(self):
        return f"Forward(strike={self.strike}, term={self.term}, spot_ref={self.spot_price})"
