import numpy as np

from deep_hedging.base import Instrument


class EuropeanCall(Instrument):
    def __init__(
            self,
            rates_difference: float,
            spot_price: float,
            term: float
    ):
        super().__init__()
        self.rates_difference = rates_difference
        self.spot_price = spot_price
        self.term = term

    def coupon(self, frequency: float = 0., *args, **kwargs) -> float:
        return 0

    def pv_coupons(self) -> float:
        return 0

    def get_strike(self, spot_price: [float, None] = None) -> float:
        return self.spot_price

    @property
    def strike(self) -> float:
        return self.get_strike()

    def payoff(self, spot: [float, np.array]) -> float:
        if isinstance(spot, float):
            final_fixing = spot
        else:
            assert spot.ndim == 1, f"You should pass exactly one spot path here, got {spot.ndim}-dim array"
            final_fixing = spot[-1]
        return max(final_fixing - self.strike, 0)

    def __repr__(self):
        return f"EuropeanCall(strike={self.strike}, term={self.term}, spot_ref={self.spot_price})"


class EuropeanPut(Instrument):
    def __init__(
            self,
            rates_difference: float,
            spot_price: float,
            term: float
    ):
        super().__init__()
        self.rates_difference = rates_difference
        self.spot_price = spot_price
        self.term = term

    def coupon(self, frequency: float = 0., *args, **kwargs) -> float:
        return 0

    def pv_coupons(self) -> float:
        return 0

    def get_strike(self, spot_price: [float, None] = None) -> float:
        return self.spot_price

    @property
    def strike(self) -> float:
        return self.get_strike()

    def payoff(self, spot: [float, np.array]) -> float:
        if isinstance(spot, float):
            final_fixing = spot
        else:
            assert spot.ndim == 1, f"You should pass exactly one spot path here, got {spot.ndim}-dim array"
            final_fixing = spot[-1]
        return max(self.strike - final_fixing, 0)

    def __repr__(self):
        return f"EuropeanPut(strike={self.strike}, term={self.term}, spot_ref={self.spot_price})"
