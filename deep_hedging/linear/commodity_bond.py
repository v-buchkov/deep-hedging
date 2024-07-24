from dataclasses import dataclass
import datetime as dt

import numpy as np
import numpy_financial as npf
import pandas as pd

from deep_hedging.base.instrument import StructuredNote
from deep_hedging.base.frequency import Frequency
from deep_hedging.curve.yield_curve import YieldCurve, YieldCurves
from deep_hedging.fixed_income.zero_coupon_bond import ZeroCouponBond
from deep_hedging.linear.forward import Forward
from deep_hedging.utils.fixing_dates import generate_schedule, days_from_schedule


@dataclass
class CommodityBondSolution:
    zcb_body_weight: float
    zcb_coupon_weights: np.array
    fixed_coupon: float


class CommodityBond(StructuredNote):
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
        frequency: Frequency,
        yield_curve_commodity: YieldCurve,
        forward_yield_curves: [YieldCurves, None] = None,
    ):
        super().__init__(instruments=[])
        self.yield_curve = yield_curve
        self.frequency = frequency

        self.yield_curve_commodity = yield_curve_commodity
        self.forward_yield_curves = forward_yield_curves

        self.schedule = generate_schedule(start_date, end_date, frequency)
        self.start_date = pd.to_datetime(self.schedule[0])
        self.end_date = pd.to_datetime(self.schedule[-1])

        self._days_till_maturity = (self.end_date - self.start_date).days

        self._effective_yield = None

        self._initialize()

    def _initialize(self):
        self._days = days_from_schedule(self.schedule)
        self._cumsum_days = np.cumsum(self._days)

        self._no_arb_solution = self._solve_by_no_arbitrage()
        self.fixed_coupon = self._no_arb_solution.fixed_coupon
        self._create_instruments(self._no_arb_solution)

    def substitute_schedule(self, new_schedule: list[pd.Timestamp]) -> None:
        self.schedule = new_schedule
        self._initialize()

    def get_daily_payments(self, spot_fixings: np.array) -> np.array:
        assert (
            len(self._cumsum_days) + 1 == spot_fixings.shape[0]
        ), f"Instrument has {len(self._cumsum_days) + 1} fixings, while {spot_fixings.shape[0]} spot points were provided!"
        spot_fixings = spot_fixings / spot_fixings[0]
        spot = np.zeros((1, self.days_till_maturity + 1))
        for i, days in enumerate(self._cumsum_days):
            spot[:, days] = spot_fixings[i + 1]
        payments = self.payoff(spot)
        return payments

    def payments(self, spot_fixings: np.array) -> np.array:
        payments = self.get_daily_payments(spot_fixings)
        return payments[payments != 0]

    def _create_coefficients_matrix(self) -> np.array:
        coefs = np.zeros((len(self._days), len(self._days) + 1))
        for i, days in enumerate(self._cumsum_days):
            coefs[i, i] = self.yield_curve.fv_discount_factors(days)
            coefs[i, -1] = -self.yield_curve_commodity.fv_discount_factors(days)

        coefs = np.concatenate(
            [np.array([[1.0] * len(self._days) + [0.0]]), coefs], axis=0
        )
        return coefs

    def _create_total_sums_vectors(self) -> np.ndarray:
        maturity_days = self._cumsum_days[-1]
        return np.array(
            [1.0]
            + [0.0] * (len(self._days) - 1)
            + [self.yield_curve_commodity.fv_discount_factors(maturity_days)]
        )

    def _solve_by_no_arbitrage(self) -> CommodityBondSolution:
        coefficients = self._create_coefficients_matrix()
        total_sums = self._create_total_sums_vectors()

        x = np.linalg.solve(coefficients, total_sums)

        solution = CommodityBondSolution(
            zcb_body_weight=float(x[-2]),
            zcb_coupon_weights=x[:-2],
            fixed_coupon=float(x[-1]),
        )

        return solution

    def _create_instruments(self, no_arb_solution: CommodityBondSolution) -> None:
        instruments = (
            ZeroCouponBond(
                yield_curve=self.yield_curve,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            * self.yield_curve.to_future_value(1, self.days_till_maturity)
            * no_arb_solution.zcb_body_weight
        )

        instruments += Forward(
            yield_curve=self.yield_curve_commodity,
            start_date=self.start_date,
            end_date=self.end_date,
        ) * (1 + no_arb_solution.fixed_coupon)

        for i, date in enumerate(self.schedule[1:-1]):
            instruments += (
                ZeroCouponBond(
                    yield_curve=self.yield_curve,
                    start_date=self.start_date,
                    end_date=date,
                )
                * self.yield_curve.to_future_value(1, (date - self.start_date).days)
                * no_arb_solution.zcb_coupon_weights[i]
            )
            instruments += (
                Forward(
                    yield_curve=self.yield_curve_commodity,
                    start_date=self.start_date,
                    end_date=date,
                )
                * no_arb_solution.fixed_coupon
            )

        instruments = instruments * self._scaled
        self.instruments = instruments.instruments
        self._effective_yield = self.calc_effective_yield()

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        return self.fixed_coupon

    def calc_effective_yield(self) -> float:
        spot_fixings = np.zeros((1, self.days_till_maturity + 1))
        for _, instrument in self.instruments:
            if isinstance(instrument, Forward):
                spot_fixings[:, instrument.days_till_maturity] = instrument.strike
        payments = self.payoff(np.array(spot_fixings))
        _, payments_idx = np.where(payments != 0)
        discount_factors = self.yield_curve.pv_discount_factors(payments_idx + 1)
        payments = payments[:, payments_idx] * discount_factors
        return npf.irr(np.concatenate([[-self.price()], payments.squeeze(0)]))

    @property
    def effective_yield(self):
        if self._effective_yield is None:
            self._effective_yield = self.calc_effective_yield()
        return self._effective_yield
