from dataclasses import dataclass
import datetime as dt

import numpy as np
import pandas as pd

from deep_hedging.base.instrument import Instrument
from deep_hedging.base.frequency import Frequency
from deep_hedging.curve.yield_curve import YieldCurve, YieldCurves
from deep_hedging.curve.fixed_maturity_mixin import FixedMaturityMixin
from deep_hedging.fixed_income.zero_coupon_bond import ZeroCouponBond
from deep_hedging.linear.forward import Forward
from deep_hedging.utils.fixing_dates import generate_schedule, days_from_schedule

from deep_hedging.config.mm_conventions import DiscountingConventions


@dataclass
class CommodityBondSolution:
    zcb_body_weight: float
    zcb_coupon_weights: np.array
    fixed_coupon: float


class CommodityBond(FixedMaturityMixin, Instrument):
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
        frequency: Frequency,
        yield_curve_commodity: YieldCurve,
        forward_yield_curves: [YieldCurves, None] = None,
    ):
        super().__init__(
            yield_curve, start_date, end_date, frequency, yield_curve_commodity
        )
        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

        self.yield_curve_commodity = yield_curve_commodity
        self.forward_yield_curves = forward_yield_curves

        self.schedule = generate_schedule(start_date, end_date, frequency)

        self._days = days_from_schedule(self.schedule)
        self._cumsum_days = np.cumsum(self._days)

        self._initialize()

    def _initialize(self):
        self._no_arb_solution = self._solve_by_no_arbitrage()
        self.fixed_coupon = self._no_arb_solution.fixed_coupon
        self.portfolio = self._create_portfolio(self._no_arb_solution)

    def substitute_schedule(self, new_schedule: list[pd.Timestamp]) -> None:
        self.schedule = new_schedule
        self._initialize()

    def _create_coefficients_matrix(self) -> np.array:
        coefs = np.zeros((len(self._days), len(self._days) + 1))
        for i, days in enumerate(self._cumsum_days):
            coefs[i, i] = self.yield_curve.fv_discount_factors(days)
            coefs[i, -1] = -self.yield_curve_commodity.fv_discount_factors(days)

        coefs = np.concatenate(
            [np.array([[1.0] * len(self._days) + [0.0]]), coefs], axis=0
        )
        return coefs

    def _create_total_sums_vectors(self):
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

    def _create_portfolio(self, no_arb_solution: CommodityBondSolution):
        portfolio = (
            ZeroCouponBond(
                yield_curve=self.yield_curve,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            * no_arb_solution.zcb_body_weight
        )

        portfolio += Forward(
            yield_curve=self.yield_curve,
            start_date=self.start_date,
            end_date=self.end_date,
        ) * (1 + no_arb_solution.fixed_coupon)

        for i, date in enumerate(self.schedule[1:-1]):
            portfolio += (
                ZeroCouponBond(
                    yield_curve=self.yield_curve,
                    start_date=self.start_date,
                    end_date=date,
                )
                * no_arb_solution.zcb_coupon_weights[i]
            )
            portfolio += (
                Forward(
                    yield_curve=self.yield_curve,
                    start_date=self.start_date,
                    end_date=date,
                )
                * no_arb_solution.fixed_coupon
            )

        return portfolio

    def pv_coupons(self) -> float:
        return self.portfolio.pv_coupons()

    def coupon(self, frequency: float = 0.0, *args, **kwargs) -> float:
        return self.fixed_coupon

    def price(self):
        return self.portfolio.price()

    def payoff(self, spot_paths: np.array) -> float:
        return self.portfolio.payoff(spot_paths)

    def __repr__(self):
        return self.portfolio.__repr__()
