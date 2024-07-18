from dataclasses import dataclass
import datetime as dt

import numpy as np

from deep_hedging.base.instrument import Instrument
from deep_hedging.base.frequency import Frequency
from deep_hedging.curve.yield_curve import YieldCurve
from deep_hedging.fixed_income.zero_coupon_bond import ZeroCouponBond
from deep_hedging.linear.forward import Forward
from deep_hedging.utils.fixing_dates import get_annual_indices

from deep_hedging.config.global_config import GlobalConfig


@dataclass
class CommodityBondSolution:
    zcb_body_weight: float
    zcb_coupon_weights: np.array
    fixed_coupon: float


class CommodityBond(Instrument):
    def __init__(
            self,
            yield_curve: YieldCurve,
            start_date: dt.datetime,
            end_date: dt.datetime,
            frequency: Frequency,
            yield_curve_commodity: YieldCurve,
    ):
        super().__init__()
        self.yield_curve = yield_curve
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency

        self.yield_curve_commodity = yield_curve_commodity

        self.time_till_maturity = (self.end_date - self.start_date).days / GlobalConfig.CALENDAR_DAYS

        self.factors = get_annual_indices(
            self.time_till_maturity,
            self.frequency,
        )

        self._no_arb_solution = self._solve_by_no_arbitrage()
        self.fixed_coupon = self._no_arb_solution.fixed_coupon
        self.portfolio = self._create_portfolio(self._no_arb_solution)

    def _create_coefficients_matrix(self, factors: np.array) -> np.array:
        k_0 = np.array([[1.] * len(factors) + [0.]])
        k_other = []
        for i, factor in enumerate(factors):
            line = [0.] * (len(factors) + 1)
            line[i] = (1 + self.yield_curve.get_rate(factor)) ** factor
            line[-1] = -(1 + self.yield_curve_commodity.get_rate(factor)) ** factor
            k_other.append(line)

        k_other = np.array(k_other)
        k = np.concatenate([k_0, k_other], axis=0)

        return k

    def _create_total_sums_vectors(self, factors: np.array):
        maturity_period = factors[-1]
        return np.array([1.] + [0.] * (len(factors) - 1) + [(1 + self.yield_curve_commodity.get_rate(maturity_period)) ** maturity_period])

    def _solve_by_no_arbitrage(self) -> CommodityBondSolution:
        coefficients = self._create_coefficients_matrix(self.factors)
        total_sums = self._create_total_sums_vectors(self.factors)

        x = np.linalg.solve(coefficients, total_sums)

        solution = CommodityBondSolution(
            zcb_body_weight=float(x[-2]),
            zcb_coupon_weights=x[:-2],
            fixed_coupon=float(x[-1]),
        )

        return solution

    def _create_portfolio(self, no_arb_solution: CommodityBondSolution):
        portfolio = ZeroCouponBond(
            yield_curve=self.yield_curve,
            start_date=self.start_date,
            end_date=self.end_date,
        ) * no_arb_solution.zcb_body_weight

        portfolio += Forward(
            yield_curve=self.yield_curve,
            start_date=self.start_date,
            end_date=self.end_date,
        ) * (1 + no_arb_solution.fixed_coupon)

        for i, point in enumerate(self.factors[:-1]):
            portfolio += (
                    ZeroCouponBond(
                        yield_curve=self.yield_curve,
                        start_date=self.start_date,
                        end_date=self.start_date + dt.timedelta(days=int(round(point * GlobalConfig.CALENDAR_DAYS))),
                    )
                    * no_arb_solution.zcb_coupon_weights[i]
            )
            portfolio += Forward(
                yield_curve=self.yield_curve,
                start_date=self.start_date,
                end_date=self.start_date + dt.timedelta(days=int(round(point * GlobalConfig.CALENDAR_DAYS))),
            ) * no_arb_solution.fixed_coupon

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
