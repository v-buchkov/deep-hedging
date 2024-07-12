import abc
from functools import lru_cache

import numpy as np
import pandas as pd

from deep_hedging.base import Currency

from deep_hedging.config.global_config import GlobalConfig


class YieldCurve:
    def __init__(
        self,
        initial_terms: np.array,
        currency: str = None,
        create_curve_only: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.currency = Currency[currency]

        self._rates_df = None
        self._discount_factors = None
        self._instant_fwd_rate = None

        self.create_curve_only = create_curve_only

        self._initialize(initial_terms)

    def _initialize(self, terms: np.array) -> None:
        self.create_curve(terms=terms)

    @abc.abstractmethod
    def get_rates(self, terms: list[float]) -> np.array:
        pass

    def create_curve(self, terms: list[float]) -> None:
        self._rates_df = pd.DataFrame(
            self.get_rates(terms), index=terms, columns=[GlobalConfig.TARGET_COLUMN]
        )

        if not self.create_curve_only:
            self._create_discount_factors()
            self._create_instant_fwd_rates()

    def _create_discount_factors(self) -> pd.DataFrame:
        if self._rates_df is None:
            raise ValueError("Rate data is not fitted yet!")
        discount_factors = np.exp(
            -self._rates_df[GlobalConfig.TARGET_COLUMN] * self._rates_df.index
        )
        self._discount_factors = pd.DataFrame(
            discount_factors,
            index=self._rates_df.index,
            columns=[GlobalConfig.DISCOUNT_FACTOR_COLUMN],
        )
        return self._discount_factors

    def _create_instant_fwd_rates(self) -> pd.DataFrame:
        if self._discount_factors is None:
            raise ValueError("Discount factor data is not fitted yet!")

        t_old = self._rates_df.index[0]
        instant_fwd_rates = []
        for t in self._rates_df.index[1:]:
            dt = t - t_old
            instant_fwd_rates.append(
                -1
                / dt
                * (
                    np.log(
                        self._discount_factors.loc[
                            t, GlobalConfig.DISCOUNT_FACTOR_COLUMN
                        ]
                        / self._discount_factors.loc[
                            t_old, GlobalConfig.DISCOUNT_FACTOR_COLUMN
                        ]
                    )
                )
            )
            t_old = t
        self._instant_fwd_rate = pd.DataFrame(
            instant_fwd_rates,
            index=self._rates_df.index[1:],
            columns=[GlobalConfig.FWD_RATE_COLUMN],
        )
        return self._instant_fwd_rate

    @property
    def curve_df(self) -> pd.DataFrame:
        if self._rates_df is None:
            raise ValueError("Rate data is not fitted yet! Call .create_curve() first.")
        return self._rates_df

    @property
    def discount_factors_df(self) -> pd.DataFrame:
        return self._discount_factors

    @property
    def instant_fwd_rates_df(self) -> pd.DataFrame:
        return self._instant_fwd_rate

    @staticmethod
    def _find_point(curve: pd.DataFrame, term: float) -> float:
        index = np.absolute(curve.index - term).argmin()
        return curve.iloc[index].values[0]

    @lru_cache(maxsize=None)
    def get_rate(self, term: float) -> float:
        return self._find_point(self._rates_df, term)

    @lru_cache(maxsize=None)
    def get_discount_factor(self, term: float) -> float:
        return self._find_point(self._discount_factors, term)

    @lru_cache(maxsize=None)
    def get_instant_fwd_rate(self, term: float) -> float:
        return self._find_point(self._instant_fwd_rate, term)


class YieldCurves:
    def __init__(self, curves: list[YieldCurve]):
        self.curves_list = curves

        self._initialize()

    def _initialize(self):
        self._curve_dict = self._get_curve_dict(self.curves_list)

    @staticmethod
    def _get_curve_dict(curves: list[YieldCurve]) -> dict[Currency, YieldCurve]:
        return {curve.currency: curve for curve in curves}

    def __len__(self):
        return len(self.curves_list)

    def __getitem__(self, item) -> YieldCurve:
        return self._curve_dict[Currency[item]]

    def __add__(self, other):
        return YieldCurves(self.curves_list + other.curves_list)

    def __iter__(self):
        return iter(self.curves_list)
