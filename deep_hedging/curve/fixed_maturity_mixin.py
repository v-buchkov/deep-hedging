import datetime as dt

import pandas as pd
from pandas.tseries.offsets import BDay

from deep_hedging.curve.yield_curve import YieldCurve

from deep_hedging.config.mm_conventions import DiscountingConventions


class FixedMaturityMixin:
    def __init__(
        self,
        yield_curve: YieldCurve,
        start_date: dt.datetime,
        end_date: dt.datetime,
        *args,
        **kwargs
    ):
        super().__init__(
            yield_curve=yield_curve,
            start_date=start_date,
            end_date=end_date,
            *args,
            **kwargs
        )

        self.start_date = start_date + 0 * BDay()
        self.end_date = end_date + 0 * BDay()

        self.days_till_maturity = (self.end_date - self.start_date).days
        self.time_till_maturity = DiscountingConventions()[yield_curve.currency](
            self.days_till_maturity
        )
