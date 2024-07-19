import datetime as dt

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
        super().__init__(yield_curve=yield_curve, start_date=start_date, end_date=end_date, *args, **kwargs)
        self.days_till_maturity = (end_date - start_date).days
        self.time_till_maturity = DiscountingConventions()[yield_curve.currency](self.days_till_maturity)
