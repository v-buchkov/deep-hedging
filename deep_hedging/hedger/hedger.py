from enum import Enum

import numpy as np

from deep_hedging.base import Instrument, StructuredNote


class HedgingMode(Enum):
    DELTA = "delta"
    NEURAL = "neural"


class Hedger:
    def __init__(
        self,
        instrument: [Instrument, StructuredNote],
        hedging_mode: str = None,
        look_ahead: bool = False,
    ):
        self.instrument = instrument

        if hedging_mode is None:
            self.hedging_mode = HedgingMode.DELTA
        else:
            self.hedging_mode = HedgingMode[hedging_mode]

        self.look_ahead = look_ahead

    @staticmethod
    def calc_pnl(
        weights: np.array,
        bids: np.array,
        asks: np.array,
        rates_borrow: np.array,
        rates_lend: np.array,
    ) -> tuple[np.array, np.array]:
        assert len(bids) == len(
            asks
        ), f"Bid-Ask shapes mismatch ({len(bids)} != {len(asks)})"

        weights_all = np.concatenate(
            [
                np.zeros((bids.shape[0], 1)),
                weights,
                np.zeros((bids.shape[0], 1)),
            ],
            axis=1,
        )
        weights_diff = np.diff(weights_all, n=1, axis=1)

        bought = np.where(weights_diff > 0, weights_diff, 0)
        sold = np.where(weights_diff < 0, -weights_diff, 0)

        cash_outflow = -asks * bought
        cash_inflow = bids * sold

        cash_position = cash_outflow.cumsum(axis=1) + cash_inflow.cumsum(axis=1)

        rates = np.where(cash_position[:, :-1] > 0, rates_lend, rates_borrow)
        interest = (rates * cash_position[:, :-1]).sum(axis=1)

        return cash_outflow.sum(axis=1) + cash_inflow.sum(axis=1), interest

    def _get_weights_path(self, mid: np.array) -> [np.array, None]:
        if isinstance(self.instrument, StructuredNote):
            instruments = self.instrument.instruments
        else:
            instruments = [self.instrument]

        all_weights = []
        for instrument in instruments:
            if hasattr(instrument, "time_till_maturity") and hasattr(
                instrument, self.hedging_mode.value
            ):
                till_maturity = np.linspace(
                    instrument.time_till_maturity, 1e-6, mid.shape[1]
                )

                hedging_fn = getattr(instrument, self.hedging_mode.value)

                weights_path = []
                for day in range(mid.shape[1]):
                    weights_path.append(
                        list(
                            hedging_fn(
                                spot=mid[:, day], till_maturity=till_maturity[day]
                            ).flatten()
                        )
                    )
                all_weights.append(weights_path)

        return np.array(all_weights).T.sum(axis=2)

    def simulate(
        self,
        bids: np.array,
        asks: np.array,
        rates_borrow: np.array,
        rates_lend: np.array,
        weights: [np.array, None] = None,
    ) -> [np.array, np.array]:
        mid = (bids + asks) / 2

        if weights is None:
            weights = self._get_weights_path(mid)

        if self.look_ahead:
            weights = weights[:, :-1]
        else:
            weights = weights[:, 1:]

        spot_pnl, interest = self.calc_pnl(
            weights=weights,
            bids=bids,
            asks=asks,
            rates_borrow=rates_borrow,
            rates_lend=rates_lend,
        )

        payoff = self.instrument.payoff(mid)

        return spot_pnl, interest, payoff

    def price(
        self,
        bids: np.array,
        asks: np.array,
        rates_borrow: np.array,
        rates_lend: np.array,
    ) -> float:
        spot_pnl, interest, payoff = self.simulate(
            bids=bids,
            asks=asks,
            rates_borrow=rates_borrow,
            rates_lend=rates_lend,
        )
        return np.mean(payoff - spot_pnl - interest)

    def std(
        self,
        bids: np.array,
        asks: np.array,
        rates_borrow: np.array,
        rates_lend: np.array,
    ) -> float:
        pnl, payoff = self.simulate(
            bids=bids,
            asks=asks,
            rates_borrow=rates_borrow,
            rates_lend=rates_lend,
        )
        return np.std(pnl - payoff)
