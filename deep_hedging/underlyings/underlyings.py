import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

import yfinance as yfin

from deep_hedging.underlyings.ticker import Ticker
from deep_hedging.underlyings.tickers import Tickers
from deep_hedging.utils.linalg import corr_matrix_from_cov
from deep_hedging.config.global_config import GlobalConfig


class Underlyings:
    TARGET_COLUMN = "Adj Close"

    def __init__(
        self,
        tickers: [list[Ticker], Tickers],
        start: dt.datetime,
        end: dt.datetime,
        sampling_period: str = "D",
        data: [pd.DataFrame, None] = None,
        dividends: [np.array, None] = None,
        means: [np.array, None] = None,
        var_covar: [np.array, None] = None,
    ):
        self.data = data
        self.dividends = dividends
        self.means = means
        self.var_covar = var_covar

        self.sampling_period = sampling_period

        if isinstance(tickers, list):
            self.tickers = Tickers(tickers)
        else:
            self.tickers = tickers
        self.end = end

        self.start = start

        self._initialize()

    def __len__(self):
        return len(self.tickers)

    def _initialize(self) -> None:
        if self.means is None or self.var_covar is None:
            if self.data is None:
                self._load_yahoo()
            self._resample_data()

    def _load_yahoo(self) -> None:
        self.data = yfin.download(self.tickers.codes, self.start, self.end)[
            self.TARGET_COLUMN
        ]
        if self.data.shape[1] != len(self.tickers.codes):
            for i, ticker in enumerate(self.tickers.codes):
                if i < self.data.shape[1]:
                    if self.data.columns[i] != ticker:
                        self.data.insert(
                            i, column=f"{ticker}_2", value=self.data.loc[:, ticker]
                        )
                else:
                    self.data.insert(
                        i, column=f"{ticker}_2", value=self.data.loc[:, ticker]
                    )

    def _resample_data(self) -> None:
        if self.data is None:
            self._load_yahoo()
        self.data = self.data.resample(self.sampling_period).first().dropna(axis=0)
        # self.df_returns = self.data.pct_change(fill_method=None).dropna(axis=0)
        # self.df_returns = np.log(self.data / self.data.shift(1)).dropna(axis=0)
        self.df_returns = (self.data / self.data.shift(1) - 1).dropna(axis=0)

    def plot(self) -> None:
        n_stocks = len(self.df_returns.columns)

        ax = (
            self.df_returns.stack()
            .reset_index()
            .rename(columns={0: "return"})
            .hist(
                column="return",
                by="Ticker",
                range=[self.data.min().min(), self.data.max().max()],
                bins=100,
                grid=False,
                figsize=(16, 16),
                layout=(n_stocks, 1),
                sharex=True,
                color="#86bf91",
                zorder=2,
                rwidth=0.9,
            )
        )

        for i, x in enumerate(ax):
            x.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="on",
                left="off",
                right="off",
                labelleft="on",
            )

            vals = x.get_yticks()
            for tick in vals:
                x.axhline(
                    y=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1
                )

            x.set_xlabel(
                f"Daily Return ({self.start.year}-{self.end.year})",
                labelpad=20,
                weight="bold",
                size=16,
            )

            x.set_title(f"{self.tickers[self.data.columns[i]]}", size=12)

            if i == n_stocks // 2:
                x.set_ylabel("Frequency", labelpad=50, weight="bold", size=12)

            x.tick_params(axis="x", rotation=0)

    def __getitem__(self, item: [int, str], *args, **kwargs):
        if isinstance(item, int):
            return self.data.iloc[:, item].values
        elif isinstance(item, str):
            if item in self.data.columns:
                return self.data.loc[:, item].values
            else:
                return self.data.loc[:, self.tickers[item]].values
        else:
            raise TypeError(f"Item {item} is not a valid ticker or index")

    def get_means(self) -> np.array:
        if self.means is None:
            return self.df_returns.mean().to_numpy() * GlobalConfig.TRADING_DAYS
        return self.means

    def get_var_covar(self) -> np.array:
        if self.var_covar is None:
            return self.df_returns.cov().to_numpy() * GlobalConfig.TRADING_DAYS
        return self.var_covar

    def get_corr(self) -> np.array:
        if self.var_covar is None:
            return self.df_returns.corr().to_numpy()
        return corr_matrix_from_cov(self.var_covar)

    @lru_cache(maxsize=None)
    def get_dividends(self) -> np.array:
        if self.dividends is None:
            return np.array(
                [
                    yfin.Ticker(ticker).dividends.iloc[-1] / 100
                    for ticker in self.tickers.codes
                ]
            )
        return self.dividends

    def __repr__(self):
        s = "Underlyings:\n"
        for ticker in self.tickers:
            s += f"{ticker}\n"
        return s

    def __str__(self):
        return self.__repr__()
