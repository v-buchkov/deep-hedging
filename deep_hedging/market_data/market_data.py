import datetime as dt
from functools import lru_cache

import numpy as np

import pandas_datareader.data as reader
import yfinance as yfin

from .tickers import Tickers
from deep_hedging.config.global_config import GlobalConfig

yfin.pdr_override()


class MarketData:
    TARGET_COLUMN = "Adj Close"

    def __init__(self, tickers: Tickers, start: dt.datetime, end: dt.datetime, sampling_period: str = "D"):
        self.df = None
        self.sampling_period = sampling_period

        self.tickers = tickers
        self.end = end

        self.start = start

        self._initialize()

    def __len__(self):
        return len(self.tickers)

    def _load_yahoo(self) -> None:
        self.df = reader.get_data_yahoo(self.tickers.codes, self.start, self.end)[self.TARGET_COLUMN]

    def _resample_data(self) -> None:
        if self.df is None:
            self._load_yahoo()
        self.df = self.df.resample(self.sampling_period).first().dropna(axis=0)
        self._df_returns = self.df.pct_change(fill_method=None).dropna(axis=0)

    def _initialize(self) -> None:
        self._load_yahoo()
        self._resample_data()

    def plot(self) -> None:
        n_stocks = len(self._df_returns.columns)

        ax = self._df_returns.stack().reset_index().rename(columns={0: "return"}).hist(column="return", by="Ticker",
                                                                                       range=[self.df.min().min(),
                                                                                              self.df.max().max()],
                                                                                       bins=100,
                                                                                       grid=False, figsize=(16, 16),
                                                                                       layout=(n_stocks, 1),
                                                                                       sharex=True,
                                                                                       color='#86bf91', zorder=2,
                                                                                       rwidth=0.9)

        for i, x in enumerate(ax):
            x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
                          labelleft="on")

            vals = x.get_yticks()
            for tick in vals:
                x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

            x.set_xlabel(f"Daily Return ({self.start.year}-{self.end.year})", labelpad=20, weight='bold', size=16)

            x.set_title(f"{self.tickers[self.df.columns[i]]}", size=12)

            if i == n_stocks // 2:
                x.set_ylabel("Frequency", labelpad=50, weight='bold', size=12)

            x.tick_params(axis='x', rotation=0)

    def __getitem__(self, item: [int, str], *args, **kwargs):
        if isinstance(item, int):
            return self.df.iloc[:, item].values
        elif isinstance(item, str):
            if item in self.df.columns:
                return self.df.loc[:, item].values
            else:
                return self.df.loc[:, self.tickers[item]].values
        else:
            raise TypeError(f"Item {item} is not a valid ticker or index")

    def get_means(self) -> np.array:
        return self._df_returns.mean().to_numpy() * GlobalConfig.TRADING_DAYS

    def get_var_covar(self) -> np.array:
        return self._df_returns.cov().to_numpy() * GlobalConfig.TRADING_DAYS

    def get_corr(self) -> np.array:
        return self._df_returns.corr().to_numpy()

    @lru_cache(maxsize=None)
    def get_dividends(self) -> np.array:
        return np.array([yfin.Ticker(ticker).dividends.iloc[-1] / 100 for ticker in self.tickers.codes])
