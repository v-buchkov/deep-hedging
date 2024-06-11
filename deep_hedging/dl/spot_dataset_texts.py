import os
import datetime as dt
from typing import Union, Type
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from deep_hedging.base import Instrument
from deep_hedging.config.global_config import GlobalConfig
from deep_hedging.config.experiment_config import ExperimentConfig


class SpotDatasetTexts(Dataset):
    def __init__(
        self,
        n_days: int,
        instrument_cls: Type[Instrument],
        data: Union[pd.DataFrame, None] = None,
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.instrument_cls = instrument_cls
        self.n_days = n_days
        self.config = config

        self.df = (
            self._create_df(config.DATA_ROOT, config.DATA_FILENAME)
            if data is None
            else data.copy()
        )
        self.df = self._add_time_diff(self.df)
        self.df.drop(
            [GlobalConfig.TEXT_COLUMN, GlobalConfig.LEMMAS_COLUMN], axis=1, inplace=True
        )

        self.df = self.df.resample(config.REBAL_FREQ).ffill()
        self.df = self.df.dropna()

    @staticmethod
    def _create_df(path: Path, filename: str) -> pd.DataFrame:
        filename += ".pkl"
        file_path = path / filename

        if filename in os.listdir(path):
            return pd.read_pickle(file_path)
        else:
            raise FileNotFoundError

    @staticmethod
    def _add_time_diff(df: pd.DataFrame) -> pd.DataFrame:
        df[GlobalConfig.TIME_DIFF_COLUMN] = df.index.to_series().diff()

        df.loc[df.index[0], GlobalConfig.TIME_DIFF_COLUMN] = pd.to_timedelta(
            "0 days 00:00:00"
        )

        day_delta = np.timedelta64(1, "D") * GlobalConfig.CALENDAR_DAYS
        df[GlobalConfig.TIME_DIFF_COLUMN] = (
            df[GlobalConfig.TIME_DIFF_COLUMN].cumsum() / day_delta
        )

        return df

    def _create_instrument(self, period_df: pd.DataFrame) -> [Instrument, float]:
        start = period_df.loc[period_df.index.min()]
        spot_start = (
            start[GlobalConfig.BID_COLUMN] + start[GlobalConfig.ASK_COLUMN]
        ) / 2
        return (
            self.instrument_cls(
                rates_difference=start[GlobalConfig.RATE_DOMESTIC_COLUMN]
                - start[GlobalConfig.RATE_FOREIGN_COLUMN],
                spot_price=spot_start,
                term=self.n_days / GlobalConfig.CALENDAR_DAYS,
            ),
            spot_start,
        )

    def __len__(self):
        return len(
            self.df[
                self.df.index < self.df.index.max() - dt.timedelta(days=self.n_days)
            ]
        )

    def __getitem__(self, idx: int):
        start_date = self.df.index[idx]
        end_date = start_date + dt.timedelta(days=self.n_days)

        features = self.df[
            (self.df.index >= start_date) & (self.df.index <= end_date)
        ].copy()

        # Calculate time till maturity
        features[GlobalConfig.TIME_DIFF_COLUMN] = (
            features.iloc[-1, -1] - features[GlobalConfig.TIME_DIFF_COLUMN]
        )

        embeds = torch.stack(features[GlobalConfig.EMBEDDING_COLUMN].tolist())[
            :, : self.config.EMBED_MAX_DIM, :
        ]
        features.drop([GlobalConfig.EMBEDDING_COLUMN], axis=1, inplace=True)

        instrument, spot_start = self._create_instrument(features)
        features[GlobalConfig.SPOT_START_COLUMN] = spot_start

        target = instrument.payoff(spot=features.ask.iloc[-1])

        return (
            torch.Tensor(features.to_numpy()).to(torch.float32),
            torch.Tensor(embeds).to(torch.float32),
            torch.Tensor([target]).to(torch.float32),
        )

    @property
    def average_dt(self):
        day_delta = np.timedelta64(1, "D") * GlobalConfig.TRADING_DAYS
        return self.df.index.to_series().diff(1).mean() / day_delta
