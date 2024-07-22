import os
import datetime as dt
from typing import Union, Type
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from deep_hedging.base import Instrument
from deep_hedging.non_linear.base_option import BaseOption
from deep_hedging.curve.constant_rate import ConstantRateCurve

from deep_hedging.config.global_config import GlobalConfig
from deep_hedging.config.experiment_config import ExperimentConfig


class SpotDataset(Dataset):
    def __init__(
        self,
        n_days: int,
        instrument_cls: Type[Instrument],
        data: Union[pd.DataFrame, None] = None,
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.instrument_cls = instrument_cls
        # self.n_days = 2 * n_days
        self.n_days = n_days
        self.config = config

        self.df = (
            self._create_df(self.config.DATA_ROOT, self.config.DATA_FILENAME)
            if data is None
            else data.copy()
        )
        self.df = self._add_time_diff(self.df)

        self.df = self.df.ffill()

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
        start = period_df.index.min()
        end = period_df.index.max()

        start_df = period_df.loc[start]
        spot_start = (
            start_df[GlobalConfig.BID_COLUMN] + start_df[GlobalConfig.ASK_COLUMN]
        ) / 2

        params = {
            "yield_curve": ConstantRateCurve(
                constant_rate=start_df[GlobalConfig.RATE_DOMESTIC_COLUMN]
                - start_df[GlobalConfig.RATE_FOREIGN_COLUMN]
            ),
            "start_date": start,
            "end_date": end,
        }

        if issubclass(self.instrument_cls, BaseOption):
            params["strike_level"] = self.config.OPT_STRIKE

        return (
            self.instrument_cls(**params),
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

        instrument, spot_start = self._create_instrument(features)
        features[GlobalConfig.SPOT_START_COLUMN] = spot_start

        target = (
            instrument.payoff(
                spot=np.array([features.ask.iloc[-1] / spot_start]).reshape(1, -1)
            )[0]
            * spot_start
        )

        return torch.Tensor(features.to_numpy()).to(torch.float32), torch.Tensor(
            [target]
        ).to(torch.float32)

    @property
    def average_dt(self):
        day_delta = np.timedelta64(1, "D") * GlobalConfig.TRADING_DAYS
        return self.df.index.to_series().diff(1).mean() / day_delta
