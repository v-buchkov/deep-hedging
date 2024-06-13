import os
import datetime as dt
from typing import Type, Union
from pathlib import Path

import numpy as np
import pandas as pd

import gym
from gym import spaces

from deep_hedging.base import Instrument
from deep_hedging.config import GlobalConfig, ExperimentConfig

pd.options.mode.chained_assignment = None


class DerivativeEnvSequence(gym.Env):
    METADATA = {"render.modes": ["human"]}

    def __init__(
        self,
        n_days: int,
        instrument_cls: Type[Instrument],
        data: Union[pd.DataFrame, None] = None,
        root_dir: Path = ExperimentConfig.DATA_ROOT,
        filename: str = ExperimentConfig.DATA_FILENAME,
    ):
        # Internal attributes
        self.instrument_cls = instrument_cls
        self.n_days = n_days

        self._trajectory_data, self._target_pnl, self._max_step = None, None, None

        self.random_seed = ExperimentConfig.RANDOM_SEED

        self.df = self._create_df(root_dir, filename) if data is None else data.copy()
        self.df.dropna(inplace=True)

        self.df = self._add_time_diff(self.df)

        self.dt = self.get_average_dt()

        last_point_idx = self.df.index >= self.df.index.max() - dt.timedelta(
            days=self.n_days
        )
        self._last_point = self.df.shape[0] - self.df[last_point_idx].shape[0]

        # RL attributes
        self._initialize()

        self.action_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self._trajectory_data.shape[0] - 2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._trajectory_data.shape,
            dtype=np.float32,
        )

    def _initialize(self):
        self._pnl = 0
        self.weights = np.array([0])
        self.pnl_path = []
        self.diff_path = []

        (
            self._trajectory_data,
            self._target_pnl,
            self._max_step,
        ) = self._sample_trajectory()

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
        return self.instrument_cls(
            rates_difference=start[GlobalConfig.RATE_DOMESTIC_COLUMN]
            - start[GlobalConfig.RATE_FOREIGN_COLUMN],
            spot_price=spot_start,
            term=self.n_days / GlobalConfig.CALENDAR_DAYS,
        )

    def _sample_trajectory(self) -> tuple[np.array, np.array, int]:
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        idx = np.random.choice(np.arange(self._last_point), replace=True)
        start_date = self.df.index[idx]
        end_date = start_date + dt.timedelta(days=self.n_days)

        features = self.df[(self.df.index >= start_date) & (self.df.index <= end_date)]
        time_start = features.iloc[0, -1]
        features[GlobalConfig.TIME_DIFF_COLUMN] = features[
            GlobalConfig.TIME_DIFF_COLUMN
        ].apply(lambda x: x - time_start)
        target = self._create_instrument(features).payoff(spot=features.ask.iloc[-1])

        data = features.to_numpy()
        target_pnl = target
        max_step = features.shape[0] - 2

        return data, target_pnl, max_step

    def get_average_dt(self):
        return self.df.index.to_series().diff(1).mean() / (
            np.timedelta64(1, "D") * GlobalConfig.TRADING_DAYS
        )

    def step(self, action):
        diff = self._target_pnl - self.get_pnl(
            weights=action, spot=self._trajectory_data
        )
        reward = 1 / diff**2

        self.pnl_path.append(self._pnl)
        self.diff_path.append(diff)

        (
            self._trajectory_data,
            self._target_pnl,
            self._max_step,
        ) = self._sample_trajectory()

        done = 1

        return self._trajectory_data, reward, done, {}

    def reset(self, *args, **kwargs):
        # Reset the state of the environment to an initial state
        self._initialize()

        return self._trajectory_data

    def get_pnl(self, weights: np.array, spot: np.array) -> float:
        weights_all = np.concatenate([np.zeros((1,)), weights, np.zeros((1,))])
        weights_diff = np.diff(weights_all, n=1)

        rates_diff = spot[:, 2] - spot[:, 3]

        bought = np.where(weights_diff > 0, weights_diff, 0)
        sold = np.where(weights_diff < 0, weights_diff, 0)

        interest = (rates_diff * -weights_all * self.dt).sum()

        cash_outflow = (-spot[1:, 1] * bought).sum()
        cash_inflow = (-spot[1:, 0] * sold).sum()

        return cash_outflow + cash_inflow + interest

    def render(self, mode: str = "human", close: bool = False):
        print(self.pnl_path)
        print(self.diff_path)
