import os
from typing import Type

import pandas as pd
import gym

from stable_baselines3.common.vec_env import DummyVecEnv

from deep_hedging.base import Instrument
from deep_hedging.config import ExperimentConfig


class RLTrainer:
    def __init__(
        self,
        model,
        environment_cls: Type[gym.Env],
        instrument_cls: Type[Instrument],
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.hedger = model
        self.config = config

        data_df = self._load_df().resample(self.config.REBAL_FREQ).ffill()
        self.env = DummyVecEnv(
            [
                lambda: environment_cls(
                    n_days=config.N_DAYS, data=data_df, instrument_cls=instrument_cls
                )
            ]
        )

    def _load_df(self) -> pd.DataFrame:
        filename = self.config.DATA_FILENAME + ".pkl"
        file_path = self.config.DATA_ROOT / filename
        if filename in os.listdir(self.config.DATA_ROOT):
            return pd.read_pickle(file_path)
        else:
            raise FileNotFoundError

    def learn(self, n_steps: int = ExperimentConfig.N_STEPS_RL_TRAIN) -> None:
        self.hedger.learn(total_timesteps=n_steps)

    def assess(self, n_steps: int) -> list[float]:
        vec_env = self.hedger.get_env()
        obs = vec_env.reset()
        diffs = []
        for _ in range(n_steps):
            action, _states = self.hedger.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if reward != 0:
                diffs.append(reward)

        return diffs
