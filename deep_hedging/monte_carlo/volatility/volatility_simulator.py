import abc

import numpy as np

from deep_hedging.config.global_config import GlobalConfig


class VolatilitySimulator:
    def __init__(
            self,
            random_seed: [int, None] = None,
    ):
        self.random_seed = random_seed

    @abc.abstractmethod
    def get_paths(self, noise: np.array = None, n_paths: int = GlobalConfig.MONTE_CARLO_PATHS, *args, **kwargs) -> np.array:
        raise NotImplementedError
