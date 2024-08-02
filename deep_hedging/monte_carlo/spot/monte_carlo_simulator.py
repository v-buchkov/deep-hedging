import abc
from typing import Callable

import numpy as np


class MonteCarloSimulator:
    def __init__(
        self,
        payoff_function: Callable[[np.array], float],
        random_seed: [int, None] = None,
    ):
        self.payoff_function = payoff_function
        self.random_seed = random_seed

    @abc.abstractmethod
    def get_paths(self, *args, **kwargs) -> np.array:
        raise NotImplementedError

    def future_value(
        self,
        spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ) -> float:
        random_paths = self.get_paths(
            spot=spot,
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths,
        )

        instrument_payoffs = self.payoff_function(random_paths)
        return np.mean(instrument_payoffs)

    def std(
        self,
        spot: list[float],
        time_till_maturity: float,
        risk_free_rate_fn: Callable[[float], float],
        dividends_fn: Callable[[float], float],
        var_covar_fn: Callable[[float], np.array],
        n_paths: [int, None] = None,
    ) -> float:
        random_paths = self.get_paths(
            spot=spot,
            time_till_maturity=time_till_maturity,
            risk_free_rate_fn=risk_free_rate_fn,
            dividends_fn=dividends_fn,
            var_covar_fn=var_covar_fn,
            n_paths=n_paths,
        )

        instrument_payoffs = self.payoff_function(random_paths)
        return np.std(instrument_payoffs)
