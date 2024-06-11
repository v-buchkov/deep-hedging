import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from deep_hedging.dl.train import validation_epoch
from deep_hedging.config import ExperimentConfig


class Assessor:
    def __init__(
        self,
        model: nn.Module,
        baseline: nn.Module,
        test_loader: DataLoader,
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.model = model
        self.baseline = baseline

        self.test_loader = test_loader

        self.config = config

    def run(self) -> None:
        _, weights, _, model_diff = validation_epoch(
            self.model, nn.MSELoss(), self.test_loader
        )
        _, _, _, baseline_diff = validation_epoch(
            self.baseline, nn.MSELoss(), self.test_loader
        )

        model_diff = np.concatenate(model_diff, axis=0)
        baseline_diff = np.concatenate(baseline_diff, axis=0)

        print(
            f"Average weight = {weights[-1].mean()}, Weights = [{weights[-1].min()}; {weights[-1].max()}]"
        )

        print(
            f"Means: model = {model_diff.mean():.6f}, baseline = {baseline_diff.mean():.6f}"
        )

        print(
            f"Stds: model = {model_diff.std():.6f}, baseline = {baseline_diff.std():.6f}"
        )

        var_model = -np.quantile(model_diff, self.config.VAR_QUANTILE)
        var_baseline = -np.quantile(baseline_diff, 0.05)
        print(f"VaRs 5%: model = {var_model:.6f}, baseline = {var_baseline:.6f}")

        t_value = (model_diff.mean() - baseline_diff.mean()) / np.sqrt(
            model_diff.std() ** 2 / model_diff.shape[0]
            + baseline_diff.std() ** 2 / baseline_diff.shape[0]
        )
        print(f"T-stat = {t_value:.6f}")

        bins = np.linspace(-0.25, 0.25, 100)

        plt.hist(model_diff, bins, alpha=0.5, label="model")
        plt.hist(baseline_diff, bins, alpha=0.5, label="baseline")
        plt.legend(loc="upper right")
        plt.show()
