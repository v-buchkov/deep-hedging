import os
import datetime as dt
from typing import Type, Tuple, Optional, Any, Union
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_hedging.dl.spot_dataset import SpotDataset
from deep_hedging.base import Instrument
from deep_hedging.config import ExperimentConfig
from deep_hedging.dl.train import train_epoch, validation_epoch, plot_losses


class Trainer:
    def __init__(
        self,
        model_cls: Type[torch.nn.Module],
        instrument_cls: Type[Instrument],
        config: ExperimentConfig = ExperimentConfig(),
    ):
        self.model_cls = model_cls
        self.instrument_cls = instrument_cls
        self.config = config

        self.hedger = None
        self.train_loader, self.test_loader = None, None
        self.weights, self.train_diffs, self.val_diffs = None, None, None
        self.dt = None

        self._initialize()

    def _initialize(self):
        data_df = self._load_df().resample(self.config.REBAL_FREQ).ffill()
        train_set, test_set = self._get_datasets(data_df)
        self.train_loader, self.test_loader = self._get_dataloaders(train_set, test_set)
        self.dt = train_set.average_dt

        self.hedger = self.model_cls(
            input_size=train_set[0][0].shape[1],
            num_layers=self.config.NUM_LAYERS,
            hidden_size=self.config.HIDDEN_DIM,
            dt=self.dt,
            layer_norm=self.config.LAYER_NORM,
        )
        self.hedger = self.hedger.to(self.config.DEVICE)

    def _get_dataloaders(
        self, train_set: SpotDataset, test_set: SpotDataset
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_set, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=False
        )
        test_loader = DataLoader(
            test_set, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=False
        )
        return train_loader, test_loader

    def _get_datasets(self, data: pd.DataFrame) -> Tuple[SpotDataset, SpotDataset]:
        time_split = data.index[int(data.index.shape[0] * (1 - self.config.TEST_SIZE))]
        train_df, test_df = (
            data[data.index <= time_split],
            data[data.index > time_split],
        )

        train_set = SpotDataset(
            data=train_df, instrument_cls=self.instrument_cls, n_days=self.config.N_DAYS
        )
        test_set = SpotDataset(
            data=test_df, instrument_cls=self.instrument_cls, n_days=self.config.N_DAYS
        )

        return train_set, test_set

    def _load_df(self) -> pd.DataFrame:
        filename = self.config.DATA_FILENAME + ".pkl"
        file_path = self.config.DATA_ROOT / filename
        if filename in os.listdir(self.config.DATA_ROOT):
            return pd.read_pickle(file_path)
        else:
            raise FileNotFoundError

    def _train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        print_logs: bool = True,
    ):
        train_losses, val_losses = [], []
        train_diffs, val_diffs = [], []
        criterion = nn.MSELoss().to(self.config.DEVICE)

        for epoch in range(1, num_epochs + 1):
            if print_logs:
                desc_train = f"Training {epoch}/{num_epochs}"
                desc_val = f"Validation {epoch}/{num_epochs}"
            else:
                desc_train, desc_val = None, None

            train_loss, weights, train_diff, train_path = train_epoch(
                model, optimizer, criterion, train_loader, tqdm_desc=desc_train
            )
            val_loss, weights, val_diff, val_path = validation_epoch(
                model, criterion, val_loader, tqdm_desc=desc_val
            )

            if scheduler is not None:
                scheduler.step()

            train_losses += [train_loss]
            val_losses += [val_loss]

            train_diffs += [train_diff]
            val_diffs += [val_diff]

            plot_losses(train_losses, val_losses, train_diffs, val_diffs)

        return train_losses, val_losses, weights, train_diffs, val_diffs

    def run(self, n_epochs: Union[int, None] = None) -> None:
        optimizer = self.config.OPTIMIZER(self.hedger.parameters(), lr=self.config.LR)

        if n_epochs is None:
            n_epochs = self.config.N_EPOCHS

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        _, _, self.weights, self.train_diffs, self.val_diffs = self._train(
            model=self.hedger,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=self.train_loader,
            val_loader=self.test_loader,
            num_epochs=n_epochs,
            print_logs=True,
        )

    def save(self, path: Path) -> None:
        torch.save(self.hedger, path / f"run_{dt.datetime.now()}.pt")
