from typing import Type
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class ExperimentConfig:
    OPT_STRIKE: float = field(
        default=1.0,
        metadata={"docs": "Option strike, defined as % of spot level"},
    )

    N_DAYS: int = field(
        default=5,
        metadata={"docs": "Number of days till maturity of the considered derivative"},
    )

    REBAL_FREQ: str = field(
        default="30 min", metadata={"docs": "Hedging rebalancing frequency"}
    )

    VAR_QUANTILE: float = field(
        default=0.05, metadata={"docs": "Quantile for Value-at-Risk calculation"}
    )

    TEST_SIZE: float = field(default=0.1, metadata={"docs": "Size of the test sample"})

    RANDOM_SEED: int = field(
        default=12, metadata={"docs": "Fixing randomization with integer state"}
    )

    DATA_ROOT: Path = field(
        default=Path("data"), metadata={"docs": "Data root directory"}
    )

    OUTPUT_ROOT: Path = field(
        default=Path("output"), metadata={"docs": "Output root directory"}
    )

    DATA_FILENAME: str = field(default="data_full", metadata={"docs": "Data filename"})

    LAYER_NORM: bool = field(
        default=False,
        metadata={
            "docs": "Boolean, whether to normalize the data by LayerNorm of BatchNorm"
        },
    )

    USE_TIME_DIFF: bool = field(
        default=True,
        metadata={"docs": "Boolean, whether to use time difference as a feature"},
    )

    USE_SPOT_START: bool = field(
        default=True,
        metadata={"docs": "Boolean, whether to use spot start as a feature"},
    )

    N_EPOCHS: int = field(default=20, metadata={"docs": "Number of training epochs"})

    N_STEPS_RL_TRAIN: int = field(
        default=1_000_000, metadata={"docs": "Number of steps for training RL model"}
    )

    LR: float = field(default=1e-2, metadata={"docs": "Starting learning rate"})

    BATCH_SIZE: int = field(
        default=32, metadata={"docs": "Batch size to use in learning"}
    )

    NUM_LAYERS: int = field(default=3, metadata={"docs": "Number of hidden layers"})

    HIDDEN_DIM: int = field(default=32, metadata={"docs": "Size of hidden layers"})

    EMBED_MAX_DIM: int = field(
        default=128,
    )

    OPTIMIZER: Type[torch.optim.Optimizer] = field(
        default=torch.optim.Adam, metadata={"docs": "Optimizer to use in learning"}
    )

    NUM_WORKERS: int = field(
        default=2, metadata={"docs": "Number of available workers"}
    )

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
