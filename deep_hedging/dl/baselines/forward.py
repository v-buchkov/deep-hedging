import torch
import torch.nn as nn

from deep_hedging.dl.models import AbstractHedger


class BaselineForward(AbstractHedger):
    def __init__(self, dt: float):
        super().__init__()

        self.lstm = nn.LSTM(1, 1, num_layers=1, batch_first=True)
        self.dt = dt

    def forward(
        self,
        spot: torch.Tensor,
        text: [torch.Tensor, None] = None,
        hidden: [torch.Tensor, (torch.Tensor, torch.Tensor), None] = None,
        return_hidden: bool = False,
    ) -> [torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)]:
        model_device = next(self.parameters()).device
        return (
            torch.Tensor([[1] * (spot.shape[1] - 2)] * spot.shape[0])
            .to(torch.float32)
            .to(model_device)
        )
