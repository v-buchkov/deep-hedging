import torch
import torch.nn as nn


class BaselineForward(nn.Module):
    def __init__(self, dt: float):
        super().__init__()

        self.lstm = nn.LSTM(1, 1, num_layers=1, batch_first=True)
        self.dt = dt

    def forward(self, spot: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        return torch.Tensor([[1] * (spot.shape[1] - 2)] * spot.shape[0]).to(torch.float32).to(DEVICE)

    def get_pnl(self, spot: torch.Tensor) -> torch.float32:
        # hedging_weights = nn.Softmax()(self.linear(spot, return_hidden=False), dim=XXX)
        hedging_weights = self.forward(spot, return_hidden=False)
        return hedging_weights, get_pnl(spot=spot, weights=hedging_weights, dt=self.dt)
