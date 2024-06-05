import abc

import torch
import torch.nn as nn


def get_pnl(spot: torch.Tensor, weights: torch.Tensor, dt: float) -> torch.float32:
    model_device = spot.device
    weights_all = torch.concat(
        [
            torch.zeros(spot.shape[0], 1, requires_grad=False).to(model_device),
            weights,
            torch.zeros(spot.shape[0], 1, requires_grad=False).to(model_device),
        ],
        dim=1,
    )
    weights_diff = weights_all.diff(n=1, dim=1)

    rates_diff = spot[:, :, 2] - spot[:, :, 3]

    bought = torch.where(weights_diff > 0, weights_diff, 0)
    sold = torch.where(weights_diff < 0, weights_diff, 0)

    interest = (rates_diff * -weights_all).sum(dim=1) * dt

    cash_outflow = (-spot[:, 1:, 1] * bought).sum(dim=1)
    cash_inflow = (-spot[:, 1:, 0] * sold).sum(dim=1)

    return (cash_outflow + cash_inflow + interest).unsqueeze(1)


class AbstractHedger(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        spot: torch.Tensor,
        text: [torch.Tensor, None] = None,
        hidden: [torch.Tensor, (torch.Tensor, torch.Tensor), None] = None,
        return_hidden: bool = False,
    ) -> [torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)]:
        raise NotImplementedError

    def get_pnl(self, spot: torch.Tensor) -> [torch.Tensor, torch.float32]:
        # hedging_weights = nn.Softmax()(self.linear(spot, return_hidden=False), dim=XXX)
        hedging_weights = self.forward(spot, return_hidden=False)
        return hedging_weights, get_pnl(spot=spot, weights=hedging_weights, dt=self.dt)
