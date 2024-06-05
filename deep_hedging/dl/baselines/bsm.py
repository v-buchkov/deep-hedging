import torch
import torch.nn as nn

from deep_hedging.dl.models import AbstractHedger


class BaselineEuropeanCall(AbstractHedger):
    def __init__(self, dt: float, strike: float = 1):
        super().__init__()

        self.lstm = nn.LSTM(1, 1, num_layers=1, batch_first=True)

        self.strike = strike
        self.dt = dt

    def _call_delta(
        self, mid: torch.Tensor, rates: torch.Tensor, terms: torch.Tensor
    ) -> torch.float32:
        """
        Call non_linear delta [dV/dS] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        delta : float
            Option delta.
        """
        strikes = mid[:, 0] * self.strike
        sigma = mid.std(dim=1).unsqueeze(1)
        d1 = (
            torch.log(mid / strikes.unsqueeze(1)) + (rates + sigma**2 / 2) * terms
        ) / (sigma * torch.sqrt(terms))
        d1 = d1[:, 1:-1]

        cdf_d1 = torch.distributions.normal.Normal(0, 1).cdf(d1)

        return cdf_d1

    def forward(
        self,
        spot: torch.Tensor,
        text: [torch.Tensor, None] = None,
        hidden: [torch.Tensor, (torch.Tensor, torch.Tensor), None] = None,
        return_hidden: bool = False,
    ) -> [torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)]:
        model_device = next(self.parameters()).device
        mid = (spot[:, :, 0] + spot[:, :, 1]) / 2
        rates = spot[:, :, 2] - spot[:, :, 3]
        terms = spot[:, :, 4]
        return self._call_delta(mid=mid, rates=rates, terms=terms).to(model_device)
