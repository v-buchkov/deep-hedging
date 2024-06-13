import torch
import torch.nn as nn

from deep_hedging.dl.models.abstract_hedger import AbstractHedger


class LSTMHedger(AbstractHedger):
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        dt: float,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt

        if layer_norm:
            self.norm = nn.LayerNorm(self.input_size)
        else:
            self.norm = nn.BatchNorm1d(self.input_size)

        self.lstm = nn.LSTM(
            input_size, self.hidden_size, num_layers=num_layers, batch_first=True
        )

        self.hedging_weights = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(
        self,
        spot: torch.Tensor,
        text: [torch.Tensor, None] = None,
        hidden: [torch.Tensor, (torch.Tensor, torch.Tensor), None] = None,
        return_hidden: bool = False,
    ) -> [torch.Tensor, (torch.Tensor, torch.Tensor, torch.Tensor)]:
        model_device = spot.device
        if hidden is None:
            h_t = torch.zeros(
                self.num_layers, spot.size(0), self.hidden_size, dtype=torch.float32
            ).to(model_device)
            c_t = torch.zeros(
                self.num_layers, spot.size(0), self.hidden_size, dtype=torch.float32
            ).to(model_device)
        elif len(hidden) != 2:
            raise ValueError(f"Expected two hidden state variables, got {len(hidden)}")
        else:
            h_t, c_t = hidden

        mid = (spot[:, :, 0] + spot[:, :, 1]) / 2
        mid = torch.log(mid / mid[:, 0].unsqueeze(1)).unsqueeze(2)

        bid_ask_spread = (spot[:, :, 1] - spot[:, :, 0]).unsqueeze(2)
        rates = spot[:, :, 2:4]

        spot = torch.cat([mid, bid_ask_spread, rates, spot[:, :, 4:]], dim=2)

        h_t, c_t = self.lstm(spot, (h_t, c_t))
        outputs = self.hedging_weights(h_t)[:, :-2, :].squeeze(2)

        if return_hidden:
            return outputs, (h_t, c_t)
        else:
            return outputs
