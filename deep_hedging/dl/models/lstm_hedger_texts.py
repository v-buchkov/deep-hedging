import torch
import torch.nn as nn

from deep_hedging.dl.models.abstract_hedger import AbstractHedger


class LSTMHedgerTexts(AbstractHedger):
    def __init__(self, input_size: int, num_layers: int, hidden_size: int, dt: float):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dt = dt

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

        hidden = []
        for input_t in text.chunk(text.size(0), dim=0):
            h, c = self.lstm_text(input_t.squeeze(0))
            x = self.sentence(h)
            hidden.append(x.squeeze(1))
        hidden = torch.stack(hidden).squeeze(3)

        spot = torch.cat([spot, hidden], dim=2)

        h_t, c_t = self.lstm(spot, (h_t, c_t))
        outputs = self.hedging_weights(h_t)[:, :-2, :].squeeze(2)

        if return_hidden:
            return outputs, (h_t, c_t)
        else:
            return outputs
