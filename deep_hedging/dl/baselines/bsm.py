class BaselineEuropeanCall(nn.Module):
    def __init__(self, dt: float = AVERAGE_DT):
        super().__init__()

        self.lstm = nn.LSTM(1, 1, num_layers=1, batch_first=True)
        self.dt = dt

        self.strike = 1

    def _call_delta(self, mid: torch.Tensor, rates: torch.Tensor, terms: torch.Tensor) -> torch.float32:
        """
        Call non_linear delta [dV/dS] via analytical form solution of Black-Scholes-Merton.

        Returns
        -------
        delta : float
            Option delta.
        """
        strikes = mid[:, 0] * self.strike
        # print(strikes[: -1])
        sigma = mid.std(dim=1).unsqueeze(1)
        # print("***")
        d1 = (torch.log(mid / strikes.unsqueeze(1)) + (rates + sigma ** 2 / 2) * terms) / (sigma * torch.sqrt(terms))
        d1 = d1[:, 1:-1]
        # print(d1.shape)
        # print("***")

        cdf_d1 = torch.distributions.normal.Normal(0, 1).cdf(d1)

        return cdf_d1

    def forward(self, spot: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        mid = (spot[:, :, 0] + spot[:, :, 1]) / 2
        rates = spot[:, :, 2] - spot[:, :, 3]
        terms = spot[:, :, 4]
        return self._call_delta(mid=mid, rates=rates, terms=terms)

    def get_pnl(self, spot: torch.Tensor) -> torch.float32:
        # hedging_weights = nn.Softmax()(self.linear(spot, return_hidden=False), dim=XXX)
        hedging_weights = self.forward(spot, return_hidden=False)
        return hedging_weights, get_pnl(spot=spot, weights=hedging_weights, dt=self.dt)
