from deep_hedging.market_data.ticker import Ticker


class Tickers:
    def __init__(self, tickers: list[Ticker]):
        self._tickers = tickers

        self._ticker_dict = self._get_ticker_dict(tickers)
        self._ticker_inverse_dict = self._get_ticker_inverse_dict(tickers)

        self.names = list(self._ticker_dict.keys())
        self.codes = list(self._ticker_dict.values())

    @staticmethod
    def _get_ticker_dict(tickers: list[Ticker]) -> dict[str, str]:
        return {ticker.name: ticker.code for ticker in tickers}

    @staticmethod
    def _get_ticker_inverse_dict(tickers: list[Ticker]) -> dict[str, str]:
        return {ticker.code: ticker.name for ticker in tickers}

    def __len__(self):
        return len(self._tickers)

    def __getitem__(self, item: [int, str]) -> [str, None]:
        if isinstance(item, int):
            return self._tickers[item]
        elif isinstance(item, str):
            if item in self._ticker_dict.keys():
                return self.get(item)
            elif item in self._ticker_inverse_dict.keys():
                return self.get_inverse(item)
            else:
                return None
        else:
            raise TypeError(f"Item {item} is not a valid ticker or index")

    def get(self, name: str) -> str:
        return self._ticker_dict[name]

    def get_inverse(self, code: str) -> str:
        return self._ticker_inverse_dict[code]

    def __iter__(self):
        return iter(self._tickers)
