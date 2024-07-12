from deep_hedging.underlyings.ticker import Ticker


class Tickers:
    def __init__(self, tickers: list[Ticker]):
        self.tickers_list = tickers

        self._initialize()

    def _initialize(self):
        self._ticker_dict = self._get_ticker_dict(self.tickers_list)
        self._ticker_inverse_dict = self._get_ticker_inverse_dict(self.tickers_list)

        self.names = list(self._ticker_inverse_dict.keys())
        self.codes = list(self._ticker_dict.keys())

    @staticmethod
    def _get_ticker_dict(tickers: list[Ticker]) -> dict[str, Ticker]:
        return {ticker.code: ticker for ticker in tickers}

    @staticmethod
    def _get_ticker_inverse_dict(tickers: list[Ticker]) -> dict[str, Ticker]:
        return {ticker.name: ticker for ticker in tickers}

    def __len__(self):
        return len(self.tickers_list)

    def __getitem__(self, item: [int, str]) -> [str, None]:
        if isinstance(item, int):
            return self.tickers_list[item]
        elif isinstance(item, str):
            if item in self._ticker_dict.keys():
                return self.get(item)
            elif item in self._ticker_inverse_dict.keys():
                return self.get_inverse(item)
            else:
                return None
        else:
            raise TypeError(f"Item {item} is not a valid ticker or index")

    def __add__(self, other):
        return Tickers(self.tickers_list + other.tickers_list)

    def get(self, name: str) -> Ticker:
        return self._ticker_dict[name]

    def get_inverse(self, code: str) -> Ticker:
        return self._ticker_inverse_dict[code]

    def __iter__(self):
        return iter(self.tickers_list)
