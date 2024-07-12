from dataclasses import dataclass


@dataclass
class Ticker:
    name: str
    code: str
    currency: str = None
