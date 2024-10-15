from dataclasses import dataclass, field


@dataclass
class GlobalConfig:
    #  Financial conventions
    TRADING_DAYS: int = field(
        default=252, metadata={"docs": "Number of trading days in a year"}
    )

    CALENDAR_DAYS: int = field(
        default=365, metadata={"docs": "Number of calendar days in a year"}
    )

    MONTE_CARLO_PATHS: int = field(
        default=100_000,
        metadata={"docs": "Default number of paths for Monte Carlo Simulation"},
    )

    YEARS_IN_CURVE: float = field(
        default=25.0,
        metadata={"docs": "Default maximum years for construction of a yield curve"},
    )

    N_POINTS_CURVE: int = field(
        default=100,
        metadata={"docs": "Default number of points for construction of a yield curve"},
    )

    # Naming conventions
    BID_COLUMN: str = field(
        default="bid",
        metadata={"docs": "Default name of the column for bid price in market data"},
    )

    ASK_COLUMN: str = field(
        default="ask",
        metadata={"docs": "Default name of the column for ask price in market data"},
    )

    RATE_DOMESTIC_COLUMN: str = field(
        default="rub_rate",
        metadata={"docs": "Default name of the domestic interest rate in market data"},
    )

    RATE_FOREIGN_COLUMN: str = field(
        default="usd_rate",
        metadata={"docs": "Default name of the foreign interest rate in market data"},
    )

    TIME_DIFF_COLUMN: str = field(
        default="time_diff",
        metadata={"docs": "Default name of the time difference in years between observed points"},
    )

    TEXT_COLUMN: str = field(
        default="text",
        metadata={"docs": "Default name of the relevant text features in market data"},
    )

    LEMMAS_COLUMN: str = field(
        default="lemmas",
        metadata={"docs": "Default name of the column for lemmas from relevant text feature in market data"},
    )

    EMBEDDING_COLUMN: str = field(
        default="embed",
        metadata={"docs": "Default name of the column for embeddings from relevant text feature in market data"},
    )

    SPOT_START_COLUMN: str = field(
        default="spot_start",
        metadata={"docs": "Default name of the column for spot price reference at the inception of the derivative"},
    )

    TARGET_COLUMN: str = field(
        default="ytm",
    )

    DISCOUNT_FACTOR_COLUMN: str = field(
        default="discount_factor",
        metadata={"docs": "Default name of the column for discount factor"}
    )

    FWD_RATE_COLUMN: str = field(
        default="fwd_rate",
        metadata={"docs": "Default name of the column for forward rate in market data"},
    )

    VOLATILITY_ROLLING_DAYS: int = field(
        default=12,
        metadata={"docs": "Default number of daily returns for rolling volatility estimation"},
    )
