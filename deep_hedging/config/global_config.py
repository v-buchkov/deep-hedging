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
    )

    N_POINTS_CURVE: int = field(
        default=100,
    )

    # Naming conventions
    BID_COLUMN: str = field(
        default="bid",
    )

    ASK_COLUMN: str = field(
        default="ask",
    )

    RATE_DOMESTIC_COLUMN: str = field(
        default="rub_rate",
    )

    RATE_FOREIGN_COLUMN: str = field(
        default="usd_rate",
    )

    TIME_DIFF_COLUMN: str = field(
        default="time_diff",
    )

    TEXT_COLUMN: str = field(
        default="text",
    )

    LEMMAS_COLUMN: str = field(
        default="lemmas",
    )

    EMBEDDING_COLUMN: str = field(
        default="embed",
    )

    SPOT_START_COLUMN: str = field(
        default="spot_start",
    )

    TARGET_COLUMN: str = field(
        default="ytm",
    )

    DISCOUNT_FACTOR_COLUMN: str = field(
        default="discount_factor",
    )

    FWD_RATE_COLUMN: str = field(
        default="fwd_rate",
    )
