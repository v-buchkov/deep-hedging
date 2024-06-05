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
