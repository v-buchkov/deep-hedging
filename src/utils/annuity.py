def annuity_factor(annual_rate: float, frequency: float, till_maturity: float) -> float:
    rate = annual_rate * frequency
    number_of_payments = till_maturity / frequency
    return (1 - (1 + rate) ** (-number_of_payments)) / rate
