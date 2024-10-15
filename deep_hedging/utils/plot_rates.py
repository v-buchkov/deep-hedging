import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rates(
    data: pd.Series,
    simulated: [pd.DataFrame, None] = None,
    opacity: float = 0.03,
    quantiles: tuple[float] = (0.05, 0.5, 0.95),
) -> None:
    plt.figure(figsize=(14, 7))

    plt.plot(data, label="close")

    if simulated is not None:
        colors = plt.colormaps["BuPu"](np.linspace(0.0, 1.0, simulated.shape[1]))
        for i in range(simulated.shape[1]):
            plt.plot(simulated.iloc[:, i], color=colors[i], alpha=opacity)

        for q in quantiles:
            plt.plot(simulated.quantile(q, axis=1), color=colors[-33])

        plt.axvline(x=simulated.index[0], color="m", linestyle="--")

    plt.legend(["TONIA"])
    plt.title("TONIA Rate Time Series")

    plt.xlabel("Date")
    plt.ylabel("TONIA")

    plt.show()
