# MSc Thesis: “Hedging Derivatives Under Incomplete Markets with Deep Learning”
_Buchkov Viacheslav_\
MSc in Applied Mathematics and 
Informatics\
Machine Learning and Data-Intensive Systems\
Faculty of Computer Science\
NRU Higher School of Economics

## Install

```
pip install deep-hedging
```

## Deep Learning Example

```
from pathlib import Path

from deep_hedging import ExperimentConfig, EuropeanCall

from deep_hedging.dl import Trainer, Assessor
from deep_hedging.dl.models import LSTMHedger
from deep_hedging.dl.baselines import BaselineEuropeanCall

# Amend config
config = ExperimentConfig(
    DATA_ROOT=Path(...),
    OUTPUT_ROOT=Path(...),
    DATA_FILENAME="...",
    REBAL_FREQ="5 min"
)

# Train Hedger for 1 epoch
trainer = Trainer(model_cls=LSTMHedger, instrument_cls=EuropeanCall, config=config)
trainer.run(1)

# Assess obtained quality
assessor = Assessor(
    model=trainer.hedger,
    baseline=BaselineEuropeanCall(dt=trainer.dt).to(config.DEVICE),
    test_loader=trainer.test_loader,
)
assessor.run()

# Save model
trainer.save(config.OUTPUT_ROOT)
```

## Custom Derivative Example

```
from pathlib import Path

from deep_hedging import ExperimentConfig, Instrument

from deep_hedging.dl import Trainer, Assessor
from deep_hedging.dl.models import LSTMHedger
from deep_hedging.dl.baselines import BaselineEuropeanCall

# Amend config
config = ExperimentConfig(
    DATA_ROOT=Path(...),
    OUTPUT_ROOT=Path(...),
    DATA_FILENAME="...",
    REBAL_FREQ="5 min"
)

# Create custom derivative
class CustomDerivative(Instrument):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def payoff(self, spot: float) -> float:
        return ... # any payoff you want - e.g., spot ** 12 - 12 * np.random.randint(12, 121)

    def __repr__(self):
        return f"SomeCustomDerivative(param1=..., param2=...)"

# Train Hedger for 1 epoch
trainer = Trainer(model_cls=LSTMHedger, instrument_cls=CustomDerivative, config=config)
trainer.run(1)

# Save model
trainer.save(config.OUTPUT_ROOT)
```

## Reinforcement Learning Example

```
from pathlib import Path

from deep_hedging import ExperimentConfig, EuropeanCall, seed_everything
from deep_hedging.rl import DerivativeEnvStep, RLTrainer

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC, PPO

# Amend config
config = ExperimentConfig(
    DATA_ROOT=Path(...),
    OUTPUT_ROOT=Path(...),
    DATA_FILENAME="...",
    REBAL_FREQ="5 min"
)

# Create environment
env = DerivativeEnvStep(n_days=config.N_DAYS, instrument_cls=EuropeanCall)
env.reset()

# Train Hedger for 1_000 steps
trainer = RLTrainer(
    model=RecurrentPPO("MlpLstmPolicy", env, verbose=1),
    instrument_cls=EuropeanCall,
    environment_cls=DerivativeEnvStep,
    config=config,
)
trainer.learn(1_000)

# Assess obtained quality at 100 steps
trainer.assess(100)
```

## Description of Research Tasks

**Research Task**: create an universal algorithm that would produce for each point of time weights vector for replicating portfolio assets to dynamically delta-hedge a derivative, defined by payoff function only. The algorithm should take into account “state-of-the-world” embedding, historical dynamics of underlying asset and parameters of a derivative (like time till maturity for each point in time). The target function for optimization would be to minimize difference between derivative’s PnL and replicating portfolio’s PnL.
Potentially, approach might be adjusted to fit Reinforcement Learning framework.

**Data**:
MVP: Generate paths via GBM in order to test basics of the architecture (should coincide with BSM delta-hedging, if no constraints are imposed)
Base: orderbooks for FX and FX Options at Moscow Exchange
Advanced: use generative model (GANs, NFs, VAEs, Diffusion models) to create random paths and then apply hedging framework

**Baseline**:
closed-out solution of delta-hedging (start with BSM delta-hedging for vanilla option, base: local volatility, Heston, advanced: SABR)
check, if optimization returns replications, for which put-call parity robustly holds (however, due to potentially present volatility skew in real data, potentially even great model might fail such a test)

**Suggested models**:
Model 1.
* model receives an input of underlying’s historical returns and state embeddings for each point
* model receives another input with derivative’s parameters (time till maturity, strike, barrier level etc. — for MVP solution it is supposed that we train a separate model for each payoff type only)
* model receives (price of underlying, state embedding and time till maturity) and autoregressively generates $N$ vectors (each $\in \mathbb{R}^{K_{assets}}$)
loss function used is MSELoss — $$\min\limits_{{W}}(PnL_{derivative}-PnL_{portfolio}({W}))^2$$

**Architecture**:
Model 2.
Applying Reinforcement Learning for Derivatives Hedging, where reward function is the risk-adjusted PnL of the trading book.

**Experiments outline**:
* One underlying, two assets in replicating portfolio (spot + risk-free asset) fixed market data (spot prices, interest rates) — linear payoff (Forward).
* One underlying, two assets in replicating portfolio (spot + risk-free asset), fixed market data (spot prices, interest rates, historical volatility) — vanilla non-linear payoff (European Call, European Put).
* One underlying, real market data for each point of time — vanilla non-linear payoff (European Call, European Put).
* Application of real-world constraints (transaction costs — especially, real FX SWAP market data, short-selling, market impact etc.).
* One underlying, several assets in replicating portfolio (up to all assets that are available for trading), real market data for each point of time — vanilla non-linear payoff (European Call, European Put).
* One underlying, several assets in replicating portfolio (up to all assets that are available for trading), real market data for each point of time — exotic derivatives (start with Barrier Options).
* One underlying, several assets in replicating portfolio (up to all assets that are available for trading), real market data for each point of time — path-dependent exotic derivatives (start with Bermudian Knock-Out Options).
* Application of Reinforcement Learning.
* ! Correct adjustments for tail-event contingency, introduction of VaR restrictions.

**Ideas for application of real-world constraints**:
1. Transaction costs:
* Base: Not mid-price, but bid-offer (without market impact).
* Advanced: Use market impact model (exogeneous).
2. Correct risk-free rates (implied rate from FX SWAP at correct bid-offer price):
* (?) add constraints for weights of replicating portfolio assets (in order to account for risk-management, regulatory, open vega / open gamma restrictions).

**Complications / To Be Researched**:
* compute gradient at each step of autoregressive generation, not for final PnL only
* deal with overfit for correlation in non-tail state of the world, when the model is allowed to hedge by not only underlying asset, but any other asset available (as in tail event such hedge can produce unpredictably large loss / gain => until model behavior is stabilized, model is expected to produce cheap hedging in non-tail state of the world due to leverage tail event
* produce correct torch.DataLoader logic that allows to use real market data only and have sufficient number of points for training (e.g., produce batches via shifting window by some  and then shuffle in order to avoid path-dependence in optimization — expected to achieve high enough level of generalization, outputting regime-independent solution.
