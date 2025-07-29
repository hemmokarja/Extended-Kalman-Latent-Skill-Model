# 📈 EKF Skill Rating System

A sophisticated player rating system based on Extended Kalman Filters (EKF) with Probit observation model for modeling latent skill and momentum in 1-vs-1 competitive matches. Unlike traditional rating systems that treat player ability as static, this approach recognizes that performance consists of two distinct components: persistent skill that evolves slowly over time, and transient momentum that captures recent form and psychological factors. The system continuously updates these latent variables as new match results arrive, providing probabilistic predictions that account for both the uncertainty in player abilities and the time elapsed since their last competition.

## 🚀 Features

- **Dynamic Skill Tracking**: Models both long-term skill and short-term momentum as separate latent variables
- **Time-Aware Updates**: Incorporates time gaps between matches for realistic skill evolution
- **Probabilistic Predictions**: Provides win probabilities rather than point estimates
- **Uncertainty Quantification**: Maintains uncertainty estimates for all player parameters
- **Flexible Framework**: Applicable to any 1-vs-1 binary outcome sport or competition
- **Efficient Implementation**: Fast fitting and prediction using vectorized operations

## 📚 Mathematical Foundation

### Problem Formulation

Kalman Filters provide the optimal Bayesian solution for tracking latent (hidden) variables through noisy observations over time. The fundamental challenge in player rating is that we cannot directly observe true skill levels - we only see match outcomes, which are influenced by both players' abilities plus random factors like luck, conditions, and psychological state.

The Kalman Filter framework addresses this by:
- **State Estimation**: Maintaining probabilistic beliefs about unobservable player abilities
- **Temporal Dynamics**: Modeling how these abilities evolve between observations
- **Optimal Fusion**: Combining prior knowledge with new evidence in a mathematically principled way
- **Uncertainty Quantification**: Tracking confidence in estimates and propagating it through time

The Extended Kalman Filter (EKF) extends this framework to handle nonlinear observation models, which is essential here since match outcomes follow a sigmoid-like probability function rather than a simple linear relationship.

### State Space Model

The system models each player's state as a 2D vector consisting of:
- **Skill** (μₛ): Long-term ability level
- **Momentum** (μₘ): Short-term performance boost/decline

The effective playing strength is: **Effective Skill = Skill + Momentum**

### State Evolution

Between matches, player states evolve according to:

```
x_{t+1} = F_t x_t + w_t
```

Where:
- **F_t** is the state transition matrix:
  ```
  F = [1    0   ]
      [0  e^(-λΔt)]
  ```
- **λ** is the momentum decay rate (momentum half-life ≈ ln(2)/λ)
- **Δt** is the time gap since the last match
- **w_t ~ N(0, Q_t)** is process noise with covariance:
  ```
  Q = [σ²_skill × Δt        0      ]
      [    0        σ²_momentum × Δt]
  ```

### Observation Model

Match outcomes follow a Probit model where the probability of player 1 defeating player 2 is:

```
P(player1 wins) = Φ((μ₁ - μ₂)/σ_match)
```

Where:
- **Φ** is the standard normal CDF
- **μᵢ** is player i's effective skill
- **σ_match** is the match outcome noise parameter

### Extended Kalman Filter Update

The EKF handles the nonlinear observation model through linearization:

1. **Prediction Step**: Time-propagate states and covariances
2. **Linearization**: Compute Jacobian of observation function h(x) = Φ((μ₁ - μ₂)/σ_match) with respect to state vector x = [skill₁, momentum₁, skill₂, momentum₂]:
   ```
   H = ∂h/∂x = [φ(δ)/σ_match, φ(δ)/σ_match, -φ(δ)/σ_match, -φ(δ)/σ_match]
   ```
   Where φ is the standard normal PDF and δ = (μ₁ - μ₂)/σ_match

3. **Innovation**: Compare predicted vs actual outcome
4. **Kalman Gain**: Optimal weighting of new information
5. **State Update**: Bayesian update of player parameters

### Covariance Structure

The system maintains full covariance matrices, tracking:
- Individual parameter uncertainties
- Cross-correlations between skill and momentum
- How uncertainty propagates through time

## 🛠️ Installation

This project uses the `uv` package manager. Install dependencies with:

```bash
uv sync
```

## 🧪 Examples

### Simulated Data
`example_sim.ipynb` demonstrates the model's performance on controlled simulated data:
- Generates matches with known ground truth skill levels
- Compares EKF predictions against maximum theoretical accuracy

### Real Tennis Data
`example.ipynb` applies the model to historical ATP tennis data:
- Loads all historical matches between 1995-2025 from web (approx. 100k matches)
- Compares prediction accuracy against official ATP rankings
- Demonstrates practical performance on real-world competitive data

## 🚀 Quick Start

```python
from ekf.ekf import EKFConfig, EKFSkillRating, Matches
from ekf import simulated_data

matches_df = simulated_data.simulate_matches(n_players=3, n_matches=100)

matches = Matches.from_pandas(matches_df)

config = EKFConfig()
ekf = EKFSkillRating(config)

history = model.fit(matches)

prob = model.predict_proba(
    "Player-0",
    "Player-1",
    player1_time_delta=7.0,  # days since last match
    player2_time_delta=14.0
)
```


## 📝 License

This project is licensed under the MIT License.
