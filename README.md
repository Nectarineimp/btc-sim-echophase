# EchoPhase: Harmonic-Weighted Empirical Block Bootstrap (Era 4)

## Overview

**EchoPhase** is a forward 12-month stochastic Monte Carlo simulation engine designed for Bitcoin price discovery in the mature institutional era. It synthesizes non-parametric moving block bootstrapping with harmonic spectral weighting to project forward probability distributions while preventing the exponential drift biases common to traditional continuous-time stochastic differential equations (SDEs).

Standard parametric models (such as Merton Jump-Diffusion, coupled Heston SDEs, or Ornstein-Uhlenbeck processes with linear drift) assume price is an autonomous compounding engine. In high-rate, institutional macro environments, these models repeatedly suffer from positive drift detachment, pushing median paths into unsustainable trajectories ($>\$200\text{k}\text{--}\$800\text{k}$) during prolonged periods of horizontal liquidity absorption.

EchoPhase solves this by establishing that **Bitcoin price discovery is stationary residual variance around a deterministic secular adoption backbone, modulated by cyclical institutional rebalancing windows**.

---

## Theoretical Architecture

### 1. The Deterministic Secular Backbone

Long-term adoption is anchored to genesis ($t_0 =$ January 3, 2009 18:15:05 UTC) via longitudinal power-law regression across all market history:

$$\ln(P_{\text{base}}(\tau)) = \alpha + \beta \ln(\tau)$$

Where $\tau$ is elapsed calendar time in days since genesis. This deterministic line represents structural monetization and fixed-supply monetary debasement absorption. It grows slowly and steadily in percentage terms, invariant to short-term headlines or high-frequency market noise.

### 2. Stationary Residual Space ($\epsilon_t$)

Rather than simulating raw price paths directly with geometric Brownian motion, EchoPhase computes the empirical log-residual series:

$$\epsilon_t = \ln(P_t) - (\alpha + \beta \ln(\tau_t))$$

The simulation draws exclusively from the first differences of this detrended series:

$$\Delta \epsilon_t = \epsilon_t - \epsilon_{t-1}$$

Because the expected value $\mathbb{E}[\Delta \epsilon_t] \approx 0$ throughout modern consolidation phases, the generative paths cannot run away exponentially.

### 3. Non-Parametric Era 4 Resampling

The innovation pool is strictly isolated to **Era 4 (The Spot ETF Regime: January 11, 2024 onward)**. This directly embeds the modern market microstructure characteristics documented in our cycle research:

* **Monotonic Variance Compression:** The contraction of the 5th/95th percentile return corridor and the collapse of tail volatility.
* **Order Book Institutionalization:** Dampened drawdowns and structural bid-ask depth created by daily ETF creation/redemption mechanics and algorithmic execution (TWAP/VWAP).
* **Volatility Clustering:** Contiguous block bootstrapping (default 7-day blocks) preserves empirical multi-day momentum persistence and autocorrelation without imposing parametric distribution assumptions.



### 4. Institutional Harmonic Phase Modulation ($T = 285\text{ Days}$)

Instead of assuming uniform random block selection ($1/N$), block sampling probabilities are dynamically conditioned on the phase of the sub-annual institutional cycle:

$$\theta(t) = \frac{2\pi \cdot t_{\text{mid}}}{T_{\text{cycle}}} + \phi$$

* **Dominant Periodicity ($T_{\text{cycle}} = 285.0\text{ days}$):** Extracted from Lomb-Scargle spectral density peaks representing the interaction of quarterly corporate reporting, model portfolio rebalancing review intervals, and seasonal capital allocation cycles.
* **Phase Offset ($\phi = -\frac{\pi}{2}$):** Anchors the start of the simulation (late May 2026) to the exhaustion/liquidation trough of the prior cycle. This prevents the model from assuming an immediate summer rally during macro liquidity slowdowns.


* **Soft Exponential Logit Weighting:** For each candidate block $i$ with normalized aggregate drift $z_i$, selection probability evaluates to:

$$\text{Logits}_i = \lambda \cdot \cos(\theta(t)) \cdot z_i$$

$$P(i) = \frac{\exp(\text{Logits}_i)}{\sum_j \exp(\text{Logits}_j)}$$

With logit temperature $\lambda = 0.18$, the engine gently favors expansion blocks during re-allocation windows and absorption/flat blocks during mandate trim windows, without starving sample diversity or inducing overfitting.

---

## Model Lineage & Benchmarking Suite

EchoPhase forms part of an out-of-sample forward benchmarking experiment (May 2026 – May 2027):

| Model | Classification | Core Mechanism | Status / Characteristic |
| --- | --- | --- | --- |
| **RegimeEcho**<br> | Non-Parametric

 | Era 4 Moving Block Bootstrap (Uniform)

 | Baseline conservative benchmark; flat linear quantile expansion.

 |
| **EchoPhase**<br> | Hybrid Semi-Parametric

 | Harmonic-Weighted Era 4 Block Bootstrap ($T = 285\text{d}$)

 | Captures sub-annual undulating wave while preserving empirical variance bounds.

 |
| **TrueTether**<br> | Continuous-Time SDE

 | Ornstein-Uhlenbeck Mean-Reversion + Power-Law Drift

 | Tests continuous mean-reversion strength against macro trendlines.

 |
| **TailWhip**<br> | Heavy-Tailed SDE

 | Merton Jump-Diffusion SDE

 | Tests jump intensity and skew under discontinuous shock regimes.

 |
| **Chamberlain** | Coupled Two-Factor SDE | Heston-style Price & Stochastic Volatility System ($\rho < 0$) | Tests endogenous volatility feedback and equity-style leverage effects. |

---

## Directory Structure & CLI Arguments

```
btc-sim-echophase/
├── pyproject.toml
├── poetry.lock
├── README.md
├── src/
│   └── simulate.py
└── output/
    └── monthly_forecast.csv

```

### Execution Interface

```bash
poetry run python src/simulate.py \
  --data ~/projects/data/btc_daily_price.csv \
  --iterations 10000 \
  --block-size 7 \
  --cycle-days 285.0 \
  --days 365 \
  --seed 42 \
  --output ./output/monthly_forecast.csv

```

### Output Schema

The output CSV conforms strictly to the 11-metric visualizer schema:

| Column | Description |
| --- | --- |
| `Month` | Calendar month label (`YYYY-MM`) |
| `Low_p05` | 5th percentile of path-wise monthly minimum prices |
| `Low_p16` | 16th percentile of path-wise monthly minimum prices ($1\sigma$ floor) |
| `Low_p50` | 50th percentile (median) of path-wise monthly minimum prices |
| `Low_p84` | 84th percentile of path-wise monthly minimum prices |
| `Low_p95` | 95th percentile of path-wise monthly minimum prices |
| `High_p05` | 5th percentile of path-wise monthly maximum prices |
| `High_p16` | 16th percentile of path-wise monthly maximum prices |
| `High_p50` | 50th percentile (median) of path-wise monthly maximum prices |
| `High_p84` | 84th percentile of path-wise monthly maximum prices ($1\sigma$ ceiling) |
| `High_p95` | 95th percentile of path-wise monthly maximum prices |

---

## Experimental Protocol & Scoring Criteria

1. **Parameter Freezing:** All model code, seeds, and coefficients are locked as of October 2026 to prevent confirmation bias or mid-cycle lookahead contamination.
2. **Scoring Formulation:** At month 12 (May 2027), performance will be evaluated via Continuous Ranked Probability Score (CRPS) and coverage calibration rates across realized monthly ranges, benchmarked against the qualitative macro friction ledger.