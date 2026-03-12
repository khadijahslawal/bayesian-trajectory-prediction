# Bayesian Trajectory Prediction for Autonomous Driving Safety

A Bayesian machine learning project implementing uncertainty-quantified pedestrian trajectory prediction for autonomous vehicle safety decisions. The project compares three approaches — a deterministic Baseline LSTM, Monte Carlo Dropout, and Variational Bayesian Neural Networks — evaluated on the ETH/UCY pedestrian datasets.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Safety Framework](#safety-framework)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)

---

## Project Overview

Autonomous vehicles must predict pedestrian trajectories to make safe navigation decisions. Deterministic models produce a single predicted path with no indication of how confident that prediction is — a model that is confidently wrong is more dangerous than one that flags its own uncertainty.

This project implements **Bayesian uncertainty quantification** for trajectory prediction, enabling an autonomous vehicle to ask not just *where will this pedestrian go?* but *how certain are we about that prediction?* High uncertainty triggers conservative safety decisions (yield, slow down); low uncertainty allows the vehicle to proceed.

**Primary method:** Monte Carlo Dropout — approximate Bayesian inference via stochastic forward passes at inference time.

**Secondary method:** Variational Bayesian Neural Networks via Pyro — principled Bayesian inference by learning distributions over network weights using ELBO optimisation.

**Evaluation:** Standard ADE/FDE metrics on the ZARA2 held-out test scene, following the leave-one-out protocol established by Social Force and Social GAN.

---

## Repository Structure

```
bayesian-trajectory-prediction/
│
├── data/
│   └── raw/                        # ETH/UCY dataset files
│       └── raw/
│           ├── train/              # Training splits
│           ├── val/                # Validation splits
│           └── test/               # Test splits
│
├── src/
│   ├── __init__.py
│   └── data_loader.py              # ScenesDataLoader, trajectory extraction, normalisation
│
├── baseline/
│   ├── __init__.py
│   ├── baseline_lstm.py            # Deterministic LSTM baseline
│   ├── models/
│   │   └── baseline_best.pt        # Saved best model weights
│   └── results/
│       └── training_results.csv    # Training history
│
├── mc_dropout/
│   ├── __init__.py
│   ├── mcmc.py                     # MC Dropout LSTM + training + evaluation
│   ├── models/
│   │   └── mc_dropout_best.pt      # Saved best model weights
│   └── results/
│       └── training_results.csv    # Training history
│
├── variational_bnn/
│   ├── __init__.py
│   ├── bnn.py                      # Variational BNN via Pyro + training + evaluation
│   ├── models/
│   │   ├── vbnn_best.pt            # Saved best model weights
│   │   └── vbnn_params.pt          # Saved Pyro parameter store
│   └── results/
│       └── training_results.csv    # Training history
│
├── safety/
│   ├── safety_analysis.ipynb       # Full safety framework analysis notebook
│   └── plots/                      # Generated visualisations
│       ├── mc_dropout_uncertainty_fans.png
│       ├── variational_bnn_uncertainty_fans.png
│       ├── uncertainty_over_time.png
│       ├── mc_dropout_calibration.png
│       ├── variational_bnn_calibration.png
│       ├── comparison_table.png
│       └── crossing_scenario.png
│
└── README.md
```

---

## Dataset

The project uses the **ETH/UCY pedestrian trajectory datasets**, sourced from the Trajectron++ repository. These are standard benchmarks in trajectory prediction research.

| Scene | Split | Samples |
|---|---|---|
| ETH | train / val / test | ✓ |
| Hotel | train / val / test | ✓ |
| UNIV | train / val | ✓ |
| ZARA1 | train / val / test | ✓ |
| ZARA2 | train / val / test | ✓ (held-out test) |
| ZARA3 | train / val | ✓ |
| Students001 | train / val / test | ✓ |
| Students003 | train / val / test | ✓ |

**Training scenes:** ETH, Hotel, UNIV, ZARA1, Students001 (15,328 samples)

**Test scene:** ZARA2 — held out entirely during training to assess generalisation

**Trajectory format:** Each sample consists of 8 observed timesteps and 12 prediction timesteps (0.4s per step → 3.2s observation, 4.8s prediction window).

**Normalisation:** Observations are expressed relative to the first observed position; predictions are expressed relative to the last observed position. This centres each trajectory around the origin and removes absolute coordinate dependence, making the learning problem consistent across scenes.

---

## Models

### Baseline LSTM

A standard encoder-decoder LSTM with no dropout. Encodes 8 observed displacement steps into a hidden state, then auto-regressively decodes 12 predicted steps. Produces a single deterministic trajectory — no uncertainty estimate.

```
Encoder: LSTM (input=2, hidden=128, layers=2)
Decoder: LSTMCell (input=2, hidden=128) × 12 steps
Output:  Linear (128 → 2)
Loss:    ADE (Average Displacement Error)
```

### MC Dropout LSTM

Identical architecture to the baseline but with dropout layers that remain **active at inference time**. Each forward pass samples a different dropout mask, effectively sampling a different network from the approximate posterior over weights. Running 50 stochastic forward passes produces a distribution of trajectory predictions — the mean is the predicted trajectory, the variance is the epistemic uncertainty estimate.

```
Encoder: LSTM (input=2, hidden=128, layers=2, dropout=0.3)
Decoder: LSTMCell + Dropout(0.3) × 12 steps
MC samples: 50 stochastic forward passes at inference
Uncertainty: variance across 50 samples
```

The key implementation detail is `enable_dropout()` — PyTorch's `.eval()` disables dropout by default, so dropout layers must be explicitly kept in `.train()` mode during inference to enable stochastic sampling.

### Variational BNN (Pyro)

Extends the LSTM architecture with a Bayesian output layer implemented via the [Pyro](http://pyro.ai) probabilistic programming library. Rather than point-estimate weights, the output layer learns a distribution over weights — specifically, a Normal distribution with learnable mean (`w_mu`) and variance (`softplus(w_rho)`) for each parameter.

Training maximises the Evidence Lower Bound (ELBO), which balances fitting the data against staying close to the weight prior. At inference, 50 samples are drawn from the learned posterior, producing a distribution of predictions analogous to MC Dropout.

```
Encoder/Decoder: Deterministic LSTM (same as baseline)
Output layer:    Bayesian — weights ~ Normal(w_mu, softplus(w_rho))
Prior:           Normal(0, 1)
Training:        SVI with Trace_ELBO
Inference:       50 posterior weight samples via guide trace
```

---

## Results

All models trained for 100 epochs with Adam optimiser and ReduceLROnPlateau scheduler. Evaluated on ZARA2 test set.

| Model | Best ADE | Best FDE | Uncertainty | Inference samples |
|---|---|---|---|---|
| Baseline LSTM | 0.4207 | 0.8956 | N/A | 1 |
| MC Dropout | **0.4209** | **0.8892** | 0.031 | 50 |
| Variational BNN | 0.5813 | 1.0315 | 0.003 | 50 |

**ADE** (Average Displacement Error): mean L2 distance between predicted and ground truth positions across all timesteps (metres, normalised).

**FDE** (Final Displacement Error): L2 distance at the final prediction timestep only.

MC Dropout matches the deterministic baseline on both metrics while adding meaningful uncertainty quantification — demonstrating that the Bayesian approach costs nothing in predictive performance. Variational BNN underperforms on raw metrics but still provides valid uncertainty estimates.

---

## Safety Framework

The safety framework (`safety/safety_analysis.ipynb`) loads the trained models, runs predictions on the ZARA2 test set, and produces uncertainty-aware safety decisions.

### Safety Score

Each prediction is assigned a scalar safety score in [0, 1] based on its mean epistemic uncertainty across the prediction horizon:

```
safety_score = 1 - clamp(mean_uncertainty / unsafe_threshold, 0, 1)
```

Scores are classified into three tiers:

| Classification | MC Dropout threshold | Meaning |
|---|---|---|
| SAFE | uncertainty < 0.020 | Proceed normally |
| CAUTION | 0.020 – 0.040 | Reduce speed |
| UNSAFE | uncertainty > 0.040 | Yield / stop |

### Visualisations

**Uncertainty Fans** — Each predicted trajectory is shown as a fan of 50 individual MC samples around the mean prediction. Wider fans indicate higher uncertainty. MC Dropout produces visibly wider fans than Variational BNN, reflecting its higher and more informative uncertainty scale.

**Uncertainty Over Time** — Both models show monotonically increasing uncertainty across the 12 prediction steps, which is the theoretically correct behaviour: the further into the future, the less certain the model should be. MC Dropout grows from ~0.003 at step 1 to ~0.070 at step 12 (20× increase over 4.8 seconds). This property directly supports safety-critical decision making — near-term predictions should be trusted more than far-term ones.

**Calibration Plots** — A well-calibrated model's predicted uncertainty should correlate with actual prediction error. MC Dropout shows a positive trend between uncertainty and ADE error, indicating partial calibration. Variational BNN's uncertainty range is too compressed to be informative, with most predictions clustered near-zero uncertainty regardless of error.

**ETH Crossing Scenario** — Demonstrates the practical safety decision pipeline. A pedestrian with low uncertainty (0.00012) triggers **PROCEED**; a pedestrian with high uncertainty (0.12925) triggers **YIELD / SLOW DOWN**. This is the core application: uncertainty quantification enables the autonomous vehicle to adapt its behaviour to prediction confidence rather than always acting on a single point estimate.

**Comparison Table** — MC Dropout and Variational BNN produce similar safety classification counts (22-23 SAFE, 2 CAUTION, 7-8 UNSAFE per batch) despite different uncertainty scales. MC Dropout's mean safety score of 0.65 vs Variational BNN's 0.49 reflects MC Dropout's more conservative uncertainty estimates — appropriate for safety-critical applications.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/bayesian-trajectory-prediction.git
cd bayesian-trajectory-prediction

# Install dependencies
pip install torch numpy pandas matplotlib pyro-ppl jupyter

# Download ETH/UCY data from the Trajectron++ repository
# Place files in data/raw/raw/{train,val,test}/
```

---

## Usage

### Train all models

```bash
# Baseline LSTM
python baseline/baseline_lstm.py

# MC Dropout
python mc_dropout/mcmc.py

# Variational BNN
python variational_bnn/bnn.py
```

### Run safety analysis

```bash
cd safety
jupyter notebook safety_analysis.ipynb
```

Run all cells top to bottom. All plots are saved to `safety/plots/`.

### Load a trained model for inference

```python
import torch
from mc_dropout.mcmc import MCDropoutLSTM, mc_predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MCDropoutLSTM(input_dim=2, hidden_dim=128, pred_len=12,
                      num_layers=2, dropout_p=0.3).to(device)
model.load_state_dict(torch.load('mc_dropout/models/mc_dropout_best.pt'))

# obs: (B, 8, 2) tensor of observed relative displacements
mean, variance, samples = mc_predict(model, obs, n_samples=50)
# mean:     (B, 12, 2) — predicted trajectory
# variance: (B, 12, 2) — epistemic uncertainty per step
# samples:  (50, B, 12, 2) — individual stochastic predictions
```

---

## Key Findings

**1. MC Dropout matches deterministic performance while adding uncertainty.**
The Baseline LSTM achieves FDE 0.8956; MC Dropout achieves FDE 0.8892 — essentially identical — while producing meaningful epistemic uncertainty estimates at inference time. The Bayesian approach is not a trade-off; it is a free upgrade for safety-critical systems.

**2. Uncertainty grows with prediction horizon.**
Both Bayesian models show monotonically increasing uncertainty across the 12-step prediction window, confirming the models have learned that longer-horizon predictions are inherently less reliable. This is the correct behaviour for an autonomous driving safety system.

**3. MC Dropout is better calibrated than Variational BNN.**
MC Dropout's uncertainty estimates correlate more reliably with actual prediction error. Variational BNN's weight posteriors are too concentrated, producing uncertainty values too small to differentiate high- and low-confidence predictions effectively.

**4. MC Dropout is the recommended model for safety decisions.**
Higher uncertainty estimates, better calibration, and competitive ADE/FDE make MC Dropout the appropriate choice for downstream safety classification. This is consistent with the broader literature where MC Dropout frequently matches or outperforms more principled variational approaches in practice.

**5. Uncertainty quantification enables adaptive AV behaviour.**
The crossing scenario demonstrates that uncertainty-aware safety decisions are qualitatively different from deterministic predictions — the same model can appropriately choose to proceed or yield based on prediction confidence, a capability impossible with a point-estimate baseline.

---

## Team

Bayesian Machine Learning Course Project — March 2026
