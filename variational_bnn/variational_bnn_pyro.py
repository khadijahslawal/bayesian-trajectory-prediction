"""
Variational BNN: Variational LSTM for Trajectory Prediction (Pyro)
===================================================================
Architecture: Pyro-based Bayesian LSTM encoder + LSTMCell decoder
Loss:         ELBO via Pyro's Trace_ELBO (automatic KL + NLL)
Evaluation:   ADE, FDE, NLL, ECE  (matches MC-Dropout evaluation protocol)

Install dependencies:
    pip install pyro-ppl scipy

Usage (from repo root):
    python variational_bnn/variational_bnn_pyro.py
# %%
"""

import sys
import os

# ---------------------------------------------------------------------------
# PATH SETUP  — works on any machine regardless of clone location
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import ClippedAdam

from src.data_loader import ScenesDataLoader

os.makedirs(os.path.join(REPO_ROOT, "variational_bnn/models"), exist_ok=True)

# Verify data exists
required_file = os.path.join(REPO_ROOT, 'data', 'raw', 'eth', 'train', 'biwi_hotel_train.txt')
if not os.path.exists(required_file):
    print("ERROR: Dataset not found. Please run:")
    print("  git clone https://github.com/StanfordASL/Trajectron-plus-plus.git /tmp/trajectron")
    print("  cp -r /tmp/trajectron/experiments/pedestrians/raw/* data/raw/")
    sys.exit(1)

# %%
# ---------------------------------------------------------------------------
# 1.  PYRO VARIATIONAL LSTM MODEL
# ---------------------------------------------------------------------------

class PyroVariationalLSTM(PyroModule):
    """
    Encoder  : standard nn.LSTM (deterministic, fast)
    Decoder  : nn.LSTMCell (deterministic)
    Output   : PyroModule[nn.Linear] with Bayesian weights via PyroSample
               — Pyro automatically handles prior, posterior, and KL

    The model() function defines the generative process p(y|w) * p(w).
    The guide() (AutoDiagonalNormal) defines the variational posterior q(w).
    Pyro's SVI + Trace_ELBO handles the ELBO automatically.
    """

    def __init__(
        self,
        input_dim:      int   = 2,
        hidden_dim:     int   = 64,
        pred_len:       int   = 12,
        num_enc_layers: int   = 2,
        enc_dropout:    float = 0.1,
        prior_std:      float = 0.1,
    ):
        super().__init__()
        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim

        # Deterministic encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_enc_layers,
            batch_first=True,
            dropout=enc_dropout if num_enc_layers > 1 else 0.0,
        )

        # Deterministic decoder cell
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)

        # Bayesian output layer — Pyro manages prior & posterior automatically
        # 4 outputs: [mu_x, mu_y, log_sigma_x, log_sigma_y]
        self.output_layer = PyroModule[nn.Linear](hidden_dim, 4)
        self.output_layer.weight = PyroSample(
            dist.Normal(0.0, prior_std)
                .expand([4, hidden_dim])
                .to_event(2)
        )
        self.output_layer.bias = PyroSample(
            dist.Normal(0.0, prior_std)
                .expand([4])
                .to_event(1)
        )

    def _decode(self, obs_seq: torch.Tensor):
        """
        Shared encode + decode logic used by both model() and predict().
        Returns mu (B, T, 2) and log_sigma (B, T, 2).
        """
        B = obs_seq.size(0)
        _, (h_n, c_n) = self.encoder(obs_seq)
        h = h_n[-1]   # top encoder layer hidden state  (B, H)
        c = c_n[-1]

        dec_input = obs_seq[:, -1, :]   # seed with last observed position

        mu_list        = []
        log_sigma_list = []

        for _ in range(self.pred_len):
            h, c    = self.decoder_cell(dec_input, (h, c))
            out     = self.output_layer(h)              # (B, 4)
            mu_xy   = out[:, :2]                        # (B, 2)
            log_sig = out[:, 2:].clamp(-4.0, 2.0)      # (B, 2)

            mu_list.append(mu_xy)
            log_sigma_list.append(log_sig)
            dec_input = mu_xy                           # autoregressive

        mu        = torch.stack(mu_list,        dim=1)  # (B, T, 2)
        log_sigma = torch.stack(log_sigma_list, dim=1)  # (B, T, 2)
        return mu, log_sigma

    def forward(self, obs_seq: torch.Tensor, target: torch.Tensor = None):
        """
        Pyro model: defines p(y | w) * p(w).

        Args:
            obs_seq : (B, obs_len, 2)
            target  : (B, pred_len, 2)  — None at inference time
        """
        B = obs_seq.size(0)
        mu, log_sigma = self._decode(obs_seq)
        sigma = log_sigma.exp()

        # Pyro likelihood — this is where ELBO reconstruction term comes from
        # to_event(2) treats (T, 2) as a single multivariate observation
        with pyro.plate("data", B):
            pyro.sample(
                "obs",
                dist.Normal(mu, sigma).to_event(2),
                obs=target,
            )

        return mu, log_sigma

# %%
# ---------------------------------------------------------------------------
# 2.  PROBABILISTIC INFERENCE
# ---------------------------------------------------------------------------

def variational_predict(
    model:     PyroVariationalLSTM,
    guide:     AutoDiagonalNormal,
    obs_seq:   torch.Tensor,
    n_samples: int = 50,
) -> tuple:
    """
    Draw n_samples weight samples from the guide posterior and aggregate.

    Returns:
        mean_mu    : (B, T, 2)            mean of sampled trajectory means
        mean_sigma : (B, T, 2)            mean of sampled stds (aleatoric)
        variance   : (B, T, 2)            variance of sampled means (epistemic)
        all_mu     : (n_samples, B, T, 2) all sampled trajectories
    """
    all_mu    = []
    all_sigma = []

    # Use Pyro's predictive to sample from the posterior
    predictive = pyro.infer.Predictive(
        model, guide=guide, num_samples=n_samples, return_sites=["_RETURN"]
    )

    with torch.no_grad():
        for _ in range(n_samples):
            # Sample weights from guide, run forward pass
            guide_trace = pyro.poutine.trace(guide).get_trace(obs_seq)
            model_trace = pyro.poutine.trace(
                pyro.poutine.replay(model, trace=guide_trace)
            ).get_trace(obs_seq, None)

            mu, log_sigma = model_trace.nodes["_RETURN"]["value"]
            all_mu.append(mu)
            all_sigma.append(log_sigma.exp())

    all_mu    = torch.stack(all_mu,    dim=0)   # (S, B, T, 2)
    all_sigma = torch.stack(all_sigma, dim=0)

    mean_mu    = all_mu.mean(dim=0)             # (B, T, 2)
    mean_sigma = all_sigma.mean(dim=0)
    variance   = all_mu.var(dim=0)              # epistemic uncertainty

    return mean_mu, mean_sigma, variance, all_mu

# %%
# ---------------------------------------------------------------------------
# 3.  EVALUATION METRICS: ADE, FDE, NLL, ECE
# ---------------------------------------------------------------------------

def compute_ece(
    all_mu:    torch.Tensor,
    all_sigma: torch.Tensor,
    target:    torch.Tensor,
    n_bins:    int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) for trajectory regression.

    Checks empirical vs nominal coverage of the predictive Gaussian
    at n_bins confidence levels and averages the absolute gap.

    Args:
        all_mu    : (S, B, T, 2)
        all_sigma : (S, B, T, 2)
        target    : (B, T, 2)
        n_bins    : int

    Returns:
        ece : float
    """
    from scipy.stats import norm as scipy_norm

    pred_mean  = all_mu.mean(dim=0)                         # (B, T, 2)
    aleatoric  = (all_sigma ** 2).mean(dim=0)
    epistemic  = all_mu.var(dim=0)
    pred_std   = (aleatoric + epistemic).sqrt().clamp(min=1e-6)

    z      = ((target - pred_mean) / pred_std).abs()
    z_flat = z.reshape(-1).cpu().numpy()

    confidence_levels = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    ece = 0.0
    for p in confidence_levels:
        z_alpha   = scipy_norm.ppf((1 + p) / 2)
        empirical = (z_flat <= z_alpha).mean()
        ece      += abs(empirical - p)

    return ece / n_bins


def evaluate(
    model:     PyroVariationalLSTM,
    guide:     AutoDiagonalNormal,
    loader,
    device:    torch.device,
    n_samples: int = 50,
) -> dict:
    """
    Full evaluation: ADE, FDE, NLL, ECE.

    Positions are predicted as displacements from last observation
    (matching normalisation in data_loader.py), so cumsum converts
    them to absolute offsets before distance computation.
    """
    ade_list          = []
    fde_list          = []
    nll_list          = []
    all_mu_batches    = []
    all_sigma_batches = []
    target_batches    = []

    for obs, gt in loader:
        obs = obs.to(device)
        gt  = gt.to(device)

        mean_mu, mean_sigma, variance, all_mu = variational_predict(
            model, guide, obs, n_samples=n_samples
        )

        S = all_mu.shape[0]
        mean_sigma_expanded = mean_sigma.unsqueeze(0).expand(S, -1, -1, -1)

        all_mu_batches.append(all_mu.cpu())
        all_sigma_batches.append(mean_sigma_expanded.cpu())
        target_batches.append(gt.cpu())

        # ADE / FDE in cumulative absolute-offset space
        mean_abs = mean_mu.cumsum(dim=1)
        gt_abs   = gt.cumsum(dim=1)

        disp = torch.norm(mean_abs - gt_abs, dim=-1)       # (B, T)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())

        # NLL with predicted mean and sigma
        nll_dist = Normal(mean_mu, mean_sigma.clamp(min=1e-6))
        nll_list.append(-nll_dist.log_prob(gt).mean().item())

    # ECE over full dataset
    all_mu_cat    = torch.cat([x.flatten(1, 2) for x in all_mu_batches],    dim=1)
    all_sigma_cat = torch.cat([x.flatten(1, 2) for x in all_sigma_batches], dim=1)
    target_cat    = torch.cat([x.reshape(-1, 2) for x in target_batches],   dim=0)

    all_mu_ece    = all_mu_cat.unsqueeze(1)
    all_sigma_ece = all_sigma_cat.unsqueeze(1)
    target_ece    = target_cat.reshape(1, 1, -1, 2).expand(1, 1, -1, 2)

    try:
        ece = compute_ece(all_mu_ece, all_sigma_ece, target_ece)
    except ImportError:
        ece = float('nan')
        print("  [Warning] scipy not found — ECE skipped. Run: pip install scipy")

    return {
        'ADE': np.mean(ade_list),
        'FDE': np.mean(fde_list),
        'NLL': np.mean(nll_list),
        'ECE': ece,
    }

# %%
# ---------------------------------------------------------------------------
# 4.  TRAINING  (Pyro SVI — replaces manual loss.backward())
# ---------------------------------------------------------------------------

def train_epoch(
    svi:    SVI,
    loader,
    device: torch.device,
) -> float:
    """
    One epoch of SVI training.
    Pyro's SVI.step() handles: forward pass, ELBO, KL, backward, optimizer step.

    Returns:
        avg_loss : float  average ELBO loss over the epoch
    """
    total_loss = 0.0

    for obs, gt in loader:
        obs = obs.to(device)
        gt  = gt.to(device)
        # svi.step() = forward + ELBO + KL + backward + optimizer step
        loss = svi.step(obs, gt)
        total_loss += loss

    return total_loss / len(loader)

# %%
# ---------------------------------------------------------------------------
# 5.  MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cpu')   # MPS has known issues with custom LSTM ops
    print(f"Using device: {device}")

    # Data
    loader_factory = ScenesDataLoader(data_root=os.path.join(REPO_ROOT, 'data/raw/'))
    train_loader = loader_factory.get_train_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'],
        batch_size=32,
    )
    val_loader = loader_factory.get_val_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'],
        batch_size=32,
    )
    print(f"Training samples : {len(train_loader.dataset)}")
    print(f"Val samples      : {len(val_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}\n")

    # -----------------------------------------------------------------------
    # Hyperparameter grid — add/remove combinations as needed
    # -----------------------------------------------------------------------
    hyperparam_grid = [
        {'num_epochs': 50, 'hidden_dim': 64,  'prior_std': 0.1,  'lr': 1e-3},
        {'num_epochs': 50, 'hidden_dim': 64,  'prior_std': 0.05, 'lr': 1e-3},
        {'num_epochs': 50, 'hidden_dim': 128, 'prior_std': 0.1,  'lr': 5e-4},
        {'num_epochs': 50, 'hidden_dim': 64,  'prior_std': 0.1,  'lr': 5e-4},
    ]

    all_results = []

    for run_idx, hp in enumerate(hyperparam_grid):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(hyperparam_grid)}")
        print(f"  num_epochs={hp['num_epochs']}  hidden_dim={hp['hidden_dim']}  "
              f"prior_std={hp['prior_std']}  lr={hp['lr']}")
        print(f"{'='*60}\n")

        # Clear Pyro param store between runs so they don't share state
        pyro.clear_param_store()

        # Fresh model + guide for each run
        model = PyroVariationalLSTM(
            input_dim=2,
            hidden_dim=hp['hidden_dim'],
            pred_len=12,
            num_enc_layers=2,
            enc_dropout=0.1,
            prior_std=hp['prior_std'],
        ).to(device)

        # AutoDiagonalNormal: mean-field Gaussian guide over all Bayesian params
        # This replaces your manual BayesianLinear posterior parameterisation
        guide = AutoDiagonalNormal(model)

        # ClippedAdam: Adam with gradient clipping built in (replaces clip_grad_norm_)
        optimizer = ClippedAdam({"lr": hp['lr'], "clip_norm": 1.0,
                                  "lrd": 0.5 ** (1 / hp['num_epochs'])})

        # SVI handles ELBO = NLL + KL automatically
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        best_fde = float('inf')

        for epoch in range(1, hp['num_epochs'] + 1):
            avg_loss = train_epoch(svi, train_loader, device)

            if epoch % 5 == 0:
                val_metrics = evaluate(model, guide, val_loader, device, n_samples=50)
                print(
                    f"  Epoch {epoch:3d} | "
                    f"ELBO Loss: {avg_loss:.4f} | "
                    f"Val ADE: {val_metrics['ADE']:.4f}  "
                    f"FDE: {val_metrics['FDE']:.4f}  "
                    f"NLL: {val_metrics['NLL']:.4f}  "
                    f"ECE: {val_metrics['ECE']:.4f}"
                )

                if val_metrics['FDE'] < best_fde:
                    best_fde = val_metrics['FDE']
                    # Save both model weights and guide (posterior) params
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            'pyro_params': pyro.get_param_store().get_state(),
                            'hp': hp,
                        },
                        os.path.join(REPO_ROOT, f'variational_bnn/models/pyro_run{run_idx + 1}_best.pt'),
                    )
                    print(f"    → Saved best model (FDE={best_fde:.4f})")

        # Final evaluation for this run
        print(f"\n  FINAL EVALUATION — RUN {run_idx + 1}")
        print(f"  {'-'*40}")
        final_metrics = evaluate(model, guide, val_loader, device, n_samples=50)
        print(f"  ADE : {final_metrics['ADE']:.4f} meters")
        print(f"  FDE : {final_metrics['FDE']:.4f} meters")
        print(f"  NLL : {final_metrics['NLL']:.4f}")
        print(f"  ECE : {final_metrics['ECE']:.4f}")

        all_results.append({**hp, **final_metrics})

    # -----------------------------------------------------------------------
    # Summary table across all runs
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<5} {'Epochs':<8} {'HidDim':<8} {'Prior':<7} {'LR':<8} {'ADE':<8} {'FDE':<8} {'NLL':<8} {'ECE':<8}")
    print(f"{'-'*70}")
    for i, r in enumerate(all_results):
        print(
            f"{i+1:<5} {r['num_epochs']:<8} {r['hidden_dim']:<8} {r['prior_std']:<7} "
            f"{r['lr']:<8} {r['ADE']:<8.4f} {r['FDE']:<8.4f} "
            f"{r['NLL']:<8.4f} {r['ECE']:<8.4f}"
        )

    best_run = min(all_results, key=lambda x: x['FDE'])
    print(f"\nBest run by FDE: Run {all_results.index(best_run) + 1} → FDE={best_run['FDE']:.4f}")
    print(f"  Hyperparameters: {best_run}")
    print(f"{'='*70}")
    print("\nTraining complete.")

## % 
'''
Here's what changed compared to your original file:
Model — PyroVariationalLSTM extends PyroModule instead of nn.Module. The output layer uses PyroSample to declare Bayesian weights, so Pyro automatically manages the prior p(w) and registers parameters for the posterior q(w). You no longer need BayesianLinear or VariationalLSTMCell — Pyro handles all of that.
Loss — Replaced your manual elbo_loss() and loss.backward() with Pyro's SVI + Trace_ELBO. svi.step() does the full forward pass, computes ELBO (NLL + KL), runs backprop, and updates weights in one call. This also fixes your RuntimeError from before.
Guide — AutoDiagonalNormal replaces your manual weight_mu/weight_rho parameters. It automatically creates a mean-field Gaussian posterior over every Bayesian parameter in the model.
Optimizer — ClippedAdam replaces torch.optim.Adam + clip_grad_norm_. It has gradient clipping built in and a built-in learning rate decay (lrd), so you no longer need the StepLR scheduler.
Saving — Each checkpoint now saves both model.state_dict() and pyro.get_param_store() since the guide's posterior parameters live in Pyro's param store, not the model's state dict.
Hyperparameter grid — warmup and beta_max are removed since Pyro handles KL scheduling internally. The new tunable params are hidden_dim, prior_std, and lr.
'''