# %%
"""
Variational BNN: Variational LSTM for Trajectory Prediction
============================================================
Architecture: Bayes-by-Backprop weight-sampled LSTM encoder + LSTMCell decoder
Loss:         ELBO  =  reconstruction (MSE) + KL divergence
Evaluation:   ADE, FDE, NLL, ECE   (matches MC-Dropout evaluation protocol)

Usage (from repo root):
    python variational_bnn/variational_bnn.py
"""
# %%
import sys
import os

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from src.data_loader import ScenesDataLoader

os.makedirs(os.path.join(REPO_ROOT, "variational_bnn/models"), exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# 1.  BAYESIAN LINEAR LAYER  (weight-space variational inference)
# ---------------------------------------------------------------------------

class BayesianLinear(nn.Module):
    """
    A linear layer whose weights are drawn from learned Gaussian posteriors.

    Posterior:  q(w) = N(mu_w, softplus(rho_w)^2)
    Prior:      p(w) = N(0, prior_std^2)

    During forward():
      - Training  → sample weights via reparameterisation trick
      - Inference → use posterior means (or sample if called explicitly)
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.prior_std    = prior_std

        # --- Posterior parameters ---
        # Weight mean & rho (rho → std via softplus)
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        # Bias mean & rho
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_rho   = nn.Parameter(torch.empty(out_features))

        self._init_parameters()

        # Track KL for the most recent forward pass
        self.kl: torch.Tensor = torch.tensor(0.0)

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3.0)   # softplus(-3) ≈ 0.049  (small initial std)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    @staticmethod
    def _softplus(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def _sample_weights(self):
        """Reparameterisation: w = mu + std * eps,  eps ~ N(0,1)."""
        w_std = self._softplus(self.weight_rho)
        b_std = self._softplus(self.bias_rho)

        w = self.weight_mu + w_std * torch.randn_like(self.weight_mu)
        b = self.bias_mu   + b_std * torch.randn_like(self.bias_mu)
        return w, b, w_std, b_std

    def _kl_divergence(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic KL(q || p) where p = N(0, prior_std^2)."""
        prior = Normal(
            torch.zeros_like(mu),
            torch.full_like(std, self.prior_std),
        )
        posterior = Normal(mu, std)
        return torch.distributions.kl_divergence(posterior, prior).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w, b, w_std, b_std = self._sample_weights()
            self.kl = (
                self._kl_divergence(self.weight_mu, w_std)
                + self._kl_divergence(self.bias_mu,   b_std)
            )
        else:
            # Mean-field prediction: use posterior mean
            w, b = self.weight_mu, self.bias_mu
            self.kl = torch.tensor(0.0)

        return F.linear(x, w, b)

# %%
# ---------------------------------------------------------------------------
# 2.  VARIATIONAL LSTM CELL
# ---------------------------------------------------------------------------

class VariationalLSTMCell(nn.Module):
    """
    LSTMCell whose input→hidden and hidden→hidden projections are Bayesian.
    All four gate projections share a single fused BayesianLinear for efficiency.
    """

    def __init__(self, input_size: int, hidden_size: int, prior_std: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # Fused: [i, f, g, o] gates in one matrix
        self.input_layer  = BayesianLinear(input_size,  4 * hidden_size, prior_std)
        self.hidden_layer = BayesianLinear(hidden_size, 4 * hidden_size, prior_std)

    @property
    def kl(self) -> torch.Tensor:
        return self.input_layer.kl + self.hidden_layer.kl

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = state
        gates = self.input_layer(x) + self.hidden_layer(h)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

# %%
# ---------------------------------------------------------------------------
# 3.  FULL VARIATIONAL LSTM MODEL
# ---------------------------------------------------------------------------

class VariationalLSTM(nn.Module):
    """
    Encoder  : standard nn.LSTM  (fast; encoder uncertainty is captured by the
                                   Bayesian decoder and output layer)
    Decoder  : VariationalLSTMCell — weight samples give predictive diversity
    Output   : BayesianLinear projecting hidden → (mu_x, mu_y, log_sigma_x, log_sigma_y)
               so the model outputs a Gaussian over each future position.
    """

    def __init__(
        self,
        input_dim:   int   = 2,
        hidden_dim:  int   = 64,
        pred_len:    int   = 12,
        num_enc_layers: int = 2,
        enc_dropout: float = 0.1,
        prior_std:   float = 0.1,
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

        # Bayesian decoder cell
        self.decoder_cell = VariationalLSTMCell(input_dim, hidden_dim, prior_std)

        # Output head: predicts mu and log_sigma for (x, y)
        # 4 outputs → [mu_x, mu_y, log_sigma_x, log_sigma_y]
        self.output_layer = BayesianLinear(hidden_dim, 4, prior_std)

    # ------------------------------------------------------------------
    def _collect_kl(self) -> torch.Tensor:
        """Sum KL terms across all Bayesian sub-layers."""
        return self.decoder_cell.kl + self.output_layer.kl

    # ------------------------------------------------------------------
    def forward(
        self,
        obs_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq : (B, obs_len, 2)

        Returns:
            mu        : (B, pred_len, 2)   predicted mean positions
            log_sigma : (B, pred_len, 2)   log std of positional Gaussians
            kl        : scalar             KL divergence for ELBO
        """
        B = obs_seq.size(0)

        # Encode
        _, (h_n, c_n) = self.encoder(obs_seq)
        h = h_n[-1]   # (B, H)  — top layer hidden state
        c = c_n[-1]

        dec_input = obs_seq[:, -1, :]   # seed decoder with last observed position

        mu_list        = []
        log_sigma_list = []

        for _ in range(self.pred_len):
            h, (h, c) = self.decoder_cell(dec_input, (h, c))
            out = self.output_layer(h)                    # (B, 4)
            mu_xy       = out[:, :2]                      # (B, 2)
            log_sigma   = out[:, 2:]                      # (B, 2)
            # Clamp log_sigma for numerical stability
            log_sigma   = log_sigma.clamp(-4.0, 2.0)

            mu_list.append(mu_xy)
            log_sigma_list.append(log_sigma)
            dec_input = mu_xy                             # autoregressive feed

        mu        = torch.stack(mu_list,        dim=1)    # (B, pred_len, 2)
        log_sigma = torch.stack(log_sigma_list, dim=1)    # (B, pred_len, 2)
        kl        = self._collect_kl()

        return mu, log_sigma, kl

# %%
# ---------------------------------------------------------------------------
# 4.  ELBO LOSS
# ---------------------------------------------------------------------------

def elbo_loss(
    mu:        torch.Tensor,
    log_sigma: torch.Tensor,
    target:    torch.Tensor,
    kl:        torch.Tensor,
    n_batches: int,
    beta:      float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO = E_q[log p(y|w)] - beta * KL(q||p)

    Reconstruction term:  Gaussian NLL over predicted positions.
    KL term:              Scaled by 1/n_batches (standard VI scaling).

    Args:
        mu        : (B, T, 2)  predicted means
        log_sigma : (B, T, 2)  predicted log-stds
        target    : (B, T, 2)  ground-truth future positions
        kl        : scalar     KL from model
        n_batches : int        number of batches in dataset (for KL scaling)
        beta      : float      KL weight (beta-VAE style; 1.0 = standard ELBO)

    Returns:
        loss   : scalar  total ELBO loss (negated, for minimisation)
        nll    : scalar  reconstruction NLL
        kl_val : scalar  raw KL term
    """
    sigma = torch.exp(log_sigma)                         # (B, T, 2)
    dist  = Normal(mu, sigma)
    nll   = -dist.log_prob(target).mean()                # mean over B, T, 2

    kl_scaled = (beta / n_batches) * kl
    loss      = nll + kl_scaled

    return loss, nll, kl_scaled

# %%
# ---------------------------------------------------------------------------
# 5.  PROBABILISTIC INFERENCE  (sample weights N times)
# ---------------------------------------------------------------------------

@torch.no_grad()
def variational_predict(
    model:    VariationalLSTM,
    obs_seq:  torch.Tensor,
    n_samples: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw n_samples weight samples from the posterior and aggregate.

    Returns:
        mean_mu    : (B, T, 2)           mean of sampled means
        mean_sigma : (B, T, 2)           mean of sampled stds (aleatoric)
        variance   : (B, T, 2)           variance of sampled means (epistemic)
        all_mu     : (n_samples, B, T, 2) all sampled trajectories
    """
    model.train()                          # enable weight sampling
    all_mu    = []
    all_sigma = []

    for _ in range(n_samples):
        mu, log_sigma, _ = model(obs_seq)
        all_mu.append(mu)
        all_sigma.append(torch.exp(log_sigma))

    model.eval()

    all_mu    = torch.stack(all_mu,    dim=0)    # (S, B, T, 2)
    all_sigma = torch.stack(all_sigma, dim=0)

    mean_mu    = all_mu.mean(dim=0)              # (B, T, 2)
    mean_sigma = all_sigma.mean(dim=0)
    variance   = all_mu.var(dim=0)               # epistemic uncertainty

    return mean_mu, mean_sigma, variance, all_mu

# %%
# ---------------------------------------------------------------------------
# 6.  EVALUATION METRICS: ADE, FDE, NLL, ECE
# ---------------------------------------------------------------------------

def compute_ece(
    all_mu:    torch.Tensor,
    all_sigma: torch.Tensor,
    target:    torch.Tensor,
    n_bins:    int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) for trajectory regression.

    Strategy: treat each predicted (mu, sigma) as a 1-D Gaussian per
    coordinate and check whether the true value falls within the predicted
    confidence interval at confidence level p.  Compare empirical vs
    nominal coverage across n_bins.

    Args:
        all_mu    : (S, B, T, 2)  sampled means
        all_sigma : (S, B, T, 2)  sampled stds (aleatoric)
        target    : (B, T, 2)
        n_bins    : int

    Returns:
        ece : float
    """
    # Use predictive mean and total std
    pred_mean = all_mu.mean(dim=0)                        # (B, T, 2)
    # Total predictive std = sqrt(E[sigma^2] + Var[mu])
    aleatoric  = (all_sigma ** 2).mean(dim=0)             # (B, T, 2)
    epistemic  = all_mu.var(dim=0)                        # (B, T, 2)
    pred_std   = (aleatoric + epistemic).sqrt().clamp(min=1e-6)

    # Standardised residuals
    z = ((target - pred_mean) / pred_std).abs()           # (B, T, 2)
    z_flat = z.reshape(-1).cpu().numpy()

    # Normal CDF: fraction of |z| <= z_alpha
    from scipy.stats import norm as scipy_norm

    confidence_levels = np.linspace(0.0, 1.0, n_bins + 1)[1:]   # (0.1, 0.2, …, 1.0)
    ece = 0.0
    for p in confidence_levels:
        z_alpha   = scipy_norm.ppf((1 + p) / 2)          # two-sided
        empirical = (z_flat <= z_alpha).mean()
        ece      += abs(empirical - p)

    return ece / n_bins


def evaluate(
    model:     VariationalLSTM,
    loader,
    device:    torch.device,
    n_samples: int = 50,
) -> dict:
    """
    Full evaluation: ADE, FDE, NLL, ECE.

    Positions are predicted as *displacements from last observation*
    (matching the normalisation in data_loader.py), so cumsum converts
    them to absolute offsets before distance computation.
    """
    ade_list = []
    fde_list = []
    nll_list = []
    all_mu_batches    = []
    all_sigma_batches = []
    target_batches    = []

    for obs, gt in loader:
        obs = obs.to(device)
        gt  = gt.to(device)

        mean_mu, mean_sigma, variance, all_mu = variational_predict(
            model, obs, n_samples=n_samples
        )

        # Gather for ECE
        # all_sigma: approximate from mean_sigma (mean over samples)
        # all_mu already is (S, B, T, 2) — slice to cpu for ECE
        S = all_mu.shape[0]
        mean_sigma_expanded = mean_sigma.unsqueeze(0).expand(S, -1, -1, -1)

        all_mu_batches.append(all_mu.cpu())
        all_sigma_batches.append(mean_sigma_expanded.cpu())
        target_batches.append(gt.cpu())

        # ADE / FDE in cumulative (absolute-offset) space
        mean_abs = mean_mu.cumsum(dim=1)
        gt_abs   = gt.cumsum(dim=1)

        disp = torch.norm(mean_abs - gt_abs, dim=-1)     # (B, T)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())

        # NLL: Gaussian NLL using predicted mean and sigma
        dist = Normal(mean_mu, mean_sigma.clamp(min=1e-6))
        nll  = -dist.log_prob(gt).mean().item()
        nll_list.append(nll)

    # ECE over full dataset
    all_mu_cat    = torch.cat([x.flatten(1, 2) for x in all_mu_batches],    dim=1)  # (S, N*T, 2)
    all_sigma_cat = torch.cat([x.flatten(1, 2) for x in all_sigma_batches], dim=1)
    target_cat    = torch.cat([x.reshape(-1, 2)  for x in target_batches],  dim=0)

    # Reshape for compute_ece: treat as single-batch (N*T samples, each dim=2)
    # Wrap as (S, 1, N*T, 2)
    all_mu_ece    = all_mu_cat.unsqueeze(1)
    all_sigma_ece = all_sigma_cat.unsqueeze(1)
    target_ece    = target_cat.unsqueeze(0).unsqueeze(0)    # (1, 1, N*T, 2)
    target_ece    = target_cat.reshape(1, 1, -1, 2).expand(1, 1, -1, 2)

    try:
        ece = compute_ece(all_mu_ece, all_sigma_ece, target_ece)
    except ImportError:
        ece = float('nan')
        print("  [Warning] scipy not found; ECE skipped.")

    return {
        'ADE': np.mean(ade_list),
        'FDE': np.mean(fde_list),
        'NLL': np.mean(nll_list),
        'ECE': ece,
    }

# %%
# ---------------------------------------------------------------------------
# 7.  TRAINING
# ---------------------------------------------------------------------------

def train_epoch(
    model:     VariationalLSTM,
    loader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    n_batches: int,
    beta:      float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_nll  = 0.0
    total_kl   = 0.0

    for obs, gt in loader:
        obs = obs.to(device)
        gt  = gt.to(device)

        optimizer.zero_grad()
        mu, log_sigma, kl = model(obs)

        loss, nll, kl_val = elbo_loss(mu, log_sigma, gt, kl, n_batches, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_nll  += nll.item()
        total_kl   += kl_val.item()

    n = len(loader)
    return {
        'loss': total_loss / n,
        'nll':  total_nll  / n,
        'kl':   total_kl   / n,
    }

# %%
# ---------------------------------------------------------------------------
# 8.  MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    loader_factory = ScenesDataLoader(data_root=os.path.join(repo_root, 'data/raw/'))
    train_loader = loader_factory.get_train_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'],
        batch_size=32,
    )
    val_loader = loader_factory.get_val_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'],
        batch_size=32,
    )
    n_batches = len(train_loader)
    print(f"Training samples : {len(train_loader.dataset)}")
    print(f"Val samples      : {len(val_loader.dataset)}")
    print(f"Batches per epoch: {n_batches}\n")

    # -------------------------------------------------------
    # Hyperparameter grid — add/remove values as needed
    # -------------------------------------------------------
    hyperparam_grid = [ #can add more hyperparameter combinations here
        {'num_epochs': 50,  'warmup': 10, 'beta_max': 1.0,  'hidden_dim': 64,  'lr': 1e-3},
        {'num_epochs': 50,  'warmup': 20, 'beta_max': 0.5,  'hidden_dim': 64,  'lr': 1e-3},
        {'num_epochs': 50, 'warmup': 20, 'beta_max': 1.0,  'hidden_dim': 128, 'lr': 1e-3},
        {'num_epochs': 50,  'warmup': 30, 'beta_max': 0.1,  'hidden_dim': 64,  'lr': 5e-4},
    ]

    all_results = []

    for run_idx, hp in enumerate(hyperparam_grid):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{len(hyperparam_grid)}")
        print(f"  num_epochs={hp['num_epochs']}  warmup={hp['warmup']}  "
              f"beta_max={hp['beta_max']}  hidden_dim={hp['hidden_dim']}  lr={hp['lr']}")
        print(f"{'='*60}\n")

        # Fresh model for each run
        model = VariationalLSTM(
            input_dim=2,
            hidden_dim=hp['hidden_dim'],
            pred_len=12,
            num_enc_layers=2,
            enc_dropout=0.1,
            prior_std=0.1,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        def get_beta(epoch: int) -> float:
            return min(hp['beta_max'], epoch / hp['warmup'])

        best_fde = float('inf')

        for epoch in range(1, hp['num_epochs'] + 1):
            beta = get_beta(epoch)
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, n_batches, beta=beta
            )
            scheduler.step()

            if epoch % 5 == 0:
                val_metrics = evaluate(model, val_loader, device, n_samples=50)
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Loss: {train_metrics['loss']:.4f}  "
                    f"NLL: {train_metrics['nll']:.4f}  "
                    f"KL: {train_metrics['kl']:.4f}  "
                    f"beta: {beta:.2f} | "
                    f"Val ADE: {val_metrics['ADE']:.4f}  "
                    f"FDE: {val_metrics['FDE']:.4f}  "
                    f"NLL: {val_metrics['NLL']:.4f}  "
                    f"ECE: {val_metrics['ECE']:.4f}"
                )

                if val_metrics['FDE'] < best_fde:
                    best_fde = val_metrics['FDE']
                    torch.save(
                        model.state_dict(),
                        f'variational_bnn/models/run{run_idx + 1}_best.pt',
                    )
                    print(f"    → Saved best model (FDE={best_fde:.4f})")

        # Final evaluation for this run
        print(f"\n  FINAL EVALUATION — RUN {run_idx + 1}")
        print(f"  {'-'*40}")
        final_metrics = evaluate(model, val_loader, device, n_samples=50)
        print(f"  ADE : {final_metrics['ADE']:.4f} meters")
        print(f"  FDE : {final_metrics['FDE']:.4f} meters")
        print(f"  NLL : {final_metrics['NLL']:.4f}")
        print(f"  ECE : {final_metrics['ECE']:.4f}")

        all_results.append({**hp, **final_metrics})

    # -------------------------------------------------------
    # Summary table across all runs
    # -------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'Run':<5} {'Epochs':<8} {'Warmup':<8} {'Beta':<6} {'HidDim':<8} {'LR':<8} {'ADE':<8} {'FDE':<8} {'NLL':<8} {'ECE':<8}")
    print(f"{'-'*75}")
    for i, r in enumerate(all_results):
        print(
            f"{i+1:<5} {r['num_epochs']:<8} {r['warmup']:<8} {r['beta_max']:<6} "
            f"{r['hidden_dim']:<8} {r['lr']:<8} {r['ADE']:<8.4f} {r['FDE']:<8.4f} "
            f"{r['NLL']:<8.4f} {r['ECE']:<8.4f}"
        )

    best_run = min(all_results, key=lambda x: x['FDE'])
    print(f"\nBest run by FDE: Run {all_results.index(best_run) + 1} → FDE={best_run['FDE']:.4f}")
    print(f"  Hyperparameters: {best_run}")
    print(f"{'='*60}")
    print("\nTraining complete.")