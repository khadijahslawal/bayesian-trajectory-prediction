import os
import time
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data_loader import ScenesDataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MCDropoutLSTM(nn.Module):
    """
    LSTM trajectory predictor with MC Dropout.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 2,
        pred_len: int = 12,
        num_layers: int = 2,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq: (B, obs_len, 2)

        Returns:
            pred_seq: (B, pred_len, 2)
        """
        _, (h_n, c_n) = self.encoder(obs_seq)

        h = self.dropout(h_n[-1])
        c = c_n[-1]
        dec_input = obs_seq[:, -1, :]

        preds = []
        for _ in range(self.pred_len):
            h, c = self.decoder_cell(dec_input, (h, c))
            h = self.dropout(h)
            out = self.output_layer(h)
            preds.append(out)
            dec_input = out

        return torch.stack(preds, dim=1)

    def enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()


@torch.no_grad()
def mc_predict(
    model: MCDropoutLSTM,
    obs_seq: torch.Tensor,
    n_samples: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    model.enable_dropout()

    sample_preds = torch.stack([model(obs_seq) for _ in range(n_samples)], dim=0)
    mean = sample_preds.mean(dim=0)
    variance = sample_preds.var(dim=0, unbiased=False)

    return mean, variance, sample_preds


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for obs, gt in loader:
        obs = obs.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        pred = model(obs)
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    n_samples: int = 50,
) -> Dict[str, float]:
    model.eval()

    ade_list = []
    fde_list = []
    unc_list = []

    for obs, gt in loader:
        obs = obs.to(device)
        gt = gt.to(device)

        mean, variance, _ = mc_predict(model, obs, n_samples=n_samples)

        mean_abs = mean.cumsum(dim=1)
        gt_abs = gt.cumsum(dim=1)

        disp = torch.norm(mean_abs - gt_abs, dim=-1)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())
        unc_list.append(variance.mean().item())

    return {
        "ADE": float(np.mean(ade_list)),
        "FDE": float(np.mean(fde_list)),
        "mean_uncertainty": float(np.mean(unc_list)),
    }


# -----------------------------
# Training block (rollback)
# -----------------------------
set_seed(42)
os.makedirs("mc_dropout/models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

loader = ScenesDataLoader(data_root="data/raw")
scenes = ["eth", "hotel", "univ", "zara1", "students1"]

train_loader = loader.get_train_loader(scenes=scenes, batch_size=32, shuffle=True)
val_loader = loader.get_val_loader(scenes=scenes, batch_size=32)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")

model = MCDropoutLSTM(
    input_dim=2,
    hidden_dim=64,
    pred_len=12,
    num_layers=2,
    dropout_p=0.3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

best_fde = float("inf")
num_epochs = 50
overall_start = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    )
    scheduler.step()

    epoch_time = time.time() - epoch_start

    if epoch % 5 == 0 or epoch == 1:
        metrics = evaluate(model, val_loader, device, n_samples=50)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"ADE: {metrics['ADE']:.4f} | "
            f"FDE: {metrics['FDE']:.4f} | "
            f"Unc: {metrics['mean_uncertainty']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if metrics["FDE"] < best_fde:
            best_fde = metrics["FDE"]
            torch.save(model.state_dict(), "mc_dropout/models/mc_dropout_best.pt")
            print(f"Saved best model (FDE={best_fde:.4f})")
    else:
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

total_time = time.time() - overall_start
print(f"Training complete in {total_time / 60:.2f} minutes")
print(f"Best validation FDE: {best_fde:.4f}")


"""
Rollback summary:

1. Removed teacher forcing entirely.
2. Restored hidden_dim from 128 back to 64.
3. Restored dropout from 0.2 back to 0.3.
4. Restored learning rate from 5e-4 back to 1e-3.
5. Restored training length from 100 epochs back to 50.
6. Kept the cleaner logging, timing, and fixed random seed.

Why:
The teacher-forcing version performed worse, especially on ADE and FDE.
This rollback returns to the stronger baseline while keeping the code cleaner. 


# Previous Performance:
# Epoch 50 | Loss: 0.2793 | ADE: 4.3025 | FDE: 11.2632 | Unc: 0.18535

# Updated Performance (after fixing data loader / preprocessing):
# Epoch 50/50 | Loss: 0.2843 | ADE: 1.8782 | FDE: 5.1941 | Unc: 0.038590
# Training complete in 4.12 minutes
# Best validation FDE: 4.9965
"""