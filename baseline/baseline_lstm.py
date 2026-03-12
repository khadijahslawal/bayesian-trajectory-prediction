import torch
import torch.nn as nn
import numpy as np
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import ScenesDataLoader

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.makedirs('baseline/models', exist_ok=True)
os.makedirs('baseline/results', exist_ok=True)


class BaselineLSTM(nn.Module):
    """
    Deterministic LSTM for trajectory prediction.
    No dropout at inference — point estimate only, no uncertainty.
    Used as comparison baseline for MC Dropout and Variational BNN.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 2,
        pred_len: int = 12,
        num_layers: int = 2,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_seq: torch.Tensor):
        """
        Args:
            obs_seq: (B, obs_len, 2)
        Returns:
            pred_seq: (B, pred_len, 2)
        """
        _, (h_n, c_n) = self.encoder(obs_seq)

        h = h_n[-1]   # top layer hidden state (B, hidden_dim)
        c = c_n[-1]

        dec_input = obs_seq[:, -1, :]  # last observed step
        preds = []

        for _ in range(self.pred_len):
            h, c = self.decoder_cell(dec_input, (h, c))
            out = self.output_layer(h)
            preds.append(out)
            dec_input = out  # auto-regressive

        return torch.stack(preds, dim=1)  # (B, pred_len, 2)


def ADE_loss(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        obs, gt = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(obs)
        loss = ADE_loss(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ade_list, fde_list = [], []
    for batch in loader:
        obs, gt = batch[0].to(device), batch[1].to(device)
        pred = model(obs)
        disp = torch.norm(pred - gt, dim=-1)  # (B, pred_len)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())
    return {
        'ADE': np.mean(ade_list),
        'FDE': np.mean(fde_list),
    }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Same training scenes as MC Dropout best model
    loader = ScenesDataLoader(data_root='data/raw/')
    train_loader = loader.get_train_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
    )
    val_loader = loader.get_val_loader(
        scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Val samples:      {len(val_loader.dataset)}")

    model = BaselineLSTM(
        input_dim=2,
        hidden_dim=128,
        pred_len=12,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = []
    best_fde = float('inf')

    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step(train_loss)

        if epoch % 5 == 0:
            metrics = evaluate(model, val_loader, device)
            history.append((epoch, train_loss, metrics['ADE'], metrics['FDE']))

            if metrics['FDE'] < best_fde:
                best_fde = metrics['FDE']
                torch.save(model.state_dict(), 'baseline/models/baseline_best.pt')

    # Print results table
    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'ADE':>8} | {'FDE':>8}")
    print("-" * 40)
    for epoch, loss, ade, fde in history:
        print(f"{epoch:6d} | {loss:8.4f} | {ade:8.4f} | {fde:8.4f}")
    print(f"\nBest FDE: {best_fde:.4f}")

    # Save results
    with open('baseline/results/training_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'ADE', 'FDE'])
        writer.writerows(history)
    print("Results saved to baseline/results/training_results.csv")