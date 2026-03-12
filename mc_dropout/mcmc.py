import torch 
import torch.nn as nn
import numpy as np
import os
from src.data_loader import ScenesDataLoader
import csv

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.makedirs('mc_dropout/models', exist_ok=True)
os.makedirs('mc_dropout/results', exist_ok=True)   # mcmc.py

class MCDropoutLSTM(nn.Module):
    """
    LSTM trajectory predictor with MC Droupout
    """
    def __init__(self, input_dim: int=2, hidden_dim: int=64, output_dim:int=2, pred_len:int=12,
                 num_layers:int=2, dropout_p:float=0.3):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        #Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
        )

        # Drop out layer
        self.dropout = nn.Dropout(
            p=dropout_p
        )

        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, obs_seq: torch.Tensor):
        """
        Args: obs_seq: (B, obs_len, 2)
        Returns: pred_seq (B, pred_len, 2)
        """
        B = obs_seq.size(0)

        # Encode observation
        _, (h_n, c_n) = self.encoder(obs_seq)
        # h - hidden state also called the output state, represents what the LSTM has decided to output at a specific timestamp n. It flows into the next timestamp and 
        # is what is used for making predictions
        # c - cell state is the LSTM's long term memory. It runs through time with only minor modications via the forget and input gates. It can carry information 
        # across many timesteps without it vanishing

        # Using the top LSTM layer's hidden state for decoding
        h = self.dropout(h_n[-1])
        c = c_n[-1]

        # Using the last observed displacement as the first decoder input
        dec_input = obs_seq[:, -1, :]
         
        preds = []

        for _ in range(self.pred_len):
            h, c = self.decoder_cell(dec_input, (h,c))
            h = self.dropout(h)
            out = self.output_layer(h)
            preds.append(out)
            dec_input = out 
        
        return torch.stack(preds, dim=1) #(B, pred_len, 2)
    
    def enable_droupout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


@torch.no_grad()
def mc_predict(model: MCDropoutLSTM, obs_seq=torch.Tensor, n_samples: int=50):
    model.eval()
    model.enable_droupout()

    sample_preds = torch.stack(
        [model(obs_seq) for _ in range(n_samples)]
    )

    mean = sample_preds.mean(dim=0)
    variance = sample_preds.var(dim=0)

    return mean, variance, sample_preds

def ADE_loss(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean()

criterion = ADE_loss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # print(batch)
        obs, gt = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        pred = model(obs)
        loss = criterion(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss/ len(loader)

def evaluate(model, loader, device, n_samples=50):
    model.eval()
    ade_list, fde_list, unc_list = [], [], []

    for batch in loader:
        obs, gt = batch[0].to(device), batch[1].to(device)

        mean, variance, _ = mc_predict(model, obs, n_samples=n_samples)

        # Both mean and gt are already in the same space (relative to last_obs)
        # No cumsum needed — just compare directly
        disp = torch.norm(mean - gt, dim=-1)  # (B, pred_len)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())
        unc_list.append(variance.mean().item())

    # return is OUTSIDE the loop
    return {
        'ADE': np.mean(ade_list),
        'FDE': np.mean(fde_list),
        'mean_uncertainty': np.mean(unc_list)
    }
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #Data
    loader = ScenesDataLoader(data_root='data/raw/')
    train_loader = loader.get_train_loader(
    # scenes=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'zara3', 'students1', 'students3']
    scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
    )
    val_loader = loader.get_val_loader(scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'])
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = MCDropoutLSTM(input_dim=2,
                           hidden_dim=128, # up from 64
                           pred_len=12,
                           num_layers=2,
                           dropout_p=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # adaptive scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_fde = float('inf')

    # for batch in train_loader:
    #     print(type(batch))
    #     print(len(batch))
    #     print(batch[0].shape)
    #     print(batch[1].shape if len(batch) > 1 else "only one element")
    #     break

    history = []

    for epoch in range(1, 101):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss)

        if epoch % 5 == 0:
            metrics = evaluate(model, val_loader, device, n_samples=50)
            history.append((epoch, train_loss, metrics['ADE'], metrics['FDE'], metrics['mean_uncertainty']))
            print(f"{epoch:6d} | {train_loss:8.4f} | {metrics['ADE']:8.4f} | {metrics['FDE']:8.4f} | {metrics['mean_uncertainty']:10.6f}")
            if metrics['FDE'] < best_fde:
                best_fde = metrics['FDE']
                torch.save(model.state_dict(), 'mc_dropout/models/mc_dropout_best.pt')

    # Print everything at the end
    # print(f"\n{'Epoch':>6} | {'Loss':>8} | {'ADE':>8} | {'FDE':>8} | {'Unc':>10}")
    # print("-" * 55)
    # for epoch, loss, ade, fde, unc in history:
    #     print(f"{epoch:6d} | {loss:8.4f} | {ade:8.4f} | {fde:8.4f} | {unc:10.6f}")
    print(f"\nBest FDE: {best_fde:.4f}")

    # Save training history to CSV
    with open('mc_dropout/results/training_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'ADE', 'FDE', 'uncertainty'])
        writer.writerows(history)

    print("Results saved to mc_dropout/results/training_results.csv")

# Final MC Dropout results are:

# ADE: 0.4187 
# FDE: 0.8892
# Uncertainty: 0.025117
# Epoch 50

# Training samples: 15328
# Val samples: 2720
#      5 |   0.5514 |   0.5059 |   1.0881 |   0.030295
#     10 |   0.5199 |   0.4600 |   1.0055 |   0.028203
#     15 |   0.5016 |   0.4475 |   0.9785 |   0.027917
#     20 |   0.4919 |   0.4402 |   0.9547 |   0.025524
#     25 |   0.4867 |   0.4452 |   0.9638 |   0.023327
#     30 |   0.4792 |   0.4312 |   0.9322 |   0.024623
#     35 |   0.4739 |   0.4350 |   0.9339 |   0.024343
#     40 |   0.4689 |   0.4456 |   0.9471 |   0.023404
#     45 |   0.4631 |   0.4340 |   0.9299 |   0.022692
#     50 |   0.4585 |   0.4187 |   0.8892 |   0.025117
#     55 |   0.4540 |   0.4395 |   0.9247 |   0.024812
#     60 |   0.4492 |   0.4275 |   0.9119 |   0.023329
#     65 |   0.4437 |   0.4378 |   0.9304 |   0.022820
#     70 |   0.4394 |   0.4435 |   0.9448 |   0.021978
#     75 |   0.4364 |   0.4365 |   0.9290 |   0.023704
#     80 |   0.4325 |   0.4240 |   0.9012 |   0.025239
#     85 |   0.4285 |   0.4242 |   0.8985 |   0.023498
#     90 |   0.4227 |   0.4321 |   0.9179 |   0.023355
#     95 |   0.4200 |   0.4209 |   0.8935 |   0.024006
#    100 |   0.4173 |   0.4390 |   0.9274 |   0.024678

# Best FDE: 0.8892