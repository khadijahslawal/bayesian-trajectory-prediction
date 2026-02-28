import torch 
import torch.nn as nn
import numpy as np
import os
from src.data_loader import ScenesDataLoader

os.makedirs('mc_dropout/models', exist_ok=True)

class MCDroupoutLSTM(nn.Module):
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
def mc_predict(model: MCDroupoutLSTM, obs_seq=torch.Tensor, n_samples: int=50):
    model.eval()
    model.enable_droupout()

    sample_preds = torch.stack(
        [model(obs_seq) for _ in range(n_samples)]
    )

    mean = sample_preds.mean(dim=0)
    variance = sample_preds.var(dim=0)

    return mean, variance, sample_preds

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        print(batch)
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
    ade_list, fde_list, unc_list = [], [], []

    for batch in loader:
        obs, gt = batch[0].to(device), batch[1].to(device)
        # obs = batch['obs'].to(device)
        # gt = batch['pred'].to(device)

        mean, variance, _ = mc_predict(model, obs, n_samples=n_samples)
        mean_abs = mean.cumsum(dim=1)
        gt_abs = gt.cumsum(dim=1)

        disp = torch.norm(mean_abs - gt_abs, dim=-1)
        ade_list.append(disp.mean().item())
        fde_list.append(disp[:, -1].mean().item())
        unc_list.append(variance.mean().item())

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
    scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
    )
    val_loader = loader.get_val_loader(scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'])
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = MCDroupoutLSTM(input_dim=2,
                           hidden_dim=64,
                           pred_len=12,
                           num_layers=2,
                           dropout_p=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_fde = float('inf')

    # for batch in train_loader:
    #     print(type(batch))
    #     print(len(batch))
    #     print(batch[0].shape)
    #     print(batch[1].shape if len(batch) > 1 else "only one element")
    #     break

    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if epoch % 5 == 0:
            metrics = evaluate(model, val_loader, device, n_samples=50)
            print(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"ADE: {metrics['ADE']:.4f} | FDE: {metrics['FDE']:.4f} | "
                f"Unc: {metrics['mean_uncertainty']:.6f}"
            )

            if metrics['FDE'] < best_fde:
                best_fde = metrics['FDE']
                torch.save(model.state_dict(), 'mc_dropout/models/mc_dropout_best.pt')
                print(f"Saved best model (FDE={best_fde:.4f})")


# Current Performance: Epoch  50 | Loss: 0.2793 | ADE: 4.3025 | FDE: 11.2632 | Unc: 0.18535