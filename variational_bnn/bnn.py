import torch 
import torch.nn as nn 
import pyro 
import pyro.distributions as dist 
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam as PyroAdam 
from src.data_loader import ScenesDataLoader
import os 
import numpy as np
import torch.nn.functional as F
import csv 

 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
pyro.set_rng_seed(SEED)  # only needed in bnn.py
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.makedirs('variational_bnn/models', exist_ok=True)
os.makedirs('variational_bnn/results', exist_ok=True)  # bnn.py


class VariationalLSTM(nn.Module):
    """
    LSTM trajectory predictor with Bayesian linear output layer via Pyro
    Rather than point estimates, the output layer weights are modeled as distributions:

    w ~ Normal(w_mu, softplus(w_rho))

    Training is performed using variational inference (SVI in Pyro),
    which learns the parameters (w_mu, w_rho) of the approximate posterior
    distribution over the weights.

    The encoder and decoder LSTM layers remain deterministic. Making only
    the output layer Bayesian is a common practical approach that captures
    meaningful predictive uncertainty while avoiding the computational
    complexity of fully Bayesian recurrent networks.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128, 
                 output_dim: int = 2, 
                 pred_len: int = 12,
                 num_layers: int = 2,
                 dropout_p: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
        # Encoder - same structure as MC Dropout
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers= num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)

        # Deterministic output layer which pyro wraps this with weight distributions
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_seq: torch.Tensor):
        B = obs_seq.size(0)

        _, (h_n, c_n) = self.encoder(obs_seq)
        h = self.dropout(h_n[-1])
        c = c_n[-1]

        dec_input = obs_seq[:, -1, :]
        preds = []

        for _ in range(self.pred_len):
            h, c = self.decoder_cell(dec_input, (h,c))
            h = self.dropout(h)
            out = self.output_layer(h)
            preds.append(out)
            dec_input = out 
        return torch.stack(preds, dim=1) #(B, pred_len, 2)


#Pyro Model
class BayesianTrajectoryPredictor:
    """
    Wraps Variational LSTM with Pyro model/ guide for ELBO training. 
    model: defines the generative process p(y | x, w) * p(w)
    guide: defines the variational posterior q(w | x, y) -> this is what we learn 
    """

    def __init__(self, lstm:VariationalLSTM, device):
        self.lstm = lstm 
        self.device = device
        self.out_in = lstm.hidden_dim
        self.out_out = 2 #x, y

    def model(self, obs_seq, gt=None):
        """
        Generative model: sample weights from prior, then predict, then observe
        """
        
        # Prior over output layer weights: N(0, 1)
        w_prior = dist.Normal(
            torch.zeros(self.out_out, self.out_in).to(self.device),
            torch.ones(self.out_out, self.out_in).to(self.device)
        )
        b_prior = dist.Normal(
            torch.zeros(self.out_out).to(self.device),
            torch.ones(self.out_out).to(self.device)
        )

        # Sample weights from prior
        w = pyro.sample("output_w", w_prior.to_event(2))
        b = pyro.sample("output_b", b_prior.to_event(1))

        B = obs_seq.size(0)
        _, (h_n, c_n) = self.lstm.encoder(obs_seq)
        h = self.lstm.dropout(h_n[-1])
        c = c_n[-1]
        dec_input = obs_seq[:, -1, :]

        preds = []
        for _ in range(self.lstm.pred_len):
            h, c = self.lstm.decoder_cell(dec_input, (h, c))
            h = self.lstm.dropout(h)
            out = F.linear(h, w, b)   # ← functional, no mutation
            preds.append(out)
            dec_input = out

        pred = torch.stack(preds, dim=1)

        obs_noise = torch.tensor(0.1).to(self.device)
        with pyro.plate("data", B):
            pyro.sample("obs", dist.Normal(pred, obs_noise).to_event(2), obs=gt)

        return pred

    
    def guide(self, obs_seq, gt=None):
        """
        Variational posterior: learn mean and variance for output weights
        """

        w_mu  = pyro.param("w_mu",  torch.zeros(self.out_out, self.out_in).to(self.device))
        w_rho = pyro.param("w_rho", -5 * torch.ones(self.out_out, self.out_in).to(self.device))  # tighter init
        b_mu  = pyro.param("b_mu",  torch.zeros(self.out_out).to(self.device))
        b_rho = pyro.param("b_rho", -5 * torch.ones(self.out_out).to(self.device))

        # Soft plus ensures positive standard deviation
        w_sigma = torch.nn.functional.softplus(w_rho)
        b_sigma = torch.nn.functional.softplus(b_rho)

         # Sample from variational posterior
        pyro.sample("output_w", dist.Normal(w_mu, w_sigma).to_event(2))
        pyro.sample("output_b", dist.Normal(b_mu, b_sigma).to_event(1))


#Trainin

def train_variational(lstm, bayes_predictor, train_loader, val_loader, device, num_epochs=100):
    pyro.clear_param_store()

    svi = SVI(
        model = bayes_predictor.model,
        guide = bayes_predictor.guide,
        optim = PyroAdam({"lr": 1e-3}),
        loss = Trace_ELBO()
    )

    history = []
    best_fde = float('inf')

    for epoch in range(1, num_epochs + 1):
        lstm.train()
        total_loss = 0.0
        for batch in train_loader:
            obs, gt = batch[0].to(device), batch[1].to(device)
            loss = svi.step(obs, gt)
            total_loss += loss 
    
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            metrics = evaluate_variational(lstm, bayes_predictor, val_loader, device)
            history.append((epoch, avg_loss, metrics['ADE'], metrics['FDE'], metrics['mean_uncertainty']))

            print(f"{epoch:6d} | {avg_loss:8.4f} | {metrics['ADE']:8.4f} | {metrics['FDE']:8.4f} | {metrics['mean_uncertainty']:10.6f}")

            if metrics['FDE'] < best_fde:
                best_fde = metrics['FDE']
                torch.save(lstm.state_dict(), 'variational_bnn/models/vbnn_best.pt')
                pyro.get_param_store().save('variational_bnn/models/vbnn_params.pt')
            
    
    print("Done")
    print(f"\n{'Epoch':>6} | {'ELBO Loss':>10} | {'ADE':>8} | {'FDE':>8} | {'Unc':>10}")
    for epoch, loss, ade, fde, unc in history:
        print(f"{epoch:6d} | {loss:10.2f} | {ade:8.4f} | {fde:8.4f} | {unc:10.6f}")
    print(f"\nBest FDE: {best_fde:.4f}")
    return history
                                           



# Inference

@torch.no_grad()
def vbnn_predict(lstm, bayes_predictor, obs_seq, n_samples=50):
    """
    Sample weights from the learned posterior in n_samples times. 
    Each sample gives a different trajectory predicton
    Returns: mean, variance, samples
    """

    lstm.eval()
    preds = []

    for _ in range(n_samples):
        guide_trace = pyro.poutine.trace(bayes_predictor.guide).get_trace(obs_seq)
        w = guide_trace.nodes["output_w"]["value"]
        b = guide_trace.nodes["output_b"]["value"]

        B = obs_seq.size(0)
        _, (h_n, c_n) = lstm.encoder(obs_seq)
        h = lstm.dropout(h_n[-1])
        c = c_n[-1]
        dec_input = obs_seq[:, -1, :]

        step_preds = []
        for _ in range(lstm.pred_len):
            h, c = lstm.decoder_cell(dec_input, (h, c))
            h = lstm.dropout(h)
            out = F.linear(h, w, b)
            step_preds.append(out)
            dec_input = out

        preds.append(torch.stack(step_preds, dim=1))

    samples = torch.stack(preds, dim=0)
    return samples.mean(0), samples.var(0), samples

# Evaluation
def evaluate_variational(lstm, bayes_predictor, loader, device, n_samples=50):
    lstm.eval()
    ade_list, fde_list, unc_list = [], [], []

    for batch in loader:
        obs, gt = batch[0].to(device), batch[1].to(device)
        mean, variance, _ = vbnn_predict(lstm, bayes_predictor, obs, n_samples)

        disp = torch.norm(mean - gt, dim=-1)
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

    loader = ScenesDataLoader(data_root='data/raw/')
    train_loader = loader.get_train_loader(
    scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
    )
    val_loader = loader.get_val_loader(scenes=['eth', 'hotel', 'univ', 'zara1', 'students1'])



    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Val samples:      {len(val_loader.dataset)}")

    lstm = VariationalLSTM(
        input_dim=2,
        hidden_dim=128,
        pred_len=12,
        num_layers=2,
        dropout_p=0.1,
    ).to(device)

    bayes_predictor = BayesianTrajectoryPredictor(lstm, device)
    history = train_variational(
        lstm, bayes_predictor, train_loader, val_loader, device, num_epochs=100
    )

    with open('variational_bnn/results/training_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'ADE', 'FDE', 'uncertainty'])
        writer.writerows(history)

    print("Results saved to variational_bnn/results/training_results.csv")



#  Epoch |  ELBO Loss |      ADE |      FDE |        Unc
#      5 |   19074.13 |   0.6763 |   1.1522 |   0.001549
#     10 |   18361.44 |   0.6394 |   1.1074 |   0.001451
#     15 |   17949.18 |   0.6238 |   1.0707 |   0.001650
#     20 |   17687.06 |   0.6246 |   1.1014 |   0.001861
#     25 |   17608.08 |   0.6065 |   1.0535 |   0.002086
#     30 |   17413.74 |   0.6120 |   1.0873 |   0.002197
#     35 |   17316.50 |   0.6048 |   1.0728 |   0.002430
#     40 |   17246.42 |   0.5994 |   1.0605 |   0.002726
#     45 |   17191.61 |   0.5951 |   1.0550 |   0.002679
#     50 |   17212.83 |   0.5972 |   1.0772 |   0.002949
#     55 |   17131.49 |   0.6017 |   1.0755 |   0.002872
#     60 |   17199.75 |   0.5881 |   1.0438 |   0.003089
#     65 |   17021.03 |   0.5962 |   1.0769 |   0.003084
#     70 |   17080.43 |   0.5892 |   1.0630 |   0.003217
#     75 |   17148.74 |   0.5819 |   1.0315 |   0.003372
#     80 |   17111.80 |   0.5819 |   1.0355 |   0.003297
#     85 |   17091.33 |   0.5831 |   1.0526 |   0.003398
#     90 |   17142.70 |   0.5933 |   1.0745 |   0.003183
#     95 |   17128.57 |   0.5813 |   1.0402 |   0.003321
#    100 |   17100.18 |   0.5916 |   1.0764 |   0.003243