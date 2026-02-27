import pandas as pd
import numpy as np 
import torch
import os
from torch.utils.data import Dataset, DataLoader


def load_raw_data(filepath):
    """
    Load raw data from .txt file

    Args: filepath e.g 'raw/eth/train/biwi_eth.txt'

    Returns: Dataframe with columns frame, pedestrian_id, x, y
    """

    data = pd.read_csv(filepath, sep='\t', 
                       names=['frame', 'ped_id', 'x', 'y'],
                       dtype={'frame': int, 'ped_id': int, 'x': float, 'y': float}
    )

    return data 


# why are we doing 8 and 12
def extract_trajectories(data, obs_len=8, pred_len=12):
    """
    Extract sequence of 8 observed + 12 predicted timestamp
    """

    observations = []
    predictions = []

    seq_len = obs_len + pred_len

    for ped_id in data['ped_id'].unique():
        # sort frame for each pedestrians
        ped_data = data[data['ped_id'] == ped_id].sort_values('frame')
        positions = ped_data[['x', 'y']].values

        for i in range(len(positions) - seq_len + 1):
            sequence = positions[i: i+seq_len]

            obs = sequence[:obs_len]     #First n observation
            pred = sequence[obs_len:]    #First n predictions

            observations.append(obs)
            predictions.append(pred)
        
    observations  = np.array(observations)
    predictions = np.array(predictions)

    return observations, predictions


train_data = load_raw_data('data/raw/eth/train/biwi_hotel_train.txt')
obs, pred = extract_trajectories(train_data)
# print(f"Extracted {len(obs)} trajectories")
# print(f"Observation shape: {obs.shape}")   # (N, 8, 2)
# print(f"Prediction shape: {pred.shape}")    # (N, 12, 2)
# obs.shape: (32, 8, 2)
#            ↑   ↑  ↑
#            |   |  └─ 2D coordinates (x, y)
#            |   └──── 8 timesteps
#            └──────── 32 trajectories in batch

# Observation (8, 2):          Prediction (12, 2):
# [[x₁, y₁],                   [[x₉,  y₉],
#  [x₂, y₂],                    [x₁₀, y₁₀],
#  [x₃, y₃],                    [x₁₁, y₁₁],
#  [x₄, y₄],                    [x₁₂, y₁₂],
#  [x₅, y₅],                    [x₁₃, y₁₃],
#  [x₆, y₆],                    [x₁₄, y₁₄],
#  [x₇, y₇],                    [x₁₅, y₁₅],
#  [x₈, y₈]]                    [x₁₆, y₁₆],
#                               [x₁₇, y₁₇],
#                               [x₁₈, y₁₈],
#                               [x₁₉, y₁₉],
#                               [x₂₀, y₂₀]]
# ↑ Past (what we observe)     ↑ Future (what we predict)

def normalize_trajectories(observations, predictions):
    """
    Normalize to relative coordinates (relative to last observation)
    This makes the model's job easier!
    
    Args:
        observations: (N, 8, 2)
        predictions: (N, 12, 2)
    
    Returns:
        obs_normalized: (N, 8, 2) - relative to first position
        pred_normalized: (N, 12, 2) - relative to last observation
        stats: dict with normalization parameters (for denormalizing later)
    """
    N = observations.shape[0]
    
    # Make relative to last observation position
    first_obs = observations[:, 0:1, :]  # (N, 1, 2)
    last_obs = observations[:, -1:, :]  # (N, 1, 2) - last observed position

    # Normalize observations (relative to first position)
    obs_normalized = observations - first_obs
    # Normalize predictions (relative to last observation)
    pred_normalized = predictions - last_obs
    
    stats = {
        'first_obs': first_obs,
        'last_obs': last_obs
    }
    
    return obs_normalized, pred_normalized, stats

obs_norm, pred_norm, stats = normalize_trajectories(obs, pred)
# print(f"Normalized observation range: [{obs_norm.min():.2f}, {obs_norm.max():.2f}]")


class TrajectoryDataset(Dataset):
    def __init__(self, observations, predictions):
        self.obs = torch.FloatTensor(observations)
        self.pred = torch.FloatTensor(predictions)

    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.pred[idx]

class ScenesDataLoader:
    
    SCENE_FILES = {
        'eth': 'biwi_eth',
        'hotel': 'biwi_hotel',
        'univ': 'uni_examples',
        'zara1': 'crowds_zara01',
        'zara2': 'crowds_zara02',
        # Bonus datasets
        'zara3': 'crowds_zara03',
        'students1': 'students001',
        'students3': 'students003'
    }
    def __init__(self, data_root='data/raw/'):
        self.data_root = data_root
        self.consolidated_path = os.path.join(data_root, 'raw')


    def get_train_loader(self, scenes=['eth', 'hotel', 'univ', 'zara1'], batch_size=32, shuffle=True):
        """
        Get training data loader 

        Args:
            scene: 'eth', 'hotel', 'univ', 'zara1', or 'zara2'
            batch_size: Batch size
            shuffle: Whether to shuffle
        
        Returns:
            DataLoader yielding batches of (obs, pred)
        """
        all_obs = []
        all_pred = []
        for scene in scenes:
            file_base = self.SCENE_FILES[scene]
            filepath = os.path.join(
                self.consolidated_path,
                'train',
                f'{file_base}_train.txt'
            )

            data = load_raw_data(filepath)
            obs, pred = extract_trajectories(data)
            all_obs.append(obs)
            all_pred.append(pred)
        
        combined_obs = np.concatenate(all_obs, axis=0)
        combined_pred = np.concatenate(all_pred, axis=0)

        # Normalize
        obs_norm, pred_norm, _ = normalize_trajectories(
            combined_obs, combined_pred
        )

        dataset  = TrajectoryDataset(obs_norm, pred_norm)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_val_loader(self, scenes=['eth', 'hotel', 'univ', 'zara1'], batch_size=32):
        """
        Load validation data
        """
        all_obs = []
        all_pred = []

        for scene in scenes:
            file_base = self.SCENE_FILES[scene]
            filepath = os.path.join(
                self.consolidated_path,
                'val',
                f'{file_base}_val.txt'
            )

            data = load_raw_data(filepath)
            obs, pred = extract_trajectories(data)
            all_obs.append(obs)
            all_pred.append(pred)

        combined_obs = np.concatenate(all_obs, axis=0)
        combined_pred = np.concatenate(all_pred, axis=0)

        # Normalize
        obs_norm, pred_norm, _ = normalize_trajectories(
            combined_obs, combined_pred
        )

        dataset  = TrajectoryDataset(obs_norm, pred_norm)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def get_test_loader(self, scene = 'zara2', batch_size=32):
        """
        Load test data
        """
        file_base = self.SCENE_FILES[scene]
        filepath = os.path.join(
                self.consolidated_path,
                'train',
                f'{file_base}_val.txt'
        )

        data = load_raw_data(filepath)
        obs, pred = extract_trajectories(data)
        obs_norm, pred_norm, _ = normalize_trajectories(
            obs, pred
        )

        dataset  = TrajectoryDataset(obs_norm, pred_norm)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# dataset = TrajectoryDataset(obs_norm, pred_norm)
# print(f"Dataset size: {len(dataset)}")