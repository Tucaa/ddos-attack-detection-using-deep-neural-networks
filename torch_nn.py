import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


SEQUENCE_LEN = 50
BATCH_SIZE = 50
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 2 
LEARING_RATE = 1e-3
EPOCHS = 20


EXCLUDED_COLS = ['label', 'window_id', 'timestamp', 'ts_formated', 'attack_active', 'instance_id', 'vector_id']

LABELS = [
    'normal',
    'udp_flood_large',
    'dns_amplification', 
    'subnet_carpet_bombing', 
    'syn_flood',
    'icmp_flood', 
    'udp_flood_mixed',
    'ntp_amplification',
    'ack_flood'
]


class DDoSDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.Y = torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def prepare_data(path: str, seq_len: int = SEQUENCE_LEN):

    try:
    
        df = pd.read_csv(path)

        # Sortiranje po timestampu
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        features = [col for col in df.columns if col not in EXCLUDED_COLS]
        X_raw = df[features].values.astype(np.float32)

        # Enkodiranje labela
        label_enc = LabelEncoder()
        label_enc.classes_ = np.array(LABELS)
        Y_raw = label_enc.transform(df['label'].values)

        # Normalizacija
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # Sekvence sa kliznim prozorima
        sequneces, labels = [], []

        for i in range(len(X_scaled)-seq_len):
            sequneces.append(X_scaled[i: i + seq_len])
            labels.append(Y_raw[i + seq_len - 1])

        sequneces = np.array(sequneces)
        labels = np.array(labels)

        return sequneces, labels, scaler, label_enc, features
    
    except Exception as e:
        print(f'Exception torch_nn | prepare_data: {e} Line: {sys.exc_info()[2].tb_lineno}')



if __name__ == "__main__":
    path = input('Insert csf file path: ').strip()
    data = prepare_data(path)
    print(data)
