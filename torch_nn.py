import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


SEQUENCE_LEN = 50
BATCH_SIZE = 50
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 0.3 
LEARNING_RATE = 1e-3
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
        self.Y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def prepare_data(csv_path: str, seq_len: int = SEQUENCE_LEN):

    try:
    
        df = pd.read_csv(csv_path)

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



def make_dataloaders(csv_path: str, seq_len: int = SEQUENCE_LEN, batch_size: int = BATCH_SIZE):
    try:
        sequences, labels, scaler, le, feature_cols = prepare_data(csv_path, seq_len)

        X_train, X_val, y_train, y_val = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_ds = DDoSDataset(X_train, y_train)
        val_ds   = DDoSDataset(X_val,   y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, scaler, le, len(feature_cols)
    
    except Exception as e:
        print(f'Exception torch_nn | make_data: {e} Line: {sys.exc_info()[2].tb_lineno}')


class DDoSLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,num_classes: int, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Uzimamo samo poslednji timestep
            last_hidden = lstm_out[:, -1, :]
            return self.classifier(last_hidden)
        except Exception as e:
                print(f'Exception torch_nn | DDosLSTM.forward: {e} Line: {sys.exc_info()[2].tb_lineno}')


def singular_epoch(model, loader, optimizer, criterion, device):
    try:
        model.train()
        total_loss, correct = 0.0, 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping - sprečava exploding gradients u LSTM-u
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n
    except Exception as e:
        print(f'Exception torch_nn | singular_epoch: {e} Line: {sys.exc_info()[2].tb_lineno}')


# Obicna metrika
# Kasnije dodaj slozeniju funkciju za metriku, (sa vizualizacijom)
# Vidi da li ces sve ovo raditi u ovoj funkciji ili ces kasnije namestiti nove
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    try:
        model.eval()
        total_loss, correct = 0.0, 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n
    except Exception as e:
        print(f'Exception torch_nn | evaluate: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Kompletna metrika
@torch.no_grad()
def evaluate_full(model, loader, criterion, device, label_names: list[str]):
# def evaluate(model, loader, criterion, device):
    """"
    Complete metrics:
    - Loss i accuracy
    - Confusion matrix
    - Classification report (precision, recall, F1)
    - ROC-AUC (one-vs-rest)
    - Matthews Correlation Coefficient
    """
    try:
        model.eval()

        total_loss = 0.0
        all_preds   = []
        all_targets = []
        all_probs   = []  # Za ROC-AUC

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds   = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)

        avg_loss = total_loss / len(loader.dataset)
        accuracy = (all_preds == all_targets).mean()

        print("=" * 65)
        print(f"  Loss: {avg_loss:.4f}   Accuracy: {accuracy:.4f}")
        print("=" * 65)

        # Classification report 
        print("\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=label_names, digits=4))

        # Matthews Correlation Coefficient
        # Dobra metrika za neuravnotežene klase, -1 najgore, +1 najbolje
        mcc = matthews_corrcoef(all_targets, all_preds, len(label_names))
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")

        # ROC-AUC (one-vs-rest)
        y_bin = label_binarize(all_targets, classes=list(range(len(label_names))))
        try:
            roc_auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
            print(f"ROC-AUC (macro, one-vs-rest): {roc_auc:.4f}\n")
        except ValueError as e:
            print(f"ROC-AUC nije mogao biti izračunat: {e}\n")

        # Plotovi 
        plot_confusion_matrix(all_targets, all_preds, label_names)
        plot_roc_curves(y_bin, all_probs, label_names)

        return {
            "loss":     avg_loss,
            "accuracy": accuracy,
            "mcc":      mcc,
            "preds":    all_preds,
            "targets":  all_targets,
            "probs":    all_probs,
        }

    except Exception as e:
        print(f"Exception | evaluate_full: {e} Line: {sys.exc_info()[2].tb_lineno}")
        
# Plotovanje confusion matrice
def plot_confusion_matrix(y_true, y_pred, label_names: list[str]):
    try:
        cm = confusion_matrix(y_true, y_pred)

        # Normalizovana verzija (procenat po redu)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ["Confusion Matrix (absolute numbers)", "Confusion Matrix (normalized)"],
            ["d", ".2f"],
        ):
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_title(title, fontsize=13, pad=12)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual",    fontsize=11)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.show()
        print("Saved confusion matrix confusion_matrix.png")
    except Exception as e:
        print(f'Exception torch_nn | plot_confusion_matrix: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Plotovanje roc krive po svakoj klasi
def plot_roc_curves(y_bin, all_probs, label_names: list[str]):
    try:
        
        n_classes = len(label_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        plt.figure(figsize=(10, 7))

        for i, (name, color) in enumerate(zip(label_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            auc = roc_auc_score(y_bin[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, color=color, lw=1.8, label=f"{name}  (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate",  fontsize=12)
        plt.title("ROC curves by class (one-vs-rest)", fontsize=13)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig("roc_curves.png", dpi=150)
        plt.show()
        print("Saved ROC curve roc_curves.png")

    except Exception as e:
        print(f'Exception torch_nn | plot_roc_curves: {e} Line: {sys.exc_info()[2].tb_lineno}')


# Glavna funkcija za trening modela
def train(csv_path: str, save_path: str = "ddos_lstm.pt"):
    try:
        # Kasnije doradi ovaj deo , videces da li ces preko dokera ili drugacisje
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using : {device}")

        train_loader, val_loader, scaler, le, num_features = make_dataloaders(csv_path)
        num_classes = len(le.classes_)

        model = DDoSLSTM(
            input_size=num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            dropout=DROPOUT,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = singular_epoch(model, train_loader, optimizer, criterion, device)
            # val_loss,   val_acc   = evaluate(model, val_loader, criterion, device, LABELS)
            val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:>3}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
            )

            # Čuvamo samo najbolji model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "scaler":      scaler,
                    "label_encoder": le,
                    "num_features":  num_features,
                    "hyperparams": {
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers":  NUM_LAYERS,
                        "dropout":     DROPOUT,
                        "seq_len":     SEQUENCE_LEN,
                    },
                }, save_path)
                print(f"Saved new model at the path: {save_path}")

        print("Finished training")

        print("\n____ Final evaluation____\n")
        evaluate_full(model, val_loader, criterion, device, LABELS)
    
    except Exception as e:
        print(f'Exception torch_nn | train: {e} Line: {sys.exc_info()[2].tb_lineno}')




if __name__ == "__main__":
    path = input('Insert csf file path: ').strip()
    train(path)
    # data = prepare_data(path)
    # dataloaders = make_dataloaders(path)
    # print(data)
    # print(dataloaders)

