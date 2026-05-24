import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
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
from functions import *
from metrics_exporter import export_metrics_to_json


SEQUENCE_LEN  = 50
BATCH_SIZE    = 64
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
LEARNING_RATE = 1e-3
EPOCHS        = 20
N_SPLITS      = 5   # Broj foldova za TimeSeriesSplit
MODEL_NAME = 'ddos_lstm_attention.pt'



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
    

# Attenttion mehanizam
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Nauceni vektor paznje koji ocenjuje svaki timestep
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        lstm_out: (batch, seq_len, hidden_size)
        Vraca:
            context:  (batch, hidden_size) — tezinski zbir svih timestepova
            weights:  (batch, seq_len)     — tezine paznje po timestepu
        """
        # Skorovi paznje za svaki timestep
        scores  = self.attention(lstm_out).squeeze(-1)        # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                    # (batch, seq_len)

        # Tezinski zbir hidden stateova
        context = torch.bmm(
            weights.unsqueeze(1), lstm_out                    # (batch, 1, seq_len) x (batch, seq_len, hidden)
        ).squeeze(1)                                          # (batch, hidden_size)

        return context, weights
    

# def prepare_data(csv_path: str, seq_len: int = SEQUENCE_LEN):

#     try:
    
#         df = pd.read_csv(csv_path)

#         # Sortiranje po timestampu
#         if "timestamp" in df.columns:
#             df = df.sort_values("timestamp").reset_index(drop=True)

#         features = [col for col in df.columns if col not in EXCLUDED_COLS]
#         X_raw = df[features].values.astype(np.float32)

#         # Enkodiranje labela
#         label_enc = LabelEncoder()
#         label_enc.classes_ = np.array(LABELS)
#         Y_raw = label_enc.transform(df['label'].values)

#         # Normalizacija
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_raw)

#         # Sekvence sa kliznim prozorima
#         sequneces, labels = [], []

#         for i in range(len(X_scaled)-seq_len):
#             sequneces.append(X_scaled[i: i + seq_len])
#             labels.append(Y_raw[i + seq_len - 1])

#         sequneces = np.array(sequneces)
#         labels = np.array(labels)

#         return sequneces, labels, scaler, label_enc, features
    
#     except Exception as e:
#         print(f'Exception torch_nn | prepare_data: {e} Line: {sys.exc_info()[2].tb_lineno}')



def prepare_data(csv_path: str, seq_len: int = SEQUENCE_LEN):
    """
    Ucitava CSV, sortira po timestamp-u, skalira featuere i pravi sekvence.
    Vraca sekvence, labele, scaler, label encoder i listu feature kolona.
    """
    try:
        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
        X_raw = df[feature_cols].values.astype(np.float32)

        label_enc = LabelEncoder()
        label_enc.classes_ = np.array(LABELS)
        Y_raw = label_enc.transform(df["label"].values)

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        sequences, labels = [], []
        for i in range(len(X_scaled) - seq_len):
            sequences.append(X_scaled[i: i + seq_len])
            labels.append(Y_raw[i + seq_len - 1])

        return (
            np.array(sequences),
            np.array(labels),
            scaler,
            label_enc,
            feature_cols,
        )

    except Exception as e:
        print(f"Exception | prepare_data: {e} Line: {sys.exc_info()[2].tb_lineno}")



def make_dataloaders(csv_path: str, seq_len: int = SEQUENCE_LEN, batch_size: int = BATCH_SIZE):
    try:
        sequences, labels, scaler, le, feature_cols = prepare_data(csv_path, seq_len)

        n = len(sequences)
        train_end = int(n * 0.60)
        val_end   = int(n * 0.80)  # 60% + 20%

        X_train, y_train = sequences[:train_end],       labels[:train_end]
        X_val,   y_val   = sequences[train_end:val_end], labels[train_end:val_end]
        X_test,  y_test  = sequences[val_end:],          labels[val_end:]

        print(f"Train:      {len(X_train):>7} samples")
        print(f"Validation: {len(X_val):>7} samples")
        print(f"Test:       {len(X_test):>7} samples")

        train_ds = DDoSDataset(X_train, y_train)
        val_ds   = DDoSDataset(X_val,   y_val)
        test_ds  = DDoSDataset(X_test,  y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, scaler, le, len(feature_cols)

    except Exception as e:
        print(f'Exception torch_nn | make_dataloaders: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Obican LSTM model
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


# LSTM model sa atention mehanizmom
class DDoSLSTMAttention(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int,
        num_classes: int,
        dropout:     float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AttentionLayer(hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        Vraca logits i attention weights (korisno za vizualizaciju).
        """
        try:
            lstm_out, _ = self.lstm(x)                  # (batch, seq_len, hidden)
            context, weights = self.attention(lstm_out) # (batch, hidden), (batch, seq_len)
            logits = self.classifier(context)           # (batch, num_classes)
            return logits, weights

        except Exception as e:
            print(f"Exception | DDoSLSTMAttention.forward: {e} "
                  f"Line: {sys.exc_info()[2].tb_lineno}")


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
        print(classification_report(all_targets, all_preds, labels=list(range(len(label_names))), target_names=label_names, digits=4))

        # Matthews Correlation Coefficient
        # Dobra metrika za neuravnotežene klase, -1 najgore, +1 najbolje
        mcc = matthews_corrcoef(all_targets, all_preds)
        # mcc = matthews_corrcoef(all_targets, all_preds, len(label_names))
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
        plt.savefig("/graphs/confusion_matrix.png", dpi=150)
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
        plt.savefig("graphs/roc_curves.png", dpi=150)
        plt.show()
        print("Saved ROC curve roc_curves.png")

    except Exception as e:
        print(f'Exception torch_nn | plot_roc_curves: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Nova implementacija sa oversamplingom
def train(csv_path: str, save_path: str = "ddos_lstm.pt"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")

        # Učitavamo raw dataset za računanje weighta pre nego što pravimo sekvence
        raw_dataset = pd.read_csv(csv_path).to_dict("records")
        class_weights = get_class_weights_tensor(raw_dataset, LABELS, device)
        print(f"\nClass weights tensor: {class_weights}")

        train_loader, val_loader, test_loader, scaler, le, num_features = make_dataloaders(csv_path)
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
            val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:>3}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state":   model.state_dict(),
                    "scaler":        scaler,
                    "label_encoder": le,
                    "num_features":  num_features,
                    "hyperparams": {
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers":  NUM_LAYERS,
                        "dropout":     DROPOUT,
                        "seq_len":     SEQUENCE_LEN,
                    },
                }, save_path)
                print(f"Saved new model: {save_path}")

        # Finalna evaluacija
        # Kasnije uradi kros validaciju
        print("\n __Evaluation on validation set__")
        evaluate_full(model, val_loader, criterion, device, LABELS)

        # Test set koristimo samo jednom, na samom kraju 
        print("\n __Final evauluation on test dataset__")
        evaluate_full(model, test_loader, criterion, device, LABELS)

        print("\n Finished training!")

    except Exception as e:
        print(f'Exception torch_nn | train: {e} Line: {sys.exc_info()[2].tb_lineno}')

# Trening jednog folda (sa kros validacijom)
def train_fold(model, train_loader, val_loader,criterion, optimizer, scheduler,device, fold: int,) -> dict:
    """
    Trenira model za jedan fold i vraca metriku najboljeg epocha.
    """
    best_val_loss = float("inf")
    best_metrics  = {}

    for epoch in range(1, EPOCHS + 1):
        # Trening
        model.train()
        train_loss, train_correct = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        # Validacija
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss    += loss.item() * len(y_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                # Verovatnoca po klasi — potrebna za ROC-AUC i metrics_exporter
                all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        t_loss  = train_loss / n_train
        t_acc   = train_correct / n_train
        v_loss  = val_loss / n_val
        v_acc   = val_correct / n_val

        scheduler.step(v_loss)

        print(
            f"  Fold {fold} | Epoch {epoch:>2}/{EPOCHS} | "
            f"Train loss: {t_loss:.4f}  acc: {t_acc:.3f} | "
            f"Val loss: {v_loss:.4f}  acc: {v_acc:.3f}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_metrics  = {
                "fold":      fold,
                "val_loss":  v_loss,
                "val_acc":   v_acc,
                "mcc":       matthews_corrcoef(all_targets, all_preds),
                "preds":     all_preds,
                "targets":   all_targets,
                "probs":     np.array(all_probs),   # Potrebno za ROC-AUC i export
            }

    return best_metrics


def cross_validate(csv_path: str, save_path: str = MODEL_NAME):
    """
    TimeSeriesSplit cross-validacija sa LSTM + Attention modelom.
    Svaki fold cuva hronoloski redosled - nema data leakage-a.
    Na kraju ispisuje prosecnu metriku kroz sve foldove.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {device}")

        sequences, labels, scaler, le, feature_cols = prepare_data(csv_path)
        num_features = len(feature_cols)
        num_classes  = len(le.classes_)

        dataset = DDoSDataset(sequences, labels)
        tss     = TimeSeriesSplit(n_splits=N_SPLITS)

        fold_metrics = []
        best_overall_loss = float("inf")
        best_model_state  = None

        for fold, (train_idx, val_idx) in enumerate(tss.split(sequences), start=1):
            print(f"\n{'='*60}")
            print(f"  Fold {fold}/{N_SPLITS} | "
                  f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")
            print(f"{'='*60}")

            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=BATCH_SIZE, shuffle=False,  # shuffle=False cuva temporalni redosled
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=BATCH_SIZE, shuffle=False,
            )

            # Novi model za svaki fold
            model = DDoSLSTMAttention(
                input_size=num_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                num_classes=num_classes,
                dropout=DROPOUT,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
            criterion = nn.CrossEntropyLoss()

            metrics = train_fold(
                model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                device, fold,
            )
            fold_metrics.append(metrics)

            # Cuvamo globalno najbolji model
            if metrics["val_loss"] < best_overall_loss:
                best_overall_loss = metrics["val_loss"]
                best_model_state  = model.state_dict()
                torch.save({
                    "model_state":   best_model_state,
                    "scaler":        scaler,
                    "label_encoder": le,
                    "num_features":  num_features,
                    "architecture":  "LSTM+Attention",
                    "hyperparams": {
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers":  NUM_LAYERS,
                        "dropout":     DROPOUT,
                        "seq_len":     SEQUENCE_LEN,
                    },
                }, save_path)
                print(f"  New best model saved (fold {fold}) -> {save_path}")

        # Sumarni rezultati
        print(f"\n{'='*60}")
        print("  Cross-validation summary")
        print(f"{'='*60}")
        print(f"  {'Fold':<8} {'Val Loss':<12} {'Val Acc':<12} {'MCC'}")
        print(f"  {'-'*48}")

        for m in fold_metrics:
            print(f"  {m['fold']:<8} {m['val_loss']:<12.4f} "
                  f"{m['val_acc']:<12.4f} {m['mcc']:.4f}")

        avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
        avg_acc  = np.mean([m["val_acc"]  for m in fold_metrics])
        avg_mcc  = np.mean([m["mcc"]      for m in fold_metrics])
        std_acc  = np.std( [m["val_acc"]  for m in fold_metrics])

        print(f"  {'-'*48}")
        print(f"  {'Avg':<8} {avg_loss:<12.4f} {avg_acc:<12.4f} {avg_mcc:.4f}")
        print(f"  {'Std':<8} {'':12} {std_acc:<12.4f}")
        print(f"{'='*60}\n")

        # Classification report najboljeg folda
        best_fold = min(fold_metrics, key=lambda m: m["val_loss"])
        print(f"Classification report (best fold {best_fold['fold']}):")
        print(classification_report(
            best_fold["targets"], best_fold["preds"],
            labels=list(range(num_classes)),
            target_names=LABELS,
            digits=4,
            zero_division=0,
        ))

        # --- Kompletna evaluacija najboljeg folda ---
        # Koristimo preds/targets/probs sacuvane iz train_fold za best epoch

        best_targets = np.array(best_fold["targets"])
        best_preds   = np.array(best_fold["preds"])
        best_probs   = best_fold["probs"]   # shape: (n_val_samples, num_classes)

        # MCC za best fold
        best_mcc = best_fold["mcc"]
        print(f"Matthews Correlation Coefficient (best fold {best_fold['fold']}): {best_mcc:.4f}\n")

        # Confusion matrix — apsolutna i normalizovana
        plot_confusion_matrix(best_targets, best_preds, LABELS)

        # ROC krive po klasi (one-vs-rest)
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(best_targets, classes=list(range(num_classes)))
        plot_roc_curves(y_bin, best_probs, LABELS)

        # --- Export metrika u JSON za LangGraph analizu ---
        print("Exporting evaluation metrics to JSON...")
        json_path = export_metrics_to_json(
            y_true=best_targets,
            y_pred=best_preds,
            y_proba=best_probs,
            class_labels=LABELS,
            output_dir="results/",
            filename="eval_metrics.json",
        )
        print(f"Metrics saved -> {json_path}")
        print("Run: GET /analyze/from-file?path=results/eval_metrics.json\n")

        print("Training finished!")
        return fold_metrics

    except Exception as e:
        print(f"Exception | cross_validate: {e} Line: {sys.exc_info()[2].tb_lineno}")

# Glavna funkcija za trening modela , stara implementacija!
# def train_old(csv_path: str, save_path: str = "ddos_lstm.pt"):
#     try:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using: {device}")

#         train_loader, val_loader, test_loader, scaler, le, num_features = make_dataloaders(csv_path)
#         num_classes = len(le.classes_)

#         model = DDoSLSTM(
#             input_size=num_features,
#             hidden_size=HIDDEN_SIZE,
#             num_layers=NUM_LAYERS,
#             num_classes=num_classes,
#             dropout=DROPOUT,
#         ).to(device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
#         criterion = nn.CrossEntropyLoss()

#         best_val_loss = float("inf")

#         for epoch in range(1, EPOCHS + 1):
#             train_loss, train_acc = singular_epoch(model, train_loader, optimizer, criterion, device)
#             val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
#             scheduler.step(val_loss)

#             print(
#                 f"Epoch {epoch:>3}/{EPOCHS} | "
#                 f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
#                 f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
#             )

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save({
#                     "model_state":   model.state_dict(),
#                     "scaler":        scaler,
#                     "label_encoder": le,
#                     "num_features":  num_features,
#                     "hyperparams": {
#                         "hidden_size": HIDDEN_SIZE,
#                         "num_layers":  NUM_LAYERS,
#                         "dropout":     DROPOUT,
#                         "seq_len":     SEQUENCE_LEN,
#                     },
#                 }, save_path)
#                 print(f"Saved new model: {save_path}")

#         # Finalna evaluacija
#         # Kasnije uradi kros validaciju
#         print("\n __Evaluation on validation set__")
#         evaluate_full(model, val_loader, criterion, device, LABELS)

#         # Test set koristimo samo jednom, na samom kraju 
#         print("\n __Final evauluation on test dataset__")
#         evaluate_full(model, test_loader, criterion, device, LABELS)

#         print("\n Finished training!")

#     except Exception as e:
#         print(f'Exception torch_nn | train: {e} Line: {sys.exc_info()[2].tb_lineno}')




if __name__ == "__main__":
    import sys
    # Morao sam ovo da dodam zbog pozivanja iz subprocesa
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input('Insert csv file path: ').strip()

    save_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    cross_validate(path, save_path)

    # loaded = torch.load('ddos_lstm.pt')
    # data = prepare_data(path)
    # dataloaders = make_dataloaders(path)
    # print(data)
    # print(dataloaders)