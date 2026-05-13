# Ucitavanje modela i inference logika

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from api.config import MODEL_PATH, CLASS_LABELS

logger = logging.getLogger(__name__)

# Kolone koje se iskljucuju pri citanju CSV-a (iste kao pri treniranju)
EXCLUDED_COLS = {
    "label", "window_id", "timestamp", "ts_formated",
    "attack_active", "instance_id", "vector_id",
}


# --- Arhitektura modela (mora biti identicna treniranoj) ---

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq_len, hidden_size)
        scores  = self.attention(lstm_out).squeeze(-1)       # (batch, seq_len)
        weights = F.softmax(scores, dim=1)                   # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


class DDoSLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
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

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_size)
        lstm_out, _      = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return self.classifier(context), weights


# --- Singleton wrapper ---

class ModelWrapper:
    """
    Singleton koji drzi ucitan model, scaler i label encoder.
    Sve sto je potrebno za inference ucitava se iz checkpointa.
    """

    def __init__(self):
        self.model:         Optional[DDoSLSTMAttention] = None
        self.scaler         = None
        self.label_encoder  = None
        self.seq_len:   int = 50
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, path: str = MODEL_PATH) -> bool:
        """
        Ucitava checkpoint sa diska.
        Ocekuje format koji cuva cross_validate():
            model_state, scaler, label_encoder, num_features, hyperparams
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            hp           = checkpoint["hyperparams"]
            num_features = checkpoint["num_features"]
            num_classes  = len(checkpoint["label_encoder"].classes_)

            self.model = DDoSLSTMAttention(
                input_size=num_features,
                hidden_size=hp["hidden_size"],
                num_layers=hp["num_layers"],
                num_classes=num_classes,
                dropout=hp["dropout"],
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

            self.scaler        = checkpoint["scaler"]
            self.label_encoder = checkpoint["label_encoder"]
            self.seq_len       = hp["seq_len"]

            logger.info(f"Model loaded from '{path}' on {self.device}.")
            return True

        except FileNotFoundError:
            logger.warning(f"Model file '{path}' not found.")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _windows_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Uzima DataFrame, iskljucuje ne-feature kolone, skalira sa sacuvanim
        scalerom i pravi sliding window sekvence.
        Vraca ndarray oblika (num_windows, seq_len, num_features).
        """
        feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
        X_raw    = df[feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        n = len(X_scaled)
        if n < self.seq_len:
            raise ValueError(
                f"CSV ima {n} redova, potrebno je najmanje {self.seq_len}."
            )

        return np.stack([
            X_scaled[i: i + self.seq_len]
            for i in range(n - self.seq_len + 1)
        ])

    @torch.no_grad()
    def predict_single(self, window: list[list[float]]) -> dict:
        """
        Predikcija za jedan prozor — koristi /predict JSON endpoint.
        Ocekuje window oblika (seq_len, num_features).
        """
        tensor = torch.from_numpy(
            np.array(window, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        logits, _ = self.model(tensor)
        probs     = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_idx   = int(np.argmax(probs))
        predicted_label = CLASS_LABELS[predicted_idx]

        return {
            "predicted_class":    predicted_label,
            "confidence":         float(probs[predicted_idx]),
            "is_attack":          predicted_label != "normal",
            "class_probabilities": {
                CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))
            },
        }

    @torch.no_grad()
    def predict_from_df(self, df: pd.DataFrame, batch_size: int = 256) -> list[dict]:
        """
        Batch predikcija iz DataFrame-a — koristi /predict/file endpoint.
        Interno pravi sliding windows i prolazi kroz model u batchevima.
        """
        windows = self._windows_from_df(df)
        loader  = DataLoader(
            TensorDataset(torch.from_numpy(windows).to(self.device)),
            batch_size=batch_size,
            shuffle=False,
        )

        results = []
        for (batch,) in loader:
            logits, _ = self.model(batch)
            probs_arr = torch.softmax(logits, dim=-1).cpu().numpy()
            for probs in probs_arr:
                idx   = int(np.argmax(probs))
                label = CLASS_LABELS[idx]
                results.append({
                    "predicted_class": label,
                    "confidence":      float(probs[idx]),
                    "is_attack":       label != "normal",
                })

        return results


# Globalna instanca — deli se kroz celu aplikaciju
model_wrapper = ModelWrapper()