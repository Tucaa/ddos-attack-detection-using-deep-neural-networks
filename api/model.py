# Ucitavanje modela i inference logika
# Odvojeno od API sloja radi cistog koda i lakseg testiranja

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional
from config import (
    MODEL_PATH,
    NUM_FEATURES,
    NUM_CLASSES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    CLASS_LABELS,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


# --- Definicija LSTM arhitekture ---
# Mora biti identicna onoj koriscenog pri treniranju

class LSTMClassifier(nn.Module):
    """
    Dvoslojna LSTM mreza za klasifikaciju DDoS saobracaja.
    Ulaz: (batch, seq_len, num_features)
    Izlaz: (batch, num_classes) — logiti
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Uzimamo samo poslednji vremenski korak
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)


# --- Singleton wrapper za model ---

class ModelWrapper:
    """
    Singleton koji drzi ucitani model i obavlja inference.
    Ucitava se jednom pri pokretanju aplikacije.
    """

    def __init__(self):
        self.model: Optional[LSTMClassifier] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, path: str = MODEL_PATH) -> bool:
        """
        Ucitava tezine modela sa diska.
        Vraca True ako je uspesno, False ako nije.
        """
        try:
            self.model = LSTMClassifier(
                input_size=NUM_FEATURES,
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=LSTM_NUM_LAYERS,
                num_classes=NUM_CLASSES,
                dropout=LSTM_DROPOUT,
            )
            # Ucitavamo samo state_dict, ne ceo model objekat
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from '{path}' on {self.device}.")
            return True
        except FileNotFoundError:
            logger.warning(f"Model file '{path}' not found. Running without model.")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, window: list[list[float]]) -> dict:
        """
        Prima window kao listu lista float vrednosti.
        window shape: (WINDOW_SIZE, NUM_FEATURES)
        Vraca recnik sa predicted_class, confidence, is_attack, probabilities.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        # Konvertujemo u tensor: (1, WINDOW_SIZE, NUM_FEATURES)
        arr = np.array(window, dtype=np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)                    # (1, num_classes)
            probs = torch.softmax(logits, dim=-1)          # (1, num_classes)
            probs_np = probs.squeeze(0).cpu().numpy()      # (num_classes,)

        predicted_idx = int(np.argmax(probs_np))
        predicted_label = CLASS_LABELS[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        # Sortiramo verovatnoce opadajuce za response
        sorted_indices = np.argsort(probs_np)[::-1]
        probabilities = [
            {"label": CLASS_LABELS[i], "probability": float(probs_np[i])}
            for i in sorted_indices
        ]

        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "is_attack": predicted_label != "normal",
            "probabilities": probabilities,
        }


# Globalna instanca — deli se kroz celu aplikaciju
model_wrapper = ModelWrapper()