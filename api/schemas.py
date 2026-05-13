# Pydantic seme za request i response objekte

from typing import List
from pydantic import BaseModel, Field
from api.config import WINDOW_SIZE, NUM_FEATURES, FEATURE_NAMES, CLASS_LABELS


# --- /predict (JSON) ---

class TrafficSample(BaseModel):
    """Jedan uzorak mreznog saobracaja — jedan red prozora."""

    features: List[float] = Field(
        ...,
        min_length=NUM_FEATURES,
        max_length=NUM_FEATURES,
        description=f"Lista {NUM_FEATURES} feature-a: " + ", ".join(FEATURE_NAMES),
    )


class PredictRequest(BaseModel):
    """Klijent salje ceo sliding window kao listu uzoraka."""

    window: List[TrafficSample] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "window": [
                    {"features": [64.0, 100.0, 6400.0, 443, 54321, 6, 2, 64, 0.01, 0.2, 1, 1]}
                    for _ in range(WINDOW_SIZE)
                ]
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_class:    str
    confidence:         float = Field(..., ge=0.0, le=1.0)
    is_attack:          bool
    class_probabilities: dict[str, float]


# --- /predict/file (CSV upload) ---

class WindowPrediction(BaseModel):
    """Rezultat predikcije za jedan prozor iz fajla."""

    window_index:    int
    predicted_class: str
    confidence:      float
    is_attack:       bool


class FileInferenceResponse(BaseModel):
    total_windows:  int
    attack_windows: int
    predictions:    list[WindowPrediction]


# --- /health ---

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    window_size:  int
    num_features: int
    classes:      List[str]


# --- /simulate (Ollama) ---

class SimulateRequest(BaseModel):
    attack_type: str = Field(
        ...,
        description="Tip napada koji Ollama treba da simulira",
        examples=["syn-flood", "udp-flood-large", "dns-amplification"],
    )


class AttackAnalysis(BaseModel):
    description:      str
    mitigation_steps: list[str]


class SimulateResponse(BaseModel):
    requested_attack_type: str
    generated_window_size: int
    predicted_class:       str
    confidence:            float
    is_attack:             bool
    class_probabilities:   dict[str, float]
    analysis:              AttackAnalysis