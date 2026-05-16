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



class ClassMetrics(BaseModel):
    """Per-class metrike iz sklearn classification_report."""
 
    precision: float
    recall:    float
    support:   int
    # sklearn koristi "f1-score" sa crticom — alias omogucava oba oblika
    f1_score:  float = Field(..., alias="f1-score")
 
    model_config = {
        # Dozvoljava i "f1_score" i "f1-score" pri deserijalizaciji
        "populate_by_name": True,
    }
 
 
class AnalyzeRequest(BaseModel):
    """
    Ulazni podaci za /analyze endpoint.
    Odgovaraju formatu koji generise metrics_exporter.export_metrics_to_json().
    """
 
    classification_report: dict[str, ClassMetrics] = Field(
        ...,
        description=(
            "Per-class metrike iz sklearn classification_report. "
            "Kljucevi su nazivi klasa, vrednosti su ClassMetrics objekti."
        ),
    )
    confusion_matrix: list[list[int]] = Field(
        ...,
        description="2D matrica konfuzije (num_classes x num_classes).",
    )
    mcc_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Matthews Correlation Coefficient.",
    )
    roc_auc_scores: dict[str, float] = Field(
        ...,
        description="Per-class ROC-AUC skorovi (one-vs-rest metoda).",
    )
 
    model_config = {
        "json_schema_extra": {
            "example": {
                "classification_report": {
                    "normal": {
                        "precision": 0.98,
                        "recall": 0.97,
                        "f1-score": 0.975,
                        "support": 1200,
                    },
                    "syn_flood": {
                        "precision": 0.72,
                        "recall": 0.68,
                        "f1-score": 0.70,
                        "support": 300,
                    },
                },
                "confusion_matrix": [[1164, 36], [96, 204]],
                "mcc_score": 0.812,
                "roc_auc_scores": {"normal": 0.99, "syn_flood": 0.83},
            }
        }
    }
 
 
class AnalysisRecommendations(BaseModel):
    """Preporuke za poboljsanje podeljene po kategorijama."""
 
    data_generation: list[str] = Field(
        ...,
        description="Preporuke za unapredjenje sintetickog generatora podataka.",
    )
    model: list[str] = Field(
        ...,
        description="Preporuke za izmenu arhitekture ili hiperparametara LSTM-a.",
    )
    training: list[str] = Field(
        ...,
        description="Preporuke za izmenu strategije treniranja.",
    )
 
 
class AnalyzeResponse(BaseModel):
    """
    Izlaz /analyze endpointa — rezultat LangGraph analize.
    """
 
    weak_classes:       list[str]
    per_class_analysis: str = Field(
        ...,
        description="Tehnicka analiza zasto odredjene klase imaju loše performanse.",
    )
    confusion_patterns: str = Field(
        ...,
        description="Analiza obrazaca gresaka u matrici konfuzije.",
    )
    recommendations:    AnalysisRecommendations
    summary:            str = Field(
        ...,
        description="Kratak zakljucak sa prioritetnom akcijom za poboljsanje.",
    )