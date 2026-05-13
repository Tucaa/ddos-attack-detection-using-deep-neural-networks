# Pydantic sheme za request i response objekte
from pydantic import BaseModel, Field, model_validator
from typing import List
from api.config import WINDOW_SIZE, NUM_FEATURES, FEATURE_NAMES, CLASS_LABELS


class TrafficSample(BaseModel):
    """Jedan uzorak mreznog saobracaja — jedan red iz prozora."""

    features: List[float] = Field(
        ...,
        min_length=NUM_FEATURES,
        max_length=NUM_FEATURES,
        description=(
            f"List {NUM_FEATURES} of numeric features: "
            + ", ".join(FEATURE_NAMES)
        ),
    )


class PredictRequest(BaseModel):
    """
    Zahtev za predikciju — klijent salje ceo sliding window.
    window mora sadrzati tacno WINDOW_SIZE uzoraka,
    svaki sa tacno NUM_FEATURES feature-a.
    """

    window: List[TrafficSample] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description=f"Window of {WINDOW_SIZE} traffic samples.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "window": [
                {"features": [64.0, 100.0, 6400.0, 443, 54321, 6, 2, 64, 0.01, 0.2, 1, 1]}
                for _ in range(WINDOW_SIZE)
            ]
        }
    }}


class ClassProbability(BaseModel):
    """Naziv klase i odgovarajuca verovatnoca."""

    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """
    Odgovor na zahtev za predikciju.
    Sadrzi prediktovanu klasu, confidence i sve verovatnoce.
    """

    predicted_class: str = Field(..., description="Class with highest probability.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of predicted class."
    )
    is_attack: bool = Field(
        ..., description="True if predicted clas is not 'normal'."
    )
    probabilities: List[ClassProbability] = Field(
        ..., description="Probability for all classes sorted desc."
    )


class HealthResponse(BaseModel):
    """Odgovor health-check endpointa."""

    status: str
    model_loaded: bool
    window_size: int
    num_features: int
    classes: List[str]


# Novo za ollamu
class SimulateRequest(BaseModel):
    attack_type: str = Field(
        ...,
        description="Tip napada koji Ollama treba da simulira",
        examples=["syn-flood", "udp-flood-large", "dns-amplification"],
    )


class AttackAnalysis(BaseModel):
    description: str = Field(
        ...,
        description="Tehnicki opis detektovanog napada"
    )
    mitigation_steps: list[str] = Field(
        ...,
        description="Konkretni koraci za mitigaciju napada"
    )


class SimulateResponse(BaseModel):
    # Podaci o simulaciji
    requested_attack_type: str
    generated_window_size: int

    # Rezultati LSTM modela
    predicted_class: str
    confidence: float
    is_attack: bool
    class_probabilities: dict[str, float]

    # Ollama analiza
    analysis: AttackAnalysis