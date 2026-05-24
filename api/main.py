import io
import logging
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from api.config import WINDOW_SIZE, NUM_FEATURES, FEATURE_NAMES, CLASS_LABELS
from api.model import model_wrapper
from api.ollama_client import generate_attack_scenario, generate_attack_analysis, generate_attack_descriptions
from api.schemas import (
    PredictRequest, PredictResponse,
    FileInferenceResponse, WindowPrediction,
    HealthResponse,
    SimulateRequest, SimulateResponse, AttackAnalysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

VALID_ATTACK_TYPES = {
    "udp_flood_large", "dns_amplification", "subnet_carpet_bombing",
    "syn_flood", "icmp_flood", "udp_flood_mixed",
    "ntp_amplification", "ack_flood", "normal",
}

# Kes za opis napada
_attack_descriptions_cache: dict | None = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DDoS Detection API...")
    model_wrapper.load()
    if model_wrapper.is_loaded:
        logger.info("Model ready.")
    else:
        logger.warning("Model NOT loaded — /predict endpoints will return 503.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DDoS Detection Inference API",
    description=(
        "Inference API za PyTorch LSTM+Attention model. "
        f"Sliding window: {WINDOW_SIZE} uzoraka x {NUM_FEATURES} feature-a."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Meta ---

@app.get("/", tags=["Meta"])
async def root():
    return {"message": "DDoS Detection API is running.", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_wrapper.is_loaded,
        window_size=WINDOW_SIZE,
        num_features=NUM_FEATURES,
        classes=CLASS_LABELS,
    )


# --- Inference ---

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Predikcija za jedan rucno konstruisan sliding window (JSON)."""
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        result = model_wrapper.predict_single(
            [sample.features for sample in request.window]
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    logger.info(
        f"Prediction: {result['predicted_class']} "
        f"(confidence={result['confidence']:.3f}, attack={result['is_attack']})"
    )
    return PredictResponse(**result)


@app.post("/predict/file", response_model=FileInferenceResponse, tags=["Inference"])
async def predict_file(file: UploadFile = File(...)):
    """
    Batch predikcija iz CSV fajla.
    CSV mora imati iste feature kolone kao trening podaci, bez kolone 'label'.
    """
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        results = model_wrapper.predict_from_df(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    predictions  = [WindowPrediction(window_index=i, **r) for i, r in enumerate(results)]
    attack_count = sum(1 for r in results if r["is_attack"])

    logger.info(
        f"File inference: {len(results)} windows, "
        f"{attack_count} attacks ({file.filename})"
    )

    return FileInferenceResponse(
        total_windows=len(results),
        attack_windows=attack_count,
        predictions=predictions,
    )


# --- Simulate (Ollama) ---

@app.post("/simulate", response_model=SimulateResponse, tags=["Inference"])
async def simulate(request: SimulateRequest):
    """
    Kompletan pipeline: Ollama generise scenario => LSTM klasifikuje =>
    Ollama analizira i generise mitigaciju.
    """
    if not model_wrapper.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if request.attack_type not in VALID_ATTACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid attack type: '{request.attack_type}'. "
                f"Valid attack types: {sorted(VALID_ATTACK_TYPES)}"
            ),
        )

    # Korak 1: Ollama generise matricu saobracaja
    logger.info(f"[simulate] Generating scenario for: {request.attack_type}")
    try:
        matrix = await generate_attack_scenario(attack_type=request.attack_type,window_size=WINDOW_SIZE, feature_names=FEATURE_NAMES)
    except (ValueError, RuntimeError) as e:
        logger.error(f"[simulate] Scenario generation error: {e}")
        raise HTTPException(status_code=502, detail=f"Ollama scenario error: {e}")

    # Korak 2: LSTM klasifikacija
    logger.info("[simulate] Running LSTM inference...")
    try:
        prediction = model_wrapper.predict_single(matrix)
    except Exception as e:
        logger.error(f"[simulate] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    logger.info(
        f"[simulate] Prediction: {prediction['predicted_class']} "
        f"(confidence={prediction['confidence']:.3f})"
    )

    # Korak 3: Ollama analiza
    logger.info("[simulate] Generating analysis...")
    try:
        analysis_data = await generate_attack_analysis(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            is_attack=prediction["is_attack"],
            class_probabilities=prediction["class_probabilities"],
            attack_type_requested=request.attack_type,
        )
    except Exception as e:
        logger.error(f"[simulate] Analysis error: {e}")
        analysis_data = {
            "description": "Analysis not available.",
            "mitigation_steps": [],
        }

    return SimulateResponse(
        requested_attack_type=request.attack_type,
        generated_window_size=len(matrix),
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        is_attack=prediction["is_attack"],
        class_probabilities=prediction["class_probabilities"],
        analysis=AttackAnalysis(**analysis_data),
    )


@app.get("/attacks/info", tags=["Meta"])
async def attacks_info():
    """
    Vraća tehničke opise svih tipova napada sa kojima je model treniran.
    Opisi se generišu kroz Ollamu pri prvom pozivu i keširaju za naredne.
    """
    global _attack_descriptions_cache

    if _attack_descriptions_cache is not None:
        return _attack_descriptions_cache

    # Pronađi i učitaj attacks.py
    attacks_path = Path("attacks.py")
    if not attacks_path.exists():
        # Pokušaj jedan nivo gore (u slučaju da se API pokreće iz api/ foldera)
        attacks_path = Path("../attacks.py")

    if not attacks_path.exists():
        raise HTTPException(
            status_code=404,
            detail="attacks.py not found. Make sure it exists in the project root."
        )

    attacks_source = attacks_path.read_text(encoding="utf-8")

    try:
        descriptions = await generate_attack_descriptions(attacks_source)
    except (ValueError, RuntimeError) as e:
        logger.error(f"[attacks/info] Ollama error: {e}")
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

    _attack_descriptions_cache = {
        "total": len(descriptions),
        "attacks": descriptions,
    }

    logger.info(f"[attacks/info] Generated descriptions for {len(descriptions)} attack types.")
    return _attack_descriptions_cache