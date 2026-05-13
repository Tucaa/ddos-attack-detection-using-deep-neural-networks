import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.config import WINDOW_SIZE, NUM_FEATURES, FEATURE_NAMES, CLASS_LABELS
from api.model import model_wrapper
from api.ollama_client import generate_attack_scenario, generate_attack_analysis
from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    TrafficSample,
    SimulateRequest,
    SimulateResponse,
    AttackAnalysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

VALID_ATTACK_TYPES = {
    "udp-flood-large",
    "dns-amplification",
    "subnet-carpet-bombing",
    "syn-flood",
    "icmp-flood",
    "udp-flood-mixed",
    "ntp-amplification",
    "ack-flood",
    "normal",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DDoS Detection API...")
    model_wrapper.load()
    if model_wrapper.is_loaded:
        logger.info("Model ready.")
    else:
        logger.warning(
            "Model NOT loaded — /predict will return 503 until model is available."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="DDoS Detection Inference API",
    description=(
        "Inference API for PyTorch LSTM model for DDoS attack classification. "
        f"Takes a sliding window of {WINDOW_SIZE} samples, "
        f"each with {NUM_FEATURES} features."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Meta endpointi

@app.get("/", tags=["Meta"])
async def root():
    return {
        "message": "DDoS Detection API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_wrapper.is_loaded,
        window_size=WINDOW_SIZE,
        num_features=NUM_FEATURES,
        classes=CLASS_LABELS,
    )


# Inference endpointi
# Vidi da kasnije uradis neku validaciju inputa
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Direktna LSTM predikcija za manualno uneti sliding window.
    """
    if not model_wrapper.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    window_data = [sample.features for sample in request.window]

    try:
        result = model_wrapper.predict(window_data)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    logger.info(
        f"Prediction: {result['predicted_class']} "
        f"(confidence={result['confidence']:.3f}, attack={result['is_attack']})"
    )

    return PredictResponse(**result)


@app.post("/simulate", response_model=SimulateResponse, tags=["Inference"])
async def simulate(request: SimulateRequest):
    """
    Kompletan pipeline: Ollama generise scenario napada →
    LSTM klasifikuje → Ollama analizira i generise mitigaciju.

    - **attack_type**: tip napada koji treba simulirati
    """
    if not model_wrapper.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    if request.attack_type not in VALID_ATTACK_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown type of attack: '{request.attack_type}'. "
                f"Valid types: {sorted(VALID_ATTACK_TYPES)}"
            ),
        )

    # Korak 1: Ollama generise matricu saobracaja 
    logger.info(f"[simulate] Generating scenario for: {request.attack_type}")
    try:
        matrix = await generate_attack_scenario(
            attack_type=request.attack_type,
            window_size=WINDOW_SIZE,
            feature_names=FEATURE_NAMES,
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"[simulate] Error during scenario generation: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama failed to generate the scenario: {str(e)}",
        )

    # Korak 2: LSTM klasifikacija
    logger.info(f"[simulate] Starting LSTM inference...")
    try:
        prediction = model_wrapper.predict(matrix)
    except Exception as e:
        logger.error(f"[simulate] Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    logger.info(
        f"[simulate] Predikcija: {prediction['predicted_class']} "
        f"(confidence={prediction['confidence']:.3f})"
    )

    # Korak 3: Ollama analiza 
    logger.info(f"[simulate] Generating analysis and mitigation...")
    try:
        analysis_data = await generate_attack_analysis(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            is_attack=prediction["is_attack"],
            class_probabilities=prediction["class_probabilities"],
            attack_type_requested=request.attack_type,
        )
    except Exception as e:
        logger.error(f"[simulate] Error during analysis: {e}")
        # Analiza nije kriticna — ne prekidamo ako ne uspe
        analysis_data = {
            "description": "Analysis not available at the moment.",
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