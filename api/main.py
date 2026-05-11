import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import WINDOW_SIZE, NUM_FEATURES, FEATURE_NAMES, CLASS_LABELS
from model import model_wrapper
from schemas import PredictRequest, PredictResponse, HealthResponse

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


#Ucitavanje modela pri pokretanju
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


# App instanca 
app = FastAPI(
    title="DDoS Detection Inference API",
    description=(
        "Inference API for PyTorch LSTM model for DDoS attack classification. "
        f"Takes a sliding window of {WINDOW_SIZE} samples, "
        f"each with {NUM_FEATURES} features."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — slobodan pristup za lokalni razvoj (kasnije ogranici!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trebaces da namestis api za komunikaciju sa ollama modelom
# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """
    Health-check endpoint.
    Vraca status aplikacije i informacije o modelu.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_wrapper.is_loaded,
        window_size=WINDOW_SIZE,
        num_features=NUM_FEATURES,
        classes=CLASS_LABELS,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Predikcija klase saobracaja za dati sliding window.

    - **window**: lista od `WINDOW_SIZE` uzoraka
    - Svaki uzorak sadrzi `NUM_FEATURES` float vrednosti
    - Redosled feature-a: definisan u `config.py → FEATURE_NAMES`

    Vraca prediktovanu klasu, confidence score i verovatnoce svih klasa.
    """
    if not model_wrapper.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    # Izvlacimo feature matrice iz Pydantic objekata
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


@app.get("/", tags=["Meta"])
async def root():
    return {
        "message": "DDoS Detection API is running.",
        "docs": "/docs",
        "health": "/health",
    }