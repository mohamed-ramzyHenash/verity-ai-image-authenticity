"""Single source of truth for paths and hyperparameters.

Every other module (training, evaluation, API) imports from here.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


# --- Data -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "Real and AI Genersted Images"

# Label mapping (sigmoid output = probability of REAL)
CLASS_NAMES = {
    "FAKE": 0,
    "REAL": 1,
}

LABEL_TO_NAME = {v: k for k, v in CLASS_NAMES.items()}
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


# --- Training ---------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_STATE = 42
EPOCHS = 5
LEARNING_RATE = 1e-3
SHUFFLE_BUFFER = 2000
VAL_FRACTION = 0.15     # share of the held-out part used for validation

# --- Artifacts --------------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_resnet50.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# --- API --------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CORS_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000",
                "http://localhost:5500", "http://127.0.0.1:5500"]
