
"""FastAPI application.

Run from backend:
uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore

from app.predictor import predictor
from app.schemas import HealthResponse, PredictionResponse
from config import (
    ALLOWED_CONTENT_TYPES,
    CORS_ORIGINS,
    FRONTEND_DIR,
    MAX_UPLOAD_BYTES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the application starts
    predictor.load()

    if not predictor.loaded:
        log.error("Model could not be loaded. Prediction endpoint will be unavailable.")

    yield

    log.info("Application shutting down.")


app = FastAPI(
    title="AI Face Detector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.loaded,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    # Check model availability
    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train it first.",
        )

    # Validate declared content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG or PNG image.",
        )

    # Read only the allowed size plus one byte
    data = await file.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result = predictor.predict(data)

    except (ValueError, TypeError) as exc:
        log.warning("Invalid image %s: %s", file.filename, exc)

        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        ) from exc

    except Exception:
        log.exception("Unexpected prediction error for %s", file.filename)

        raise HTTPException(
            status_code=500,
            detail="An internal prediction error occurred.",
        )

    finally:
        await file.close()

    log.info(
        "%s -> %s (confidence: %.3f)",
        file.filename,
        result["label"],
        result["confidence"],
    )

    return PredictionResponse(
        filename=file.filename or "upload",
        **result,
    )


# Mount frontend last to avoid intercepting API routes
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
