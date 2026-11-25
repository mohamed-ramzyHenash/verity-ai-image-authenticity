from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    label: str            # "Real Images" | "AI-Generated Images"
    is_real: bool
    confidence: float     # probability of the predicted label
    probability_real: float
    probability_ai: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
