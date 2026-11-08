"""Simple FastAPI backend for the RUL models.

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive page where you can try every endpoint)
Needs: models/<FD>_best.joblib (from run_experiments.py). The /demo endpoint also needs the NASA files in data/.
Folders can be changed with the environment variables MODELS_DIR and DATA_DIR.
"""
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from src import service
from src.service import REQUIRED, ServiceError

# One operating cycle of one engine: cycle number, 3 operating settings, 21 sensors (same as the NASA files).
Reading = create_model("Reading", cycle=(int, ...), **{c: (float, ...) for c in REQUIRED if c != "cycle"})


class PredictRequest(BaseModel):
    dataset: Literal["FD001", "FD002", "FD003", "FD004"] = Field(..., description="Which trained model to use.")
    readings: List[Reading] = Field(..., description="History of ONE engine, one row per operating cycle "
                                                     "(ideally at least the model's window, see /models).")
    critical_threshold: int = Field(20, description="Predicted RUL below this => CRITICAL.")
    warning_threshold: int = Field(50, description="Predicted RUL below this => WARNING.")
    include_history: bool = Field(False, description="Also return the prediction for every cycle.")


app = FastAPI(title="Turbofan RUL API", version="1.0",
              description="Predicts remaining useful life (cycles left before failure) for turbofan engines.")


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@app.get("/health")
def health():
    """Is the service up, and which datasets have a trained model?"""
    return {"status": "ok", "trained_datasets": service.trained_datasets()}


@app.get("/models")
def models():
    """Details of every trained model: members, window length, RUL cap, sensors used."""
    return [service.model_info(fd) for fd in service.trained_datasets()]


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict remaining useful life for one engine from its sensor history."""
    readings = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in req.readings]
    return _run(service.predict_engine, req.dataset, readings, req.critical_threshold,
                req.warning_threshold, req.include_history)


@app.get("/demo/{dataset}/{unit}")
def demo(dataset: str, unit: int, critical_threshold: int = 20, warning_threshold: int = 50,
         include_history: bool = False):
    """Predict for one engine of the official NASA test set (also returns the true RUL), for quick demos."""
    return _run(service.demo_engine, dataset, unit, critical_threshold, warning_threshold, include_history)
