"""Run with: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.schema import BatchRequest, BatchResponse, Prediction, Transaction
from app.service import load_bundle, predict, validate_bundle

logger = logging.getLogger(__name__)


def create_app(model_path: Path | None = None, bundle: dict | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.bundle = None
        try:
            application.state.bundle = validate_bundle(bundle) if bundle is not None else load_bundle(
                model_path or Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
            )
        except Exception:
            logger.exception("Model unavailable; readiness and prediction will return 503")
        yield
        application.state.bundle = None

    application = FastAPI(
        title="Transaction Risk API", version="1.0.0", lifespan=lifespan,
        description="Educational inference service. Scores are not real-world fraud probabilities.",
    )

    def require_model(request: Request):
        current = getattr(request.app.state, "bundle", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Train or configure an artifact first.")
        return current

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/ready", responses={503: {"description": "Model unavailable"}})
    def ready(request: Request):
        require_model(request)
        return {"status": "ready"}

    @application.get("/model", responses={503: {"description": "Model unavailable"}})
    def model_info(request: Request):
        return require_model(request)["metadata"]

    @application.post("/predict", response_model=Prediction,
                      responses={503: {"description": "Model unavailable"}})
    def predict_one(transaction: Transaction, request: Request):
        return predict(require_model(request), [transaction])[0]

    @application.post("/predict/batch", response_model=BatchResponse,
                      responses={503: {"description": "Model unavailable"}})
    def predict_batch(payload: BatchRequest, request: Request):
        return BatchResponse(predictions=predict(require_model(request), payload.transactions))

    return application


app = create_app()
