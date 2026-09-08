from pathlib import Path

import joblib
import pandas as pd
import sklearn

from app.schema import CLASSES, FEATURES, Prediction, Transaction


def validate_bundle(bundle: dict) -> dict:
    if not isinstance(bundle, dict) or not {"model", "metadata"} <= bundle.keys():
        raise ValueError("Invalid model bundle")
    metadata = bundle["metadata"]
    if metadata.get("schema_version") != 1 or metadata.get("features") != FEATURES:
        raise ValueError("Model schema does not match the API")
    if metadata.get("training_mode") not in {"synthetic_demo", "metaverse_educational"}:
        raise ValueError("Unsupported training mode")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Train and serve with the same scikit-learn version")
    classes = list(bundle["model"].classes_)
    if classes != metadata.get("classes") or set(classes) != CLASSES:
        raise ValueError("Model class labels are invalid")
    if not metadata.get("dataset_sha256"):
        raise ValueError("Missing model version")
    return bundle


def load_bundle(path: Path) -> dict:
    # Only load artifacts you generated yourself: joblib uses pickle internally.
    return validate_bundle(joblib.load(path))


def predict(bundle: dict, transactions: list[Transaction]) -> list[Prediction]:
    frame = pd.DataFrame([item.model_dump() for item in transactions], columns=FEATURES)
    model, metadata = bundle["model"], bundle["metadata"]
    probabilities = model.predict_proba(frame)
    return [Prediction(
        label=str(model.classes_[row.argmax()]),
        class_scores={str(label): float(score) for label, score in zip(model.classes_, row)},
        training_mode=metadata["training_mode"],
        model_version=metadata["dataset_sha256"][:12],
    ) for row in probabilities]
