"""Build an explicitly synthetic artifact to run the API without external data."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.schema import FEATURES


def build_demo_bundle(rows=1200, seed=42):
    if rows < 100:
        raise ValueError("At least 100 demo rows are required")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "hour_of_day": rng.integers(0, 24, rows),
        "amount": rng.lognormal(4, 1.1, rows),
        "ip_prefix": rng.integers(1, 224, rows).astype(float),
        "login_frequency": rng.integers(0, 30, rows),
        "session_duration": rng.uniform(1, 120, rows),
        "location_region": rng.choice(["Europe", "Asia", "America"], rows),
        "purchase_pattern": rng.choice(["focused", "random", "high_value"], rows),
        "age_group": rng.choice(["young", "adult", "senior"], rows),
    })
    # Artificial recipe demonstrates the service contract, not real financial risk.
    synthetic_score = (np.log1p(frame.amount) + frame.login_frequency / 8
                       + (frame.purchase_pattern == "random") + rng.normal(0, .6, rows))
    low, high = np.quantile(synthetic_score, [.5, .8])
    labels = np.where(synthetic_score < low, "low_risk",
                      np.where(synthetic_score < high, "moderate_risk", "high_risk"))
    categorical = ["location_region", "purchase_pattern", "age_group"]
    numeric = [name for name in FEATURES if name not in categorical]
    model = Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("classifier", RandomForestClassifier(n_estimators=40, max_depth=8, random_state=seed, n_jobs=2)),
    ])
    model.fit(frame[FEATURES], labels)
    data_hash = hashlib.sha256(frame.to_csv(index=False).encode() + labels.tobytes()).hexdigest()
    return {"model": model, "metadata": {
        "schema_version": 1, "training_mode": "synthetic_demo", "features": FEATURES,
        "classes": list(model.classes_), "sklearn_version": sklearn.__version__,
        "dataset_sha256": data_hash, "seed": seed, "rows": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    bundle = build_demo_bundle(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    print(f"Saved synthetic_demo model with {args.rows} generated rows")
