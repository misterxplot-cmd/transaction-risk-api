import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from train_demo import build_demo_bundle


@pytest.fixture(scope="module")
def bundle():
    return build_demo_bundle(rows=300)


@pytest.fixture
def client(bundle):
    with TestClient(create_app(bundle=bundle)) as connection:
        yield connection


@pytest.fixture
def transaction():
    return json.loads((Path(__file__).parents[1] / "examples/transaction.json").read_text())


def test_prediction_scores_and_batch_match(client, transaction):
    result = client.post("/predict", json=transaction)
    assert result.status_code == 200
    body = result.json()
    assert body["training_mode"] == "synthetic_demo"
    assert sum(body["class_scores"].values()) == pytest.approx(1.0)
    assert body["label"] == max(body["class_scores"], key=body["class_scores"].get)
    batch = client.post("/predict/batch", json={"transactions": [transaction, transaction]})
    assert batch.status_code == 200
    predictions = batch.json()["predictions"]
    assert len(predictions) == 2
    for prediction in predictions:
        # Parallel tree accumulation can differ at floating-point roundoff.
        assert prediction["class_scores"] == pytest.approx(
            body["class_scores"], rel=1e-12, abs=1e-12
        )
        assert {key: value for key, value in prediction.items() if key != "class_scores"} == {
            key: value for key, value in body.items() if key != "class_scores"
        }


@pytest.mark.parametrize("field,value", [("amount", -1), ("hour_of_day", 24),
                                        ("login_frequency", -1), ("location_region", " ")])
def test_invalid_inputs_rejected(client, transaction, field, value):
    transaction[field] = value
    assert client.post("/predict", json=transaction).status_code == 422


def test_target_proxy_and_empty_batch_rejected(client, transaction):
    transaction["risk_score"] = 99
    assert client.post("/predict", json=transaction).status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    del transaction["risk_score"]
    assert client.post("/predict/batch", json={"transactions": [transaction] * 101}).status_code == 422


def test_missing_model_is_not_silently_replaced(tmp_path, transaction):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as connection:
        assert connection.get("/health").status_code == 200
        assert connection.get("/ready").status_code == 503
        assert connection.post("/predict", json=transaction).status_code == 503


def test_unknown_category_and_documentation(client, transaction):
    transaction["location_region"] = "unseen-region"
    assert client.post("/predict", json=transaction).status_code == 200
    assert client.get("/docs").status_code == 200
    assert "/predict/batch" in client.get("/openapi.json").json()["paths"]
    assert client.get("/ready").status_code == 200
