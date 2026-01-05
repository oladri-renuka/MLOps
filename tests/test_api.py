# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.api.main import app

client = TestClient(app)

# ------------------------
# Fixture to mock MLflow model
# ------------------------
@pytest.fixture(autouse=True)
def mock_mlflow_model():
    """
    Automatically mocks mlflow.pyfunc.load_model() so tests don't require a real model.
    Returns a dummy model with a predict() method.
    """
    with patch("app.api.ml_model.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [10.5]  # dummy prediction
        mock_load.return_value = mock_model
        yield

# ------------------------
# Test root endpoint
# ------------------------
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

# ------------------------
# Test /predict endpoint
# ------------------------
def test_predict_endpoint():
    sample_payload = [
        {
            "vendor_id": 1,
            "passenger_count": 2,
            "pickup_longitude": -73.985,
            "pickup_latitude": 40.765,
            "dropoff_longitude": -73.981,
            "dropoff_latitude": 40.751,
            "store_and_fwd_flag": 0,
            "haversine_distance": 1.5,
            "manhattan_distance": 2.0,
            "pickup_hour": 10,
            "pickup_day_of_week": 2,
            "pickup_month": 5,
            "pickup_day": 15,
            "is_rush_hour": 1,
            "is_weekend": 0,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "dow_sin": 0.9,
            "dow_cos": 0.4,
            "pickup_cluster": 1,
            "dropoff_cluster": 2
        }
    ]
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
