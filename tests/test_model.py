import pytest
from unittest.mock import patch, MagicMock
from app.api.ml_model import predict

@pytest.fixture
def sample_data_single():
    return [{
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
    }]

@pytest.fixture
def sample_data_multiple():
    return [
        {
            "vendor_id": 1,
            "passenger_count": 1,
            "pickup_longitude": -73.985,
            "pickup_latitude": 40.765,
            "dropoff_longitude": -73.981,
            "dropoff_latitude": 40.751,
            "store_and_fwd_flag": 0,
            "haversine_distance": 1.0,
            "manhattan_distance": 1.5,
            "pickup_hour": 9,
            "pickup_day_of_week": 1,
            "pickup_month": 6,
            "pickup_day": 10,
            "is_rush_hour": 1,
            "is_weekend": 0,
            "hour_sin": 0.4,
            "hour_cos": 0.9,
            "dow_sin": 0.8,
            "dow_cos": 0.6,
            "pickup_cluster": 0,
            "dropoff_cluster": 1
        },
        {
            "vendor_id": 2,
            "passenger_count": 3,
            "pickup_longitude": -73.981,
            "pickup_latitude": 40.751,
            "dropoff_longitude": -73.970,
            "dropoff_latitude": 40.730,
            "store_and_fwd_flag": 1,
            "haversine_distance": 2.0,
            "manhattan_distance": 2.5,
            "pickup_hour": 18,
            "pickup_day_of_week": 5,
            "pickup_month": 7,
            "pickup_day": 20,
            "is_rush_hour": 1,
            "is_weekend": 1,
            "hour_sin": 0.9,
            "hour_cos": -0.4,
            "dow_sin": 0.7,
            "dow_cos": 0.7,
            "pickup_cluster": 2,
            "dropoff_cluster": 3
        }
    ]

# Patch mlflow load_model to return a mock model with a predict method
@pytest.fixture(autouse=True)
def mock_mlflow_model():
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda df: [3.0 for _ in range(len(df))]  # log1p dummy
    with patch("app.api.ml_model.mlflow.pyfunc.load_model", return_value=mock_model):
        # reload module so 'model' gets the mock
        import importlib
        import app.api.ml_model as ml_model
        importlib.reload(ml_model)
        yield

def test_predict_single(sample_data_single):
    from app.api.ml_model import predict
    preds = predict(sample_data_single)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert all(p > 0 for p in preds)

def test_predict_multiple(sample_data_multiple):
    from app.api.ml_model import predict
    preds = predict(sample_data_multiple)
    assert len(preds) == 2
    assert all(p > 0 for p in preds)
