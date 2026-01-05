# tests/test_train_xgboost.py
import pandas as pd
from src.training.train_xgboost import train


def test_train_xgboost():
    # sample data
    df = pd.DataFrame({
        'vendor_id': [1, 2, 1, 2],
        'passenger_count': [1, 2, 1, 2],
        'pickup_longitude': [-73.98, -73.95, -73.99, -73.97],
        'pickup_latitude': [40.75, 40.76, 40.77, 40.74],
        'dropoff_longitude': [-73.99, -73.96, -73.98, -73.97],
        'dropoff_latitude': [40.76, 40.77, 40.75, 40.74],
        'store_and_fwd_flag': [0, 1, 0, 1],
        'haversine_distance': [1.2, 0.5, 0.8, 1.0],
        'manhattan_distance': [1.5, 0.6, 0.9, 1.1],
        'pickup_hour': [8, 9, 10, 11],
        'pickup_day_of_week': [1, 2, 3, 4],
        'pickup_month': [1, 1, 1, 1],
        'pickup_day': [5, 5, 5, 5],
        'is_rush_hour': [1, 0, 1, 0],
        'is_weekend': [0, 0, 0, 0],
        'hour_sin': [0.5, 0.6, 0.7, 0.8],
        'hour_cos': [0.87, 0.8, 0.7, 0.6],
        'dow_sin': [0.2, 0.3, 0.4, 0.1],
        'dow_cos': [0.9, 0.8, 0.7, 0.6],
        'pickup_cluster': [0, 1, 0, 1],
        'dropoff_cluster': [1, 0, 1, 0],
        'trip_duration': [600, 1200, 800, 900]
    })

    model = train(df)

    # test model type
    import xgboost as xgb
    assert isinstance(model, xgb.Booster)
