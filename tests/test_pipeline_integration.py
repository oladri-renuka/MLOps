# tests/test_pipeline_integration.py
import pytest
import pandas as pd
from src.preprocessing.load_data import load_train
from src.preprocessing.clean import clean_basic
from src.preprocessing.outliers import remove_outliers
from src.features.distance import add_distance_features
from src.features.time_features import add_time_features


def test_full_pipeline():
    df = load_train()
    df = clean_basic(df)
    df = remove_outliers(df)
    df = add_distance_features(df)
    df = add_time_features(df)

    # Check that output has expected columns
    expected_cols = ['pickup_latitude', 'pickup_longitude',
                     'dropoff_latitude', 'dropoff_longitude',
                     'haversine_distance', 'manhattan_distance',
                     'pickup_hour', 'pickup_day_of_week',
                     'pickup_month', 'pickup_day', 'is_rush_hour',
                     'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin',
                     'dow_cos']
    for col in expected_cols:
        assert col in df.columns

    # Check that no nulls exist in new features
    new_features = ['haversine_distance', 'manhattan_distance',
                    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    assert df[new_features].isnull().sum().sum() == 0

    # Check dataframe has rows left
    assert len(df) > 0
