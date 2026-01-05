# tests/test_feature_pipeline.py
import pytest
import sys
from pathlib import Path

# Allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.load_data import load_train
from src.preprocessing.clean import clean_basic
from src.preprocessing.outliers import remove_outliers
from src.features.distance import add_distance_features
from src.features.time_features import add_time_features


def test_feature_pipeline():
    # 1. Load sample data
    df = load_train()
    assert df.shape[0] > 0, "No rows loaded from train.csv"

    # 2. Basic cleaning
    df = clean_basic(df)
    assert df.isnull().sum().sum() == 0, "Nulls present after cleaning"

    # 3. Remove outliers
    df = remove_outliers(df)
    assert df.shape[0] > 0, "All rows removed by outlier filtering"

    # 4. Add distance features
    df = add_distance_features(df)
    expected_distance_cols = ['haversine_distance', 'manhattan_distance']
    for col in expected_distance_cols:
        assert col in df.columns, f"{col} missing after distance feature addition"

    # 5. Add time features
    df = add_time_features(df)
    expected_time_cols = [
        'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'pickup_day',
        'is_rush_hour', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    for col in expected_time_cols:
        assert col in df.columns, f"{col} missing after time feature addition"

    # 6. Success
    print("Feature pipeline test passed")
