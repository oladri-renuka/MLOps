# app/api/ml_model.py
import mlflow
import pandas as pd
import numpy as np

# Model URI
MODEL_URI = "runs:/377fed7b556e427faae3ae0f65fd28f8/model"

# Lazy-loaded model
_model = None

def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model

def predict(data):
    """
    data: list of dicts (each dict matches TripData)
    returns: list of predicted trip durations in seconds
    """
    df = pd.DataFrame(data)
    log_preds = get_model().predict(df)
    # Convert log trip_duration back to seconds and ensure native Python float
    return [float(np.expm1(p)) for p in log_preds]
