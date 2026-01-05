# app/api/ml_model.py
import mlflow
import pandas as pd
import numpy as np

# Load MLflow model
model_uri = "runs:/377fed7b556e427faae3ae0f65fd28f8/model"  # use your Run ID
model = mlflow.pyfunc.load_model(model_uri)


def predict(data):
    """
    data: list of dicts (each dict matches TripData)
    returns: list of predicted trip durations in seconds
    """
    df = pd.DataFrame(data)
    log_preds = model.predict(df)
    # Convert log trip_duration back to seconds and ensure native Python float
    return [float(np.expm1(p)) for p in log_preds]
