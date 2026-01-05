import mlflow
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train(df):
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    features = [c for c in df.columns if c not in [
        'id', 'pickup_datetime', 'dropoff_datetime',
        'trip_duration', 'log_trip_duration'
    ]]
    print("Features used for training:", features)

    X = df[features]
    y = df['log_trip_duration']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'objective': 'reg:squarederror',
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    with mlflow.start_run():
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=50
        )

        preds = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        mlflow.log_metric("rmse_log", rmse)
        mlflow.xgboost.log_model(model, "model")

    return model
