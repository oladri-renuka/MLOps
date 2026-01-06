# NYC Taxi Trip Duration Prediction (MLOps Project)

This project builds and deploys a **machine learning system** to predict **NYC taxi trip duration** using structured trip data.
It demonstrates an **end-to-end MLOps workflow** including data preprocessing, feature engineering, MLflow experiment tracking, model artifact management, and cloud deployment with a REST API.

---

## Problem Statement

Given trip metadata such as pickup/dropoff locations, time information, passenger count, and engineered features, predict the **trip duration** of NYC taxi rides.

The model predicts **log(trip_duration)** and converts it back to seconds during inference.

---

## Model Overview

* **Algorithm**: XGBoost Regressor
* **Target Variable**: `trip_duration`
* **Metric**: RMSE (log-scale)
* **Experiment Tracking**: MLflow

---

## 🗂 Project Structure

```
MLOps/
├── app/
│   ├── api/
│   │   ├── main.py              # FastAPI app
│   │   ├── ml_model.py          # MLflow model loader + predictor
│   │   └── schemas.py           # Input data schema
│   ├── model/
│   │   ├── MLmodel
│   │   ├── model.ubj
│   │   ├── conda.yaml
│   │   └── requirements.txt
│                
├── render.py # Client script to test API
├── pipelines/
│   └── train_pipeline.py        # End-to-end training pipeline
│
├── src/
│   ├── features/
│   │   ├── distance.py
│   │   ├── time_features.py
│   │   └── clusters.py
│   ├── preprocessing/
│   │   ├── clean.py
│   │   ├── outliers.py
│   │   └── load_data.py
│   └── training/
│       └── train_xgboost.py
│
├── data/
│   └── raw/
│       └── train.csv
│       └── test.csv
│
├── requirements.txt
└── README.md
```

---

## Feature Engineering

The following features are used during both training and inference:

### Raw Features

* `vendor_id`
* `passenger_count`
* `pickup_longitude`, `pickup_latitude`
* `dropoff_longitude`, `dropoff_latitude`
* `store_and_fwd_flag`

### Distance Features

* `haversine_distance`
* `manhattan_distance`

### Time Features

* `pickup_hour`
* `pickup_day_of_week`
* `pickup_month`
* `pickup_day`
* `is_rush_hour`
* `is_weekend`

### Cyclical Encoding

* `hour_sin`, `hour_cos`
* `dow_sin`, `dow_cos`

### Location Clusters

* `pickup_cluster`
* `dropoff_cluster`

Feature parity is maintained between training and inference.

---

## Training Pipeline

Run the training pipeline using:

```bash
python -m pipelines.train_pipeline
```

### What happens:

1. Data is loaded and cleaned
2. Outliers are removed
3. Feature engineering is applied
4. Model is trained using XGBoost
5. Metrics and model artifacts are logged to MLflow

---

## MLflow Tracking

MLflow tracks:

* Training runs
* RMSE metric
* Model artifacts

Example logged artifacts:

```
MLmodel
model.ubj
conda.yaml
requirements.txt
```

Model is exported locally for deployment.

---

## Deployment

The model is deployed as a **FastAPI REST API** and hosted on **Render**.

### Live Endpoint

```
POST https://taxi-gi7q.onrender.com/predict
```

---

## API Usage

### Request (JSON)

```json
[
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
```

---

### Response (JSON)

```json
{
  "predictions": [780.52]
}
```

*Prediction is returned in seconds.*

---

## Test Coverage Report

Automated tests were executed using **pytest** with **coverage analysis** to ensure code reliability and correctness across the feature engineering, preprocessing, and training modules.

### Coverage Summary

```
Name                             Stmts   Miss  Cover
----------------------------------------------------
src/features/__init__.py             0      0   100%
src/features/clusters.py             7      0   100%
src/features/distance.py            12      0   100%
src/features/time_features.py       14      0   100%
src/preprocessing/__init__.py        0      0   100%
src/preprocessing/clean.py           7      0   100%
src/preprocessing/load_data.py       7      1    86%
src/preprocessing/outliers.py        4      0   100%
src/training/__init__.py             0      0   100%
src/training/train_xgboost.py       22      0   100%
src/utils/__init__.py                0      0   100%
----------------------------------------------------
TOTAL                               73      1    99%
```

### Key Highlights

* **Overall Coverage**: **99%**
* All **core ML logic**, feature engineering, and training code are fully covered.
* Ensures **training–inference reliability** and **safe refactoring**.

---

## Testing the API

Use the provided client script:

```bash
python app/render.py
```

---

## Tech Stack

* Python
* XGBoost
* MLflow
* Pandas / NumPy
* FastAPI
* Render (Cloud deployment)

---

## Key MLOps Practices Demonstrated

* Modular feature engineering
* MLflow experiment tracking
* Model artifact management
* Training–inference consistency
* REST-based model serving
* Cloud deployment
