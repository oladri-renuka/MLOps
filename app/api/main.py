from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List
from app.api.schemas import TripData
from app.api.ml_model import predict
import logging
import time
import mlflow
import json
import traceback
import os

# ---------------------------
# Setup logging
# ---------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_requests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="Taxi Trip Duration Predictor")

# ---------------------------
# Middleware for logging & MLflow metrics
# ---------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    try:
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8") if body_bytes else ""

        # Log request
        logger.info(json.dumps({
            "event": "request_received",
            "method": request.method,
            "url": str(request.url),
            "body": body_str
        }))

        # Process request
        response = await call_next(request)

        duration = round(time.time() - start_time, 4)

        # Log response
        logger.info(json.dumps({
            "event": "response_sent",
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_sec": duration
        }))

        # Log metrics in MLflow
        try:
            mlflow.log_metric("api_requests", 1)
            mlflow.log_metric("response_time_sec", duration)
        except Exception as e:
            logger.warning(f"MLflow metric logging failed: {str(e)}")

        return response

    except Exception as e:
        logger.error(f"Unhandled middleware error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )

# ---------------------------
# API endpoints
# ---------------------------


@app.post("/predict")
def predict_trip(trips: List[TripData]):
    try:
        data = [trip.dict() for trip in trips]
        predictions = predict(data)
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Prediction failed. Check logs for details."}
        )


@app.get("/")
def root():
    return {"message": "MLflow + FastAPI Taxi Trip Duration API"}
