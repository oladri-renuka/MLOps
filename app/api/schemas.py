# app/api/schemas.py
from pydantic import BaseModel
from typing import List


class TripData(BaseModel):
    vendor_id: int
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    haversine_distance: float
    manhattan_distance: float
    pickup_hour: int
    pickup_day_of_week: int
    pickup_month: int
    pickup_day: int
    is_rush_hour: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    pickup_cluster: int
    dropoff_cluster: int
