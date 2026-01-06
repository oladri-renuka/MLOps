import requests

url = "https://taxi-gi7q.onrender.com/predict"

data = [
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

response = requests.post(url, json=data)
print(response.json())
