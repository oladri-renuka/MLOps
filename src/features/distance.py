import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_distance_features(df):
    df['haversine_distance'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    df['manhattan_distance'] = (
        abs(df['pickup_latitude'] - df['dropoff_latitude']) +
        abs(df['pickup_longitude'] - df['dropoff_longitude'])
    ) * 111

    return df
