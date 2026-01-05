import numpy as np


def add_time_features(df):
    dt = df['pickup_datetime']
    df['pickup_hour'] = dt.dt.hour
    df['pickup_day_of_week'] = dt.dt.dayofweek
    df['pickup_month'] = dt.dt.month
    df['pickup_day'] = dt.dt.day

    df['is_rush_hour'] = df['pickup_hour'].isin(
        [8, 9, 10, 16, 17, 18, 19, 20]).astype(int)
    df['is_weekend'] = (df['pickup_day_of_week'] >= 5).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)

    df['dow_sin'] = np.sin(2 * np.pi * df['pickup_day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['pickup_day_of_week'] / 7)

    return df
