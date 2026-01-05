def remove_outliers(df):
    df = df[(df['trip_duration'] >= 60) & (df['trip_duration'] <= 10800)]

    df = df[
        df['pickup_longitude'].between(-74.05, -73.75) &
        df['dropoff_longitude'].between(-74.05, -73.75) &
        df['pickup_latitude'].between(40.63, 40.85) &
        df['dropoff_latitude'].between(40.63, 40.85)
    ]

    return df
