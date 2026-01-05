from sklearn.cluster import KMeans


def add_location_clusters(df, n_clusters=10):
    pickup_km = KMeans(n_clusters=n_clusters, random_state=42)
    dropoff_km = KMeans(n_clusters=n_clusters, random_state=42)

    df['pickup_cluster'] = pickup_km.fit_predict(
        df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = dropoff_km.fit_predict(
        df[['dropoff_latitude', 'dropoff_longitude']])

    return df
