# tests/test_clusters.py
import pandas as pd
from src.features.clusters import add_location_clusters


def test_add_location_clusters():
    # sample data
    df = pd.DataFrame({
        'pickup_latitude': [40.7, 40.8],
        'pickup_longitude': [-73.9, -74.0],
        'dropoff_latitude': [40.6, 40.75],
        'dropoff_longitude': [-73.95, -74.05],
    })

    df_clustered = add_location_clusters(df, n_clusters=2)

    # test columns added
    assert 'pickup_cluster' in df_clustered.columns
    assert 'dropoff_cluster' in df_clustered.columns

    # test cluster values are integers
    assert df_clustered['pickup_cluster'].dtype.kind in 'iu'
    assert df_clustered['dropoff_cluster'].dtype.kind in 'iu'
