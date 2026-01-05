from src.training.train_xgboost import train
from src.features.clusters import add_location_clusters
from src.features.time_features import add_time_features
from src.features.distance import add_distance_features
from src.preprocessing.outliers import remove_outliers
from src.preprocessing.clean import clean_basic
from src.preprocessing.load_data import load_train
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

df = load_train()
df = clean_basic(df)
df = remove_outliers(df)
df = add_distance_features(df)
df = add_time_features(df)
df = add_location_clusters(df)

train(df)
