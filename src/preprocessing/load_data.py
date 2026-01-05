from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data/raw"


def load_train():
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test():
    return pd.read_csv(DATA_DIR / "test.csv")
