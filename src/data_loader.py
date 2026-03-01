import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAINING_DATA, BLIND_TEST_DATA, RANDOM_STATE, TEST_SIZE


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(TRAINING_DATA)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def load_blind_test_data() -> pd.DataFrame:
    return pd.read_csv(BLIND_TEST_DATA)


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
