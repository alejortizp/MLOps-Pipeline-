"""Modulo de carga y division de datos.

Proporciona funciones para leer los CSVs de entrenamiento y test ciego,
y para dividir los datos en conjuntos de entrenamiento y validacion.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAINING_DATA, BLIND_TEST_DATA, RANDOM_STATE, TEST_SIZE


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV de entrenamiento y separa features (X) del target (y)."""
    df = pd.read_csv(TRAINING_DATA)
    X = df.drop(columns=["target"])  # 20 columnas: feature_0 a feature_19
    y = df["target"]                 # Variable objetivo continua
    return X, y


def load_blind_test_data() -> pd.DataFrame:
    """Carga el CSV de test ciego (200 muestras, solo features, sin target)."""
    return pd.read_csv(BLIND_TEST_DATA)


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento (80%) y validacion (20%).

    Usa semilla fija (RANDOM_STATE) para garantizar reproducibilidad.
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
