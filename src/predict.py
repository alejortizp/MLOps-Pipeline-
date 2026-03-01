"""Modulo de prediccion y guardado de artefactos.

Genera predicciones usando el Pipeline entrenado (que escala internamente),
guarda las predicciones como CSV y serializa el Pipeline con joblib.
"""

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR


def generate_predictions(pipeline, X_blind: pd.DataFrame) -> np.ndarray:
    """Genera predicciones sobre datos crudos usando el Pipeline.

    El Pipeline aplica StandardScaler internamente antes de predecir,
    por lo que X_blind no necesita ser escalado manualmente.

    Args:
        pipeline: Pipeline entrenado (scaler + modelo).
        X_blind: Features del test ciego (sin escalar).

    Returns:
        Array con las predicciones del target.
    """
    return pipeline.predict(X_blind)


def save_predictions(predictions: np.ndarray) -> str:
    """Guarda las predicciones en un CSV con una sola columna 'target_pred'.

    Args:
        predictions: Array de predicciones.

    Returns:
        Ruta del archivo CSV guardado.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "target_pred.csv"
    pd.DataFrame({"target_pred": predictions}).to_csv(path, index=False)
    print(f"Predictions saved to {path}")
    return str(path)


def save_model_artifact(pipeline, name: str) -> str:
    """Serializa el Pipeline completo con joblib.

    El archivo .joblib contiene el Pipeline entero (scaler + modelo),
    permitiendo predecir directamente sobre datos crudos al cargarlo.

    Args:
        pipeline: Pipeline entrenado a serializar.
        name: Nombre del modelo (se usa en el nombre del archivo).

    Returns:
        Ruta del archivo .joblib guardado.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"best_model_{name.lower()}.joblib"
    joblib.dump(pipeline, path)
    print(f"Model artifact saved to {path}")
    return str(path)
