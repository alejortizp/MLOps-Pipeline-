import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, OUTPUTS_DIR


def generate_predictions(
    estimator, scaler: StandardScaler, X_blind: pd.DataFrame
) -> np.ndarray:
    X_scaled = scaler.transform(X_blind)
    return estimator.predict(X_scaled)


def save_predictions(predictions: np.ndarray) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "predictions.csv"
    pd.DataFrame({"target_pred": predictions}).to_csv(path, index=False)
    print(f"Predictions saved to {path}")
    return str(path)


def save_model_artifact(estimator, scaler: StandardScaler, name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"estimator": estimator, "scaler": scaler}
    path = MODELS_DIR / f"best_model_{name.lower()}.joblib"
    joblib.dump(artifact, path)
    print(f"Model artifact saved to {path}")
    return str(path)
