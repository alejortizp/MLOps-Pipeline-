import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, OUTPUTS_DIR


def generate_predictions(pipeline, X_blind: pd.DataFrame) -> np.ndarray:
    return pipeline.predict(X_blind)


def save_predictions(predictions: np.ndarray) -> str:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "target_pred.csv"
    pd.DataFrame({"target_pred": predictions}).to_csv(path, index=False)
    print(f"Predictions saved to {path}")
    return str(path)


def save_model_artifact(pipeline, name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"best_model_{name.lower()}.joblib"
    joblib.dump(pipeline, path)
    print(f"Model artifact saved to {path}")
    return str(path)
