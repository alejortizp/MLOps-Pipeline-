import time

import mlflow
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.config import (
    CV_FOLDS,
    CV_SCORING,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_ITER,
    RANDOM_STATE,
)
from src.models import get_model_registry


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_single_model(
    name: str,
    estimator,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=name):
        start = time.time()

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=N_ITER,
            cv=CV_FOLDS,
            scoring=CV_SCORING,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X, y)
        elapsed = time.time() - start

        best = search.best_estimator_

        # Additional CV scores with the best estimator
        mae_scores = cross_val_score(
            best, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(
            best, X, y, cv=CV_FOLDS, scoring="r2"
        )
        rmse_scores = cross_val_score(
            best, X, y, cv=CV_FOLDS, scoring="neg_root_mean_squared_error"
        )

        metrics = {
            "cv_mae_mean": -mae_scores.mean(),
            "cv_mae_std": mae_scores.std(),
            "cv_r2_mean": r2_scores.mean(),
            "cv_r2_std": r2_scores.std(),
            "cv_rmse_mean": -rmse_scores.mean(),
            "cv_rmse_std": rmse_scores.std(),
            "training_time_s": elapsed,
        }

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, artifact_path="model")

        print(f"  Best CV MAE:  {metrics['cv_mae_mean']:.4f} +/- {metrics['cv_mae_std']:.4f}")
        print(f"  Best CV R²:   {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")
        print(f"  Best CV RMSE: {metrics['cv_rmse_mean']:.4f} +/- {metrics['cv_rmse_std']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    return {
        "name": name,
        "estimator": best,
        "best_params": search.best_params_,
        "metrics": metrics,
    }


def train_all_models(X: np.ndarray, y: np.ndarray) -> list[dict]:
    setup_mlflow()
    registry = get_model_registry()
    results = []
    for entry in registry:
        result = train_single_model(
            name=entry["name"],
            estimator=entry["estimator"],
            params=entry["params"],
            X=X,
            y=y,
        )
        results.append(result)
    return results
