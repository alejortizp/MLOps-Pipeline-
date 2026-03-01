"""Modulo de entrenamiento de modelos.

Crea un sklearn.Pipeline (StandardScaler + estimador) para cada modelo,
ejecuta busqueda aleatoria de hiperparametros con validacion cruzada,
y registra resultados en MLflow.
"""

import time

import mlflow
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    """Configura la conexion a MLflow con URI y nombre de experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_single_model(
    name: str,
    estimator,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Entrena un modelo individual con busqueda de hiperparametros.

    Args:
        name: Nombre identificador del modelo.
        estimator: Instancia del estimador (sin ajustar).
        params: Distribuciones de hiperparametros (con prefijo model__).
        X: Features de entrenamiento (sin escalar, el Pipeline escala internamente).
        y: Target de entrenamiento.

    Returns:
        Diccionario con nombre, pipeline ajustado, mejores parametros y metricas.
    """
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    # Crear Pipeline: estandarizar features → modelo de regresion
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Estandariza a media=0, desviacion=1
        ("model", estimator),          # Modelo de regresion
    ])

    with mlflow.start_run(run_name=name):
        start = time.time()

        # Busqueda aleatoria de hiperparametros con validacion cruzada
        search = RandomizedSearchCV(
            estimator=pipeline,           # Pipeline completo como estimador
            param_distributions=params,   # Distribuciones con prefijo model__
            n_iter=N_ITER,                # 50 combinaciones aleatorias
            cv=CV_FOLDS,                  # Validacion cruzada de 5 folds
            scoring=CV_SCORING,           # Optimizar por MAE
            random_state=RANDOM_STATE,    # Semilla para reproducibilidad
            n_jobs=-1,                    # Usar todos los cores disponibles
            return_train_score=False,     # No guardar scores de entrenamiento
        )
        search.fit(X, y)
        elapsed = time.time() - start

        # Pipeline con los mejores hiperparametros encontrados
        best_pipeline = search.best_estimator_

        # Calcular metricas adicionales de CV con el mejor pipeline
        mae_scores = cross_val_score(
            best_pipeline, X, y, cv=CV_FOLDS, scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(
            best_pipeline, X, y, cv=CV_FOLDS, scoring="r2"
        )
        rmse_scores = cross_val_score(
            best_pipeline, X, y, cv=CV_FOLDS, scoring="neg_root_mean_squared_error"
        )

        # Compilar metricas (negamos MAE y RMSE porque sklearn usa convenciones negativas)
        metrics = {
            "cv_mae_mean": -mae_scores.mean(),    # Error absoluto medio (promedio)
            "cv_mae_std": mae_scores.std(),        # Desviacion estandar del MAE
            "cv_r2_mean": r2_scores.mean(),        # Coeficiente de determinacion (promedio)
            "cv_r2_std": r2_scores.std(),          # Desviacion estandar del R²
            "cv_rmse_mean": -rmse_scores.mean(),   # Raiz del error cuadratico medio (promedio)
            "cv_rmse_std": rmse_scores.std(),      # Desviacion estandar del RMSE
            "training_time_s": elapsed,            # Tiempo total de entrenamiento (segundos)
        }

        # Registrar en MLflow: parametros, metricas y modelo
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        print(f"  Best CV MAE:  {metrics['cv_mae_mean']:.4f} +/- {metrics['cv_mae_std']:.4f}")
        print(f"  Best CV R²:   {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")
        print(f"  Best CV RMSE: {metrics['cv_rmse_mean']:.4f} +/- {metrics['cv_rmse_std']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    return {
        "name": name,
        "pipeline": best_pipeline,          # Pipeline completo ajustado
        "best_params": search.best_params_, # Mejores hiperparametros encontrados
        "metrics": metrics,                 # Metricas de validacion cruzada
    }


def train_all_models(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Entrena todos los modelos del registro y retorna sus resultados.

    Args:
        X: Features de entrenamiento (sin escalar).
        y: Target de entrenamiento.

    Returns:
        Lista de diccionarios con resultados de cada modelo.
    """
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
