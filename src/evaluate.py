"""Modulo de evaluacion y comparacion de modelos.

Proporciona funciones para comparar metricas de validacion cruzada
entre modelos, seleccionar el mejor, y evaluar en el conjunto hold-out.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Construye una tabla comparativa de todos los modelos ordenada por CV MAE.

    Args:
        results: Lista de diccionarios con resultados de entrenamiento.

    Returns:
        DataFrame con metricas formateadas y columna auxiliar para ordenamiento.
    """
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Model": r["name"],
            "CV MAE": f"{m['cv_mae_mean']:.4f} +/- {m['cv_mae_std']:.4f}",
            "CV R²": f"{m['cv_r2_mean']:.4f} +/- {m['cv_r2_std']:.4f}",
            "CV RMSE": f"{m['cv_rmse_mean']:.4f} +/- {m['cv_rmse_std']:.4f}",
            "Time (s)": f"{m['training_time_s']:.1f}",
            "_sort_mae": m["cv_mae_mean"],  # Columna auxiliar para ordenar
        })
    df = pd.DataFrame(rows).sort_values("_sort_mae")  # Menor MAE primero
    return df


def select_best_model(results: list[dict]) -> dict:
    """Selecciona el modelo con menor CV MAE (error absoluto medio).

    Args:
        results: Lista de diccionarios con resultados de entrenamiento.

    Returns:
        Diccionario del modelo con mejor rendimiento.
    """
    return min(results, key=lambda r: r["metrics"]["cv_mae_mean"])


def evaluate_on_holdout(
    estimator, X_val: np.ndarray, y_val: np.ndarray
) -> dict:
    """Evalua un estimador/pipeline en el conjunto de validacion reservado (hold-out).

    Calcula MAE, RMSE y R² sobre datos no vistos durante entrenamiento.

    Args:
        estimator: Modelo o Pipeline entrenado (con metodo .predict()).
        X_val: Features del conjunto de validacion.
        y_val: Target real del conjunto de validacion.

    Returns:
        Diccionario con metricas de hold-out.
    """
    y_pred = estimator.predict(X_val)
    metrics = {
        "holdout_mae": mean_absolute_error(y_val, y_pred),             # Error absoluto medio
        "holdout_rmse": np.sqrt(mean_squared_error(y_val, y_pred)),    # Raiz del error cuadratico medio
        "holdout_r2": r2_score(y_val, y_pred),                         # Coeficiente de determinacion
    }
    return metrics


def print_comparison_table(comparison: pd.DataFrame) -> None:
    """Imprime la tabla de comparacion de modelos formateada en consola.

    Args:
        comparison: DataFrame generado por compare_models().
    """
    display_df = comparison.drop(columns=["_sort_mae"])  # Ocultar columna auxiliar
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (sorted by CV MAE)")
    print("=" * 80)
    print(display_df.to_string(index=False))
    print("=" * 80)
