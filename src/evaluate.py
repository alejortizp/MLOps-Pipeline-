import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compare_models(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Model": r["name"],
            "CV MAE": f"{m['cv_mae_mean']:.4f} ± {m['cv_mae_std']:.4f}",
            "CV R²": f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}",
            "CV RMSE": f"{m['cv_rmse_mean']:.4f} ± {m['cv_rmse_std']:.4f}",
            "Time (s)": f"{m['training_time_s']:.1f}",
            "_sort_mae": m["cv_mae_mean"],
        })
    df = pd.DataFrame(rows).sort_values("_sort_mae")
    return df


def select_best_model(results: list[dict]) -> dict:
    return min(results, key=lambda r: r["metrics"]["cv_mae_mean"])


def evaluate_on_holdout(
    estimator, X_val: np.ndarray, y_val: np.ndarray
) -> dict:
    y_pred = estimator.predict(X_val)
    metrics = {
        "holdout_mae": mean_absolute_error(y_val, y_pred),
        "holdout_rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "holdout_r2": r2_score(y_val, y_pred),
    }
    return metrics


def print_comparison_table(comparison: pd.DataFrame) -> None:
    display_df = comparison.drop(columns=["_sort_mae"])
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (sorted by CV MAE)")
    print("=" * 80)
    print(display_df.to_string(index=False))
    print("=" * 80)
