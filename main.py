"""Punto de entrada principal del pipeline de ML.

Orquesta todo el flujo: carga de datos → division → entrenamiento de 7 modelos
con busqueda de hiperparametros → comparacion → evaluacion hold-out →
prediccion de test ciego → guardado del mejor modelo.
"""

from src.data_loader import (
    load_blind_test_data,
    load_training_data,
    split_data,
)
from src.evaluate import (
    compare_models,
    evaluate_on_holdout,
    print_comparison_table,
    select_best_model,
)
from src.predict import generate_predictions, save_model_artifact, save_predictions
from src.train import train_all_models


def main():
    # --- Paso 1: Cargar datos ---
    # Lee los CSVs de entrenamiento (800 muestras) y test ciego (200 muestras)
    print("Step 1: Loading data...")
    X, y = load_training_data()
    X_blind = load_blind_test_data()
    print(f"  Training: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Blind test: {X_blind.shape[0]} samples")

    # --- Paso 2: Dividir datos ---
    # Separa 80% entrenamiento / 20% validacion (sin escalar, el Pipeline lo hace)
    print("\nStep 2: Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"  Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    # --- Paso 3: Entrenar todos los modelos ---
    # Cada modelo se envuelve en Pipeline(StandardScaler + estimador)
    # Se optimizan hiperparametros con RandomizedSearchCV (50 iter, 5-fold CV)
    print("\nStep 3: Training 7 models with hyperparameter tuning...")
    results = train_all_models(X_train, y_train)

    # --- Paso 4: Comparar modelos ---
    # Genera tabla ranking ordenada por CV MAE (menor es mejor)
    print("\nStep 4: Comparing models...")
    comparison = compare_models(results)
    print_comparison_table(comparison)

    # --- Paso 5: Seleccionar mejor modelo y evaluar en hold-out ---
    # Evalua el mejor pipeline sobre el conjunto de validacion reservado
    best = select_best_model(results)
    print(f"\nStep 5: Best model → {best['name']}")
    holdout_metrics = evaluate_on_holdout(best["pipeline"], X_val, y_val)
    print(f"  Hold-out MAE:  {holdout_metrics['holdout_mae']:.4f}")
    print(f"  Hold-out RMSE: {holdout_metrics['holdout_rmse']:.4f}")
    print(f"  Hold-out R²:   {holdout_metrics['holdout_r2']:.4f}")

    # --- Paso 6: Generar predicciones del test ciego ---
    # El pipeline escala internamente, no necesita escalado manual
    print("\nStep 6: Generating blind test predictions...")
    predictions = generate_predictions(best["pipeline"], X_blind)
    save_predictions(predictions)

    # --- Paso 7: Guardar artefacto del modelo ---
    # Serializa el Pipeline completo (scaler + modelo) con joblib
    print("\nStep 7: Saving model artifact...")
    save_model_artifact(best["pipeline"], best["name"])

    # --- Paso 8: Resumen final ---
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model: {best['name']}")
    print(f"  CV MAE:     {best['metrics']['cv_mae_mean']:.4f}")
    print(f"  CV R²:      {best['metrics']['cv_r2_mean']:.4f}")
    print(f"  Holdout MAE: {holdout_metrics['holdout_mae']:.4f}")
    print(f"  Holdout R²:  {holdout_metrics['holdout_r2']:.4f}")


if __name__ == "__main__":
    main()
