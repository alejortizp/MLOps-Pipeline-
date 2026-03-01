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
    # 1. Load data
    print("Step 1: Loading data...")
    X, y = load_training_data()
    X_blind = load_blind_test_data()
    print(f"  Training: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Blind test: {X_blind.shape[0]} samples")

    # 2. Split data (no manual scaling — Pipeline handles it)
    print("\nStep 2: Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"  Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    # 3. Train all models (Pipeline scales internally)
    print("\nStep 3: Training 7 models with hyperparameter tuning...")
    results = train_all_models(X_train, y_train)

    # 4. Compare models
    print("\nStep 4: Comparing models...")
    comparison = compare_models(results)
    print_comparison_table(comparison)

    # 5. Select best and evaluate on hold-out
    best = select_best_model(results)
    print(f"\nStep 5: Best model → {best['name']}")
    holdout_metrics = evaluate_on_holdout(best["pipeline"], X_val, y_val)
    print(f"  Hold-out MAE:  {holdout_metrics['holdout_mae']:.4f}")
    print(f"  Hold-out RMSE: {holdout_metrics['holdout_rmse']:.4f}")
    print(f"  Hold-out R²:   {holdout_metrics['holdout_r2']:.4f}")

    # 6. Generate blind test predictions
    print("\nStep 6: Generating blind test predictions...")
    predictions = generate_predictions(best["pipeline"], X_blind)
    save_predictions(predictions)

    # 7. Save model artifact
    print("\nStep 7: Saving model artifact...")
    save_model_artifact(best["pipeline"], best["name"])

    # 8. Summary
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
