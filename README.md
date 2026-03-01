# Pipeline de ML con MLflow: benchmark de 7 modelos de regresión

Multivariate regression problem: train on 800 samples (20 features → 1 continuous target), predict 200 blind test samples, and deliver a production-ready solution.

## Quick Start

```bash
# Install dependencies (requires uv and Python 3.14+)
uv sync

# Run the full pipeline (train → evaluate → predict → save)
uv run python main.py
```

## Pipeline Overview

`main.py` executes the following steps:

1. **Load data** — training (800 samples) and blind test (200 samples)
2. **Split** — 80/20 train/validation split (stratified by `random_state=42`)
3. **Train 7 models** — each wrapped in a `sklearn.Pipeline` (StandardScaler + estimator) with `RandomizedSearchCV` hyperparameter tuning (50 iterations, 5-fold CV)
4. **Compare** — rank models by CV MAE
5. **Evaluate** — hold-out set metrics (MAE, RMSE, R²)
6. **Predict** — generate blind test predictions → `outputs/target_pred.csv`
7. **Save** — serialize best pipeline → `models/best_model_<name>.joblib`

All experiments are tracked with **MLflow** (local tracking under `mlruns/`).

## Models

| Model | Library |
|---|---|
| Ridge | scikit-learn |
| ElasticNet | scikit-learn |
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| SVR | scikit-learn |

## Project Structure

```
main.py                 # Entry point — runs the full pipeline
src/
  config.py             # Paths, hyperparams, MLflow settings
  data_loader.py        # Load CSVs and train/val split
  models.py             # Model registry (estimators + param distributions)
  train.py              # Pipeline creation, RandomizedSearchCV, MLflow logging
  evaluate.py           # Model comparison and hold-out evaluation
  predict.py            # Generate predictions and save artifacts
data/
  training_data.csv     # 800 rows × 21 cols (features + target)
  blind_test_data.csv   # 200 rows × 20 cols (features only)
models/                 # Serialized sklearn.Pipeline (joblib)
outputs/                # target_pred.csv (200 rows, target_pred column)
notebooks/
  data_analisys.ipynb   # Exploratory data analysis
mlruns/                 # MLflow experiment tracking
```

## Architecture

Each model is wrapped in a **sklearn.Pipeline**:

```
Pipeline([
    ("scaler", StandardScaler()),
    ("model",  <estimator>),
])
```

This ensures preprocessing and prediction are a single atomic unit. The serialized artifact is the complete Pipeline — inference requires only:

```python
import joblib
pipeline = joblib.load("models/best_model_gradientboosting.joblib")
predictions = pipeline.predict(X_raw)  # no manual scaling needed
```

## Configuration

Key settings in `src/config.py`:

| Parameter | Value |
|---|---|
| `RANDOM_STATE` | 42 |
| `TEST_SIZE` | 0.2 |
| `CV_FOLDS` | 5 |
| `N_ITER` | 50 |
| `CV_SCORING` | neg_mean_absolute_error |

## Tech Stack

- **Python 3.14** (managed via `uv`)
- scikit-learn, xgboost, lightgbm — modeling
- pandas, numpy, scipy — data manipulation
- MLflow — experiment tracking
- joblib — model serialization
- matplotlib, seaborn — visualization (EDA notebook)
