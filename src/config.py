"""Configuracion centralizada del proyecto.

Define todas las constantes: rutas de archivos, parametros de ML,
configuracion de MLflow y metrica de evaluacion.
"""

from pathlib import Path

# --- Rutas del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Raiz del repositorio
DATA_DIR = PROJECT_ROOT / "data"        # Directorio con los CSVs de datos
MODELS_DIR = PROJECT_ROOT / "models"    # Directorio para guardar modelos entrenados
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Directorio para guardar predicciones

# --- Archivos de datos ---
TRAINING_DATA = DATA_DIR / "training_data.csv"      # 800 muestras con target
BLIND_TEST_DATA = DATA_DIR / "blind_test_data.csv"  # 200 muestras sin target

# --- Configuracion de Machine Learning ---
RANDOM_STATE = 42    # Semilla aleatoria para reproducibilidad
TEST_SIZE = 0.2      # Proporcion de datos reservados para validacion (20%)
CV_FOLDS = 5         # Numero de folds en validacion cruzada
N_ITER = 50          # Iteraciones de busqueda aleatoria de hiperparametros

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "mlops-pipeline-regression"               # Nombre del experimento
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()     # URI local de tracking

# --- Metrica de evaluacion ---
CV_SCORING = "neg_mean_absolute_error"  # Metrica para optimizar (MAE negativo, convencion sklearn)
