from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
TRAINING_DATA = DATA_DIR / "training_data.csv"
BLIND_TEST_DATA = DATA_DIR / "blind_test_data.csv"

# ML config
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 50

# MLflow
MLFLOW_EXPERIMENT_NAME = "mlops-pipeline-regression"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

# Scoring
CV_SCORING = "neg_mean_absolute_error"
