"""Registro de modelos de regresion.

Define los 7 algoritmos con sus distribuciones de hiperparametros
para la busqueda aleatoria. Los parametros usan prefijo 'model__'
para ser compatibles con el Pipeline de sklearn (paso "model").
"""

from scipy.stats import loguniform, uniform, randint
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import RANDOM_STATE


def get_model_registry() -> list[dict]:
    """Retorna la lista de modelos con estimador y distribucion de hiperparametros.

    Cada entrada contiene:
      - name: nombre identificador del modelo
      - estimator: instancia del estimador (se usara dentro del Pipeline)
      - params: distribuciones para RandomizedSearchCV (prefijo model__)
    """
    return [
        # --- Modelos lineales ---
        {
            "name": "Ridge",
            "estimator": Ridge(),
            "params": {
                # alpha: fuerza de regularizacion L2 (busqueda en escala logaritmica)
                "model__alpha": loguniform(1e-3, 1e3),
            },
        },
        {
            "name": "ElasticNet",
            "estimator": ElasticNet(max_iter=10_000),  # max_iter alto para convergencia
            "params": {
                # alpha: fuerza de regularizacion combinada L1+L2
                "model__alpha": loguniform(1e-3, 1e3),
                # l1_ratio: balance entre L1 (0) y L2 (1)
                "model__l1_ratio": uniform(0.01, 0.99),
            },
        },
        # --- Modelos basados en arboles ---
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": randint(50, 500),       # Cantidad de arboles
                "model__max_depth": randint(3, 30),            # Profundidad maxima
                "model__min_samples_split": randint(2, 20),    # Muestras minimas para dividir nodo
                "model__min_samples_leaf": randint(1, 10),     # Muestras minimas en hoja
                "model__max_features": uniform(0.3, 0.7),      # Fraccion de features por arbol
            },
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": randint(50, 500),       # Cantidad de etapas de boosting
                "model__max_depth": randint(2, 10),            # Profundidad de cada arbol
                "model__learning_rate": loguniform(1e-3, 1),   # Tasa de aprendizaje
                "model__subsample": uniform(0.5, 0.5),         # Fraccion de muestras por arbol
            },
        },
        {
            "name": "XGBoost",
            "estimator": XGBRegressor(
                random_state=RANDOM_STATE,
                verbosity=0,   # Sin mensajes de log
                n_jobs=1,      # Un hilo (paralelismo lo maneja RandomizedSearchCV)
            ),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.5, 0.5),
                "model__colsample_bytree": uniform(0.3, 0.7),  # Fraccion de features por arbol
                "model__reg_alpha": loguniform(1e-3, 10),       # Regularizacion L1
                "model__reg_lambda": loguniform(1e-3, 10),      # Regularizacion L2
                "model__min_child_weight": randint(1, 10),      # Peso minimo en hoja
            },
        },
        {
            "name": "LightGBM",
            "estimator": LGBMRegressor(
                random_state=RANDOM_STATE,
                verbosity=-1,  # Sin mensajes de log
                n_jobs=1,      # Un hilo (paralelismo lo maneja RandomizedSearchCV)
            ),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.5, 0.5),
                "model__colsample_bytree": uniform(0.3, 0.7),
                "model__num_leaves": randint(10, 100),           # Numero de hojas por arbol
                "model__min_child_samples": randint(5, 50),      # Muestras minimas en hoja
            },
        },
        # --- Modelo de soporte vectorial ---
        {
            "name": "SVR",
            "estimator": SVR(),
            "params": {
                "model__C": loguniform(1e-2, 1e3),              # Parametro de regularizacion
                "model__epsilon": loguniform(1e-3, 1),           # Margen de tolerancia
                "model__kernel": ["rbf", "linear", "poly"],      # Tipo de kernel
                "model__gamma": ["scale", "auto"],               # Coeficiente del kernel
            },
        },
    ]
