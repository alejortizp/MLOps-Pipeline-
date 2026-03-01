from scipy.stats import loguniform, uniform, randint
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import RANDOM_STATE


def get_model_registry() -> list[dict]:
    return [
        {
            "name": "Ridge",
            "estimator": Ridge(),
            "params": {
                "model__alpha": loguniform(1e-3, 1e3),
            },
        },
        {
            "name": "ElasticNet",
            "estimator": ElasticNet(max_iter=10_000),
            "params": {
                "model__alpha": loguniform(1e-3, 1e3),
                "model__l1_ratio": uniform(0.01, 0.99),
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(3, 30),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": uniform(0.3, 0.7),
            },
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.5, 0.5),
            },
        },
        {
            "name": "XGBoost",
            "estimator": XGBRegressor(
                random_state=RANDOM_STATE, verbosity=0, n_jobs=1
            ),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.5, 0.5),
                "model__colsample_bytree": uniform(0.3, 0.7),
                "model__reg_alpha": loguniform(1e-3, 10),
                "model__reg_lambda": loguniform(1e-3, 10),
                "model__min_child_weight": randint(1, 10),
            },
        },
        {
            "name": "LightGBM",
            "estimator": LGBMRegressor(
                random_state=RANDOM_STATE, verbosity=-1, n_jobs=1
            ),
            "params": {
                "model__n_estimators": randint(50, 500),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": loguniform(1e-3, 1),
                "model__subsample": uniform(0.5, 0.5),
                "model__colsample_bytree": uniform(0.3, 0.7),
                "model__num_leaves": randint(10, 100),
                "model__min_child_samples": randint(5, 50),
            },
        },
        {
            "name": "SVR",
            "estimator": SVR(),
            "params": {
                "model__C": loguniform(1e-2, 1e3),
                "model__epsilon": loguniform(1e-3, 1),
                "model__kernel": ["rbf", "linear", "poly"],
                "model__gamma": ["scale", "auto"],
            },
        },
    ]
