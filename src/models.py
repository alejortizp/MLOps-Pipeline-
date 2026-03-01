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
                "alpha": loguniform(1e-3, 1e3),
            },
        },
        {
            "name": "ElasticNet",
            "estimator": ElasticNet(max_iter=10_000),
            "params": {
                "alpha": loguniform(1e-3, 1e3),
                "l1_ratio": uniform(0.01, 0.99),
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": randint(50, 500),
                "max_depth": randint(3, 30),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": uniform(0.3, 0.7),
            },
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": randint(50, 500),
                "max_depth": randint(2, 10),
                "learning_rate": loguniform(1e-3, 1),
                "subsample": uniform(0.5, 0.5),
            },
        },
        {
            "name": "XGBoost",
            "estimator": XGBRegressor(
                random_state=RANDOM_STATE, verbosity=0, n_jobs=1
            ),
            "params": {
                "n_estimators": randint(50, 500),
                "max_depth": randint(2, 10),
                "learning_rate": loguniform(1e-3, 1),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.3, 0.7),
                "reg_alpha": loguniform(1e-3, 10),
                "reg_lambda": loguniform(1e-3, 10),
                "min_child_weight": randint(1, 10),
            },
        },
        {
            "name": "LightGBM",
            "estimator": LGBMRegressor(
                random_state=RANDOM_STATE, verbosity=-1, n_jobs=1
            ),
            "params": {
                "n_estimators": randint(50, 500),
                "max_depth": randint(2, 10),
                "learning_rate": loguniform(1e-3, 1),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.3, 0.7),
                "num_leaves": randint(10, 100),
                "min_child_samples": randint(5, 50),
            },
        },
        {
            "name": "SVR",
            "estimator": SVR(),
            "params": {
                "C": loguniform(1e-2, 1e3),
                "epsilon": loguniform(1e-3, 1),
                "kernel": ["rbf", "linear", "poly"],
                "gamma": ["scale", "auto"],
            },
        },
    ]
