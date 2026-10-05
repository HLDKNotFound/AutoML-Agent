"""
Model Factory module.
Instantiates scikit-learn, LightGBM, and XGBoost estimators safely with configured hyperparameters and seeds.
"""

from typing import Any, Dict
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from backend.app.core.config import config


def create_base_estimator(model_name: str, problem_type: str, params: Dict[str, Any] = None):
    """
    Returns an un-fitted estimator instance with the provided hyperparameters.
    """
    params = dict(params or {})
    name = model_name.lower()
    is_classification = problem_type in ("binary_classification", "multiclass_classification")

    # 1. Random Forest
    if "randomforest" in name:
        if is_classification:
            return RandomForestClassifier(random_state=config.random_state, **params)
        return RandomForestRegressor(random_state=config.random_state, **params)

    # 2. Hist Gradient Boosting (must be matched before generic gradientboosting)
    elif "histgradientboosting" in name:
        hgb_params = dict(params)
        if "n_estimators" in hgb_params:
            hgb_params["max_iter"] = hgb_params.pop("n_estimators")
        hgb_params.pop("subsample", None)
        if is_classification:
            return HistGradientBoostingClassifier(random_state=config.random_state, **hgb_params)
        return HistGradientBoostingRegressor(random_state=config.random_state, **hgb_params)

    # 3. Standard Gradient Boosting
    elif "gradientboosting" in name:
        gb_params = dict(params)
        # Cap max_depth at 5 to avoid exponential O(2^depth) slowdown on tabular data
        if gb_params.get("max_depth", 3) > 5:
            gb_params["max_depth"] = 5
        if "max_features" not in gb_params:
            gb_params["max_features"] = "sqrt"
        if is_classification:
            return GradientBoostingClassifier(random_state=config.random_state, **gb_params)
        return GradientBoostingRegressor(random_state=config.random_state, **gb_params)

    # 4. LightGBM
    elif "lightgbm" in name or "lgbm" in name:
        try:
            import lightgbm as lgb
            lgb_params = dict(params)
            lgb_params.setdefault("verbose", -1)
            lgb_params.setdefault("n_jobs", -1)
            if is_classification:
                return lgb.LGBMClassifier(random_state=config.random_state, **lgb_params)
            return lgb.LGBMRegressor(random_state=config.random_state, **lgb_params)
        except ImportError:
            if is_classification:
                return HistGradientBoostingClassifier(random_state=config.random_state, **params)
            return HistGradientBoostingRegressor(random_state=config.random_state, **params)

    # 5. XGBoost
    elif "xgboost" in name or "xgb" in name:
        try:
            import xgboost as xgb
            xgb_params = dict(params)
            xgb_params.setdefault("verbosity", 0)
            xgb_params.setdefault("n_jobs", -1)
            if is_classification:
                xgb_params.setdefault("eval_metric", "logloss")
                return xgb.XGBClassifier(random_state=config.random_state, **xgb_params)
            return xgb.XGBRegressor(random_state=config.random_state, **xgb_params)
        except ImportError:
            if is_classification:
                return GradientBoostingClassifier(random_state=config.random_state, **params)
            return GradientBoostingRegressor(random_state=config.random_state, **params)

    # 6. CatBoost
    elif "catboost" in name or "cat" in name:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            cat_params = dict(params)
            cat_params.setdefault("verbose", 0)
            cat_params.setdefault("thread_count", -1)
            cat_params.setdefault("random_seed", config.random_state)
            if is_classification:
                return CatBoostClassifier(**cat_params)
            return CatBoostRegressor(**cat_params)
        except ImportError:
            if is_classification:
                return HistGradientBoostingClassifier(random_state=config.random_state, **params)
            return HistGradientBoostingRegressor(random_state=config.random_state, **params)

    # 5. Linear / Logistic / Ridge
    elif "logistic" in name:
        # Default max_iter 1000 for convergence
        if "max_iter" not in params:
            params["max_iter"] = 1000
        return LogisticRegression(random_state=config.random_state, **params)

    elif "ridge" in name:
        return Ridge(random_state=config.random_state, **params)

    # 6. Extra Trees
    elif "extratrees" in name:
        if is_classification:
            return ExtraTreesClassifier(random_state=config.random_state, **params)
        return ExtraTreesRegressor(random_state=config.random_state, **params)

    # 7. Support Vector Machine
    elif "svc" in name:
        if "max_iter" not in params:
            params["max_iter"] = 2000
        # Enable probability for ROC-AUC and log-loss evaluation
        return SVC(probability=True, random_state=config.random_state, **params)

    elif "svr" in name:
        if "max_iter" not in params:
            params["max_iter"] = 2000
        return SVR(**params)

    # 8. K-Nearest Neighbors
    elif "kneighbors" in name:
        if is_classification:
            return KNeighborsClassifier(**params)
        return KNeighborsRegressor(**params)

    # Fallback default
    if is_classification:
        return RandomForestClassifier(random_state=config.random_state, **params)
    return RandomForestRegressor(random_state=config.random_state, **params)
