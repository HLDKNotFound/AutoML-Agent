"""
Model Factory module.
Instantiates scikit-learn, LightGBM, and XGBoost estimators safely with configured hyperparameters and seeds.
"""

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

    # 2. Gradient Boosting
    elif "gradientboosting" in name:
        if is_classification:
            return GradientBoostingClassifier(random_state=config.random_state, **params)
        return GradientBoostingRegressor(random_state=config.random_state, **params)

    elif "histgradientboosting" in name:
        if is_classification:
            return HistGradientBoostingClassifier(random_state=config.random_state, **params)
        return HistGradientBoostingRegressor(random_state=config.random_state, **params)

    # 3. LightGBM
    elif "lightgbm" in name or "lgbm" in name:
        try:
            import lightgbm as lgb
            if is_classification:
                return lgb.LGBMClassifier(random_state=config.random_state, verbose=-1, **params)
            return lgb.LGBMRegressor(random_state=config.random_state, verbose=-1, **params)
        except ImportError:
            # Fallback to HistGradientBoosting
            if is_classification:
                return HistGradientBoostingClassifier(random_state=config.random_state, **params)
            return HistGradientBoostingRegressor(random_state=config.random_state, **params)

    # 4. XGBoost
    elif "xgboost" in name or "xgb" in name:
        try:
            import xgboost as xgb
            if is_classification:
                return xgb.XGBClassifier(random_state=config.random_state, eval_metric="logloss", **params)
            return xgb.XGBRegressor(random_state=config.random_state, **params)
        except ImportError:
            if is_classification:
                return GradientBoostingClassifier(random_state=config.random_state, **params)
            return GradientBoostingRegressor(random_state=config.random_state, **params)

    # 5. Linear / Logistic / Ridge
    elif "logistic" in name:
        # Default max_iter 1000 for convergence
        if "max_iter" not in params:
            params["max_iter"] = 1000
        return LogisticRegression(random_state=config.random_state, **params)

    elif "ridge" in name:
        return Ridge(random_state=config.random_state, **params)

    # 6. Support Vector Machine
    elif "svc" in name:
        # Enable probability for ROC-AUC and log-loss evaluation
        return SVC(probability=True, random_state=config.random_state, **params)

    elif "svr" in name:
        return SVR(**params)

    # 7. K-Nearest Neighbors
    elif "kneighbors" in name:
        if is_classification:
            return KNeighborsClassifier(**params)
        return KNeighborsRegressor(**params)

    # Fallback default
    if is_classification:
        return RandomForestClassifier(random_state=config.random_state, **params)
    return RandomForestRegressor(random_state=config.random_state, **params)
