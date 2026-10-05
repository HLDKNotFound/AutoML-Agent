"""
Hyperparameter Strategy Agent.
Defines coarse (broad) and fine (Bayesian) search distributions for each candidate model.
Strictly abides by scale principles: log-uniform for learning rate and regularization,
uniform for sampling ratios, and discrete for tree depth and estimators.
"""

from typing import Any, Dict
from backend.app.schemas.ml_schemas import HyperparameterSearchPlan


def get_search_plan_for_model(model_name: str, problem_type: str) -> HyperparameterSearchPlan:
    """
    Returns search spaces for Stage 1 (Coarse) and Stage 2 (Fine Bayesian).
    """
    name = model_name.lower()
    
    # 1. Random Forest (Classifier / Regressor)
    if "randomforest" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 20, "high": 200, "step": 20},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 15},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 250, "step": 10},
            "max_depth": {"type": "int", "low": 4, "high": 16},
            "min_samples_split": {"type": "int", "low": 2, "high": 8},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 5}
        }
        notes = "Tree ensemble: coarse broad depth scan followed by fine tuning of tree complexity."

    # 2. Gradient Boosting (GBDT)
    elif "gradientboosting" in name or "histgradientboosting" in name or "lightgbm" in name or "xgboost" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 30, "high": 180, "step": 30},
            "learning_rate": {"type": "log_float", "low": 1e-3, "high": 0.3},
            "max_depth": {"type": "int", "low": 2, "high": 10},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "step": 10},
            "learning_rate": {"type": "log_float", "low": 5e-3, "high": 0.2},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0}
        }
        notes = "GBDT strategy: learning_rate and n_estimators tuned jointly with log-uniform scale and tree depth."

    # 3. Linear / Logistic / Ridge
    elif "logistic" in name or "ridge" in name:
        param_name = "C" if "logistic" in name else "alpha"
        coarse_space = {
            param_name: {"type": "log_float", "low": 1e-4, "high": 1e3}
        }
        fine_space = {
            param_name: {"type": "log_float", "low": 1e-2, "high": 1e2}
        }
        notes = f"Regularization parameter ({param_name}) searched across log-uniform orders of magnitude."

    # 4. Support Vector (SVC / SVR)
    elif "svc" in name or "svr" in name:
        coarse_space = {
            "C": {"type": "log_float", "low": 1e-3, "high": 1e2},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]}
        }
        fine_space = {
            "C": {"type": "log_float", "low": 1e-2, "high": 50.0}
        }
        notes = "Kernel SVM: margin softness C optimized over log-uniform space."

    # 5. K-Nearest Neighbors
    elif "kneighbors" in name:
        coarse_space = {
            "n_neighbors": {"type": "int", "low": 2, "high": 30},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "p": {"type": "categorical", "choices": [1, 2]}
        }
        fine_space = {
            "n_neighbors": {"type": "int", "low": 3, "high": 20},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]}
        }
        notes = "Instance-based search: neighbor cardinality and distance weighting."

    # Default fallback
    else:
        coarse_space = {"n_estimators": {"type": "int", "low": 10, "high": 100}}
        fine_space = {"n_estimators": {"type": "int", "low": 30, "high": 150}}
        notes = "Generic search space."

    return HyperparameterSearchPlan(
        model_name=model_name,
        coarse_space=coarse_space,
        fine_space=fine_space,
        strategy_notes=notes
    )
