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

    # 2. CatBoost
    elif "catboost" in name or "cat" in name:
        coarse_space = {
            "iterations": {"type": "int", "low": 50, "high": 250, "step": 50},
            "learning_rate": {"type": "log_float", "low": 0.01, "high": 0.2},
            "depth": {"type": "int", "low": 4, "high": 8},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0}
        }
        fine_space = {
            "iterations": {"type": "int", "low": 80, "high": 300, "step": 20},
            "learning_rate": {"type": "log_float", "low": 0.02, "high": 0.15},
            "depth": {"type": "int", "low": 4, "high": 7},
            "l2_leaf_reg": {"type": "float", "low": 2.0, "high": 8.0}
        }
        notes = "CatBoost: symmetric oblivious trees tuned for categorical feature interactions and regularization."

    # 3. LightGBM / XGBoost / HistGradientBoosting (Histogram-accelerated)
    elif "lightgbm" in name or "xgboost" in name or "histgradientboosting" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 40, "high": 180, "step": 35},
            "learning_rate": {"type": "log_float", "low": 0.01, "high": 0.25},
            "max_depth": {"type": "int", "low": 3, "high": 8}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 60, "high": 200, "step": 20},
            "learning_rate": {"type": "log_float", "low": 0.02, "high": 0.15},
            "max_depth": {"type": "int", "low": 4, "high": 7}
        }
        notes = "Histogram GBDT: fast binning allows deep tree exploration."

    # 3. Standard Gradient Boosting (Sequential exact greedy splits)
    elif "gradientboosting" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 20, "high": 80, "step": 20},
            "learning_rate": {"type": "log_float", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 2, "high": 5},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 20, "high": 80, "step": 10},
            "learning_rate": {"type": "log_float", "low": 0.02, "high": 0.2},
            "max_depth": {"type": "int", "low": 2, "high": 4},
            "subsample": {"type": "float", "low": 0.8, "high": 1.0}
        }
        notes = "Standard GBDT: shallow trees (depth 2-5) with sqrt subsampling for fast convergence."

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

    # 5. Extra Trees
    elif "extratrees" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 20, "high": 180, "step": 20},
            "max_depth": {"type": "int", "low": 3, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 15}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 40, "high": 200, "step": 10},
            "max_depth": {"type": "int", "low": 4, "high": 16},
            "min_samples_split": {"type": "int", "low": 2, "high": 8}
        }
        notes = "ExtraTrees randomized ensemble: tree complexity and split depth tuning."

    # 6. K-Nearest Neighbors
    elif "kneighbors" in name or "knn" in name:
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

    # 7. AdaBoost
    elif "adaboost" in name:
        coarse_space = {
            "n_estimators": {"type": "int", "low": 30, "high": 200, "step": 20},
            "learning_rate": {"type": "log_float", "low": 0.01, "high": 1.0}
        }
        fine_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 180, "step": 15},
            "learning_rate": {"type": "log_float", "low": 0.05, "high": 0.8}
        }
        notes = "AdaBoost adaptive boosting: estimator count and learning rate shrinkage."

    # 8. Multi-Layer Perceptron (Neural Network)
    elif "mlp" in name:
        coarse_space = {
            "alpha": {"type": "log_float", "low": 1e-5, "high": 1e-1},
            "learning_rate_init": {"type": "log_float", "low": 1e-4, "high": 1e-2}
        }
        fine_space = {
            "alpha": {"type": "log_float", "low": 1e-4, "high": 1e-2},
            "learning_rate_init": {"type": "log_float", "low": 5e-4, "high": 5e-3}
        }
        notes = "MLP Neural Network: weight decay L2 penalty alpha and Adam learning rate."

    # 9. Naive Bayes (GaussianNB)
    elif "gaussiannb" in name or "naivebayes" in name:
        coarse_space = {
            "var_smoothing": {"type": "log_float", "low": 1e-11, "high": 1e-2}
        }
        fine_space = {
            "var_smoothing": {"type": "log_float", "low": 1e-10, "high": 1e-4}
        }
        notes = "GaussianNB: variance smoothing portion added to variances for calculation stability."

    # 10. ElasticNet / Lasso
    elif "elasticnet" in name:
        coarse_space = {
            "alpha": {"type": "log_float", "low": 1e-4, "high": 1e2},
            "l1_ratio": {"type": "float", "low": 0.1, "high": 0.9}
        }
        fine_space = {
            "alpha": {"type": "log_float", "low": 1e-3, "high": 10.0},
            "l1_ratio": {"type": "float", "low": 0.2, "high": 0.8}
        }
        notes = "ElasticNet: regularizer penalty alpha and mixing parameter l1_ratio."

    elif "lasso" in name:
        coarse_space = {
            "alpha": {"type": "log_float", "low": 1e-4, "high": 1e2}
        }
        fine_space = {
            "alpha": {"type": "log_float", "low": 1e-3, "high": 10.0}
        }
        notes = "Lasso L1 regularizer penalty alpha."

    # 11. Huber Regressor
    elif "huber" in name:
        coarse_space = {
            "epsilon": {"type": "float", "low": 1.1, "high": 2.0},
            "alpha": {"type": "log_float", "low": 1e-5, "high": 1e1}
        }
        fine_space = {
            "epsilon": {"type": "float", "low": 1.2, "high": 1.8},
            "alpha": {"type": "log_float", "low": 1e-4, "high": 1.0}
        }
        notes = "HuberRegressor: outlier threshold epsilon and L2 regularization alpha."

    # 12. Bayesian Ridge
    elif "bayesianridge" in name:
        coarse_space = {
            "alpha_1": {"type": "log_float", "low": 1e-8, "high": 1e-2},
            "lambda_1": {"type": "log_float", "low": 1e-8, "high": 1e-2}
        }
        fine_space = {
            "alpha_1": {"type": "log_float", "low": 1e-7, "high": 1e-4},
            "lambda_1": {"type": "log_float", "low": 1e-7, "high": 1e-4}
        }
        notes = "BayesianRidge: shape parameters for Gamma prior over alpha and lambda."

    # 13. Stochastic Gradient Descent (SGD)
    elif "sgd" in name:
        coarse_space = {
            "alpha": {"type": "log_float", "low": 1e-5, "high": 1e-1},
            "penalty": {"type": "categorical", "choices": ["l2", "l1", "elasticnet"]}
        }
        fine_space = {
            "alpha": {"type": "log_float", "low": 1e-4, "high": 1e-2}
        }
        notes = "SGD linear estimator: regularization strength alpha and penalty term."

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
