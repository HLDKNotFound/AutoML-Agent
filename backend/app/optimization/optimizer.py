"""
Coarse-to-Fine Hyperparameter Optimization Engine.
Stage 1: Coarse broad search across logarithmic, uniform, and discrete parameter scales.
Stage 2: Fine search using Bayesian Optimization (Optuna TPE) explicitly refining promising regions.
Evaluates strictly on validation data to select optimal hyperparameters.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from backend.app.models.model_factory import create_base_estimator
from backend.app.preprocessing.pipeline_builder import build_full_model_pipeline
from backend.app.evaluation.evaluator import evaluate_pipeline
from backend.app.evaluation.metric_registry import get_optimization_score, is_metric_better
from backend.app.schemas.ml_schemas import PreprocessingStrategy, HyperparameterSearchPlan
from backend.app.core.config import config

logger = logging.getLogger(__name__)


def run_coarse_search(
    model_name: str,
    problem_type: str,
    search_plan: HyperparameterSearchPlan,
    preprocessing_strategy: PreprocessingStrategy,
    numeric_features: List[str],
    categorical_features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_metric: str,
    n_trials: int = 8,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    transform_target_log: bool = False,
    feature_engineer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Stage 1: Coarse Broad Search.
    Explores broad distributions across log-uniform, uniform, and discrete bounds.
    Distinguishes completed, failed, and pruned trials with explicit error tracking.
    """
    coarse_space = search_plan.coarse_space
    trials_history = []
    best_score = -float("inf")
    best_metric_val = 0.0
    best_params = {}
    best_pipeline = None

    sampler = optuna.samplers.RandomSampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial):
        nonlocal best_score, best_metric_val, best_params, best_pipeline
        params = {}
        for param_name, spec in coarse_space.items():
            p_type = spec.get("type")
            if p_type == "int":
                step = spec.get("step", 1)
                params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=step)
            elif p_type == "float":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"])
            elif p_type == "log_float":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=True)
            elif p_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, spec["choices"])

        status = "completed"
        error_msg = None
        fit_time = 0.0
        score = -1e9
        metric_val = 0.0
        pipeline = None

        try:
            estimator = create_base_estimator(model_name, problem_type, params)
            pipeline = build_full_model_pipeline(
                estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                strategy=preprocessing_strategy,
                transform_target_log=transform_target_log,
                feature_engineer=feature_engineer
            )

            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            eval_res = evaluate_pipeline(pipeline, X_val, y_val, problem_type, train_time_sec=fit_time)
            metric_val = float(eval_res["metrics"].get(primary_metric, 0.0))
            score = get_optimization_score(primary_metric, metric_val)
        except Exception as e:
            status = "failed"
            error_msg = f"{type(e).__name__}: {str(e)[:180]}"
            logger.debug("Coarse trial %d failed for %s: %s", trial.number, model_name, error_msg)

        trial_data = {
            "trial_number": trial.number,
            "stage": "coarse",
            "params": params,
            "status": status,
            "score": score if status == "completed" else None,
            "metric_value": metric_val if status == "completed" else None,
            "fit_time": fit_time,
            "error": error_msg
        }
        trials_history.append(trial_data)

        if status == "completed" and score > best_score:
            best_score = score
            best_metric_val = metric_val
            best_params = dict(params)
            best_pipeline = pipeline

        if progress_callback:
            progress_callback({
                "stage": "coarse",
                "trial": trial.number + 1,
                "total": n_trials,
                "current_score": metric_val if status == "completed" else 0.0,
                "best_score": best_metric_val,
                "status": status
            })

        return score

    timeout_sec = min(40.0, float(getattr(config, "max_training_time_sec", 300)))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=timeout_sec)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_pipeline": best_pipeline,
        "trials_history": trials_history
    }


def run_fine_search(
    model_name: str,
    problem_type: str,
    coarse_best_params: Dict[str, Any],
    search_plan: HyperparameterSearchPlan,
    preprocessing_strategy: PreprocessingStrategy,
    numeric_features: List[str],
    categorical_features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_metric: str,
    n_trials: int = 12,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    transform_target_log: bool = False,
    feature_engineer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Stage 2: Fine Search via Bayesian Optimization (Optuna TPESampler).
    Explicitly refines the search space bounds around the Stage 1 winners.
    Tracks trial states and avoids selecting failed trials.
    """
    fine_space = search_plan.fine_space
    trials_history = []
    best_score = -float("inf")
    best_metric_val = 0.0
    best_params = dict(coarse_best_params)
    best_pipeline = None

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed study with best parameters from coarse search clamped to fine bounds
    if coarse_best_params:
        seeded = {}
        for k, v in coarse_best_params.items():
            if k in fine_space:
                spec = fine_space[k]
                p_type = spec.get("type")
                if p_type in ("int", "float", "log_float"):
                    low = spec["low"]
                    high = spec["high"]
                    clamped_v = max(low, min(high, v))
                    if p_type == "int":
                        clamped_v = int(round(clamped_v))
                    seeded[k] = clamped_v
                elif p_type == "categorical":
                    if v in spec["choices"]:
                        seeded[k] = v
        if seeded:
            study.enqueue_trial(seeded)

    def objective(trial: optuna.Trial):
        nonlocal best_score, best_metric_val, best_params, best_pipeline
        params = dict(coarse_best_params)
        
        for param_name, spec in fine_space.items():
            p_type = spec.get("type")
            if p_type == "int":
                step = spec.get("step", 1)
                params[param_name] = trial.suggest_int(param_name, spec["low"], spec["high"], step=step)
            elif p_type == "float":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"])
            elif p_type == "log_float":
                params[param_name] = trial.suggest_float(param_name, spec["low"], spec["high"], log=True)
            elif p_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, spec["choices"])

        status = "completed"
        error_msg = None
        fit_time = 0.0
        score = -1e9
        metric_val = 0.0
        pipeline = None

        try:
            estimator = create_base_estimator(model_name, problem_type, params)
            pipeline = build_full_model_pipeline(
                estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                strategy=preprocessing_strategy,
                transform_target_log=transform_target_log,
                feature_engineer=feature_engineer
            )

            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            eval_res = evaluate_pipeline(pipeline, X_val, y_val, problem_type, train_time_sec=fit_time)
            metric_val = float(eval_res["metrics"].get(primary_metric, 0.0))
            score = get_optimization_score(primary_metric, metric_val)
        except Exception as e:
            status = "failed"
            error_msg = f"{type(e).__name__}: {str(e)[:180]}"
            logger.debug("Fine trial %d failed for %s: %s", trial.number, model_name, error_msg)

        trial_data = {
            "trial_number": trial.number,
            "stage": "fine",
            "params": params,
            "status": status,
            "score": score if status == "completed" else None,
            "metric_value": metric_val if status == "completed" else None,
            "fit_time": fit_time,
            "error": error_msg
        }
        trials_history.append(trial_data)

        if status == "completed" and score > best_score:
            best_score = score
            best_metric_val = metric_val
            best_params = dict(params)
            best_pipeline = pipeline

        if progress_callback:
            progress_callback({
                "stage": "fine",
                "trial": trial.number + 1,
                "total": n_trials,
                "current_score": metric_val if status == "completed" else 0.0,
                "best_score": best_metric_val,
                "status": status
            })

        return score

    timeout_sec = min(40.0, float(getattr(config, "max_training_time_sec", 300)))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=timeout_sec)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_pipeline": best_pipeline,
        "trials_history": trials_history
    }


def optimize_candidate_model(
    model_name: str,
    model_family: str,
    problem_type: str,
    search_plan: HyperparameterSearchPlan,
    preprocessing_strategy: PreprocessingStrategy,
    numeric_features: List[str],
    categorical_features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_metric: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    feature_engineer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end two-stage coarse-to-fine optimization for a single candidate model.
    Handles all-trial-failure gracefully without crashing subsequent nodes.
    """
    t_start = time.perf_counter()

    # Detect right-skewed positive targets in regression (e.g. House Prices) for log1p optimization
    transform_target_log = False
    if problem_type == "regression":
        try:
            from scipy.stats import skew
            clean_y = y_train.dropna()
            if len(clean_y) > 0 and (clean_y > 0).all() and float(skew(clean_y)) > 0.75:
                transform_target_log = True
        except Exception:
            pass

    # Stage 1: Coarse Search
    coarse_res = run_coarse_search(
        model_name=model_name,
        problem_type=problem_type,
        search_plan=search_plan,
        preprocessing_strategy=preprocessing_strategy,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        primary_metric=primary_metric,
        n_trials=config.coarse_search_trials,
        progress_callback=progress_callback,
        transform_target_log=transform_target_log,
        feature_engineer=feature_engineer
    )

    # Stage 2: Fine Search (Bayesian Optimization refining coarse results)
    fine_res = run_fine_search(
        model_name=model_name,
        problem_type=problem_type,
        coarse_best_params=coarse_res["best_params"],
        search_plan=search_plan,
        preprocessing_strategy=preprocessing_strategy,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        primary_metric=primary_metric,
        n_trials=config.fine_search_trials,
        progress_callback=progress_callback,
        transform_target_log=transform_target_log,
        feature_engineer=feature_engineer
    )

    total_time = time.perf_counter() - t_start

    # Final validation evaluation of the optimized pipeline
    best_pipeline = fine_res.get("best_pipeline") or coarse_res.get("best_pipeline")
    best_params = fine_res.get("best_params") or coarse_res.get("best_params") or {}

    # Graceful fallback: If every trial failed during hyperparameter search, fit un-tuned default
    if best_pipeline is None:
        try:
            logger.warning("All hyperparameter trials failed for %s. Attempting default estimator fit.", model_name)
            default_estimator = create_base_estimator(model_name, problem_type, {})
            fallback_pipeline = build_full_model_pipeline(
                default_estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                strategy=preprocessing_strategy,
                transform_target_log=transform_target_log,
                feature_engineer=feature_engineer
            )
            fallback_pipeline.fit(X_train, y_train)
            best_pipeline = fallback_pipeline
            best_params = {"fallback": "default_un-tuned"}
        except Exception as e:
            logger.error("Default fit fallback failed for %s: %s", model_name, e)
            return {
                "model_name": model_name,
                "model_family": model_family,
                "best_pipeline": None,
                "best_params": {},
                "validation_metrics": {},
                "train_time_sec": total_time,
                "inference_time_ms": 0.0,
                "model_size_kb": 0.0,
                "status": "failed",
                "error": f"All trials and default fallback failed: {e}",
                "optimization_history": coarse_res["trials_history"] + fine_res["trials_history"]
            }

    final_val_eval = evaluate_pipeline(
        best_pipeline,
        X_val,
        y_val,
        problem_type,
        train_time_sec=total_time
    )

    return {
        "model_name": model_name,
        "model_family": model_family,
        "best_pipeline": best_pipeline,
        "best_params": best_params,
        "validation_metrics": final_val_eval["metrics"],
        "train_time_sec": total_time,
        "inference_time_ms": final_val_eval["inference_time_ms"],
        "model_size_kb": final_val_eval["model_size_kb"],
        "status": "completed",
        "optimization_history": coarse_res["trials_history"] + fine_res["trials_history"]
    }
