"""
Robust Dataset Splitting Module for AutoML-Agent.
Supports stratified, random, group-aware, and chronological splitting strategies.
Handles rare classes, tiny datasets, missing targets, constant targets, and infeasible stratification.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split, GroupShuffleSplit

logger = logging.getLogger(__name__)


class SplitConfig(BaseModel):
    train_ratio: float = Field(default=0.70, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.5)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.5)
    random_state: int = Field(default=42)
    split_policy: str = Field(default="auto", description="'auto', 'stratified', 'random', 'group', or 'chronological'")
    group_column: Optional[str] = Field(default=None, description="Entity/cluster column for group-aware isolation")
    time_column: Optional[str] = Field(default=None, description="Datetime or ordered column for chronological splitting")


def split_dataset_robustly(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    config: Optional[SplitConfig] = None
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    """
    Executes rigorous 3-way partition (Train, Validation, Test).
    Guarantees no data leakage across partitions, audits distribution integrity,
    and returns (train_indices, val_indices, test_indices, split_metadata).
    """
    if config is None:
        config = SplitConfig()

    n_samples = len(df)
    if n_samples < 6:
        raise ValueError(f"Dataset has only {n_samples} samples; at least 6 samples required for a valid 3-way partition.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

    y = df[target_column]
    if y.isna().any():
        raise ValueError(f"Target column '{target_column}' contains {int(y.isna().sum())} missing values. Drop or clean targets before partitioning.")

    n_unique_target = y.nunique()
    if n_unique_target <= 1:
        raise ValueError(f"Target column '{target_column}' has only {n_unique_target} unique value. Supervised training is impossible.")

    indices = np.arange(n_samples)
    warnings: List[str] = []
    policy_used = config.split_policy
    is_classification = "classification" in problem_type

    # 1. Chronological Splitting Strategy
    if config.split_policy == "chronological" or config.time_column:
        policy_used = "chronological"
        sort_col = config.time_column
        if sort_col and sort_col in df.columns:
            sorted_indices = df[sort_col].sort_values().index.to_numpy()
        else:
            sorted_indices = indices

        n_train = int(np.floor(config.train_ratio * n_samples))
        n_val = int(np.floor(config.val_ratio * n_samples))
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        
        train_idx = sorted_indices[:n_train]
        val_idx = sorted_indices[n_train:n_train + n_val]
        test_idx = sorted_indices[n_train + n_val:]
        if len(test_idx) == 0:
            test_idx = val_idx[-1:]
            val_idx = val_idx[:-1]

    # 2. Group-Aware Splitting Strategy
    elif (config.split_policy == "group" or config.group_column) and config.group_column in df.columns:
        policy_used = "group"
        groups = df[config.group_column].values
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < 3:
            warnings.append(f"Only {n_unique_groups} unique groups available. Falling back to stratified/random split.")
            policy_used = "stratified" if is_classification else "random"
        else:
            # 70% Train groups, 30% Temp groups
            gss1 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - config.train_ratio), random_state=config.random_state)
            train_idx, temp_idx = next(gss1.split(df, y, groups))

            # 15% Val groups, 15% Test groups (50/50 of temp)
            temp_df = df.iloc[temp_idx]
            temp_groups = temp_df[config.group_column].values
            if len(np.unique(temp_groups)) < 2:
                # Fallback if temp groups cannot be split
                mid = len(temp_idx) // 2
                val_idx = temp_idx[:mid]
                test_idx = temp_idx[mid:]
            else:
                gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=config.random_state)
                val_rel, test_rel = next(gss2.split(temp_df, temp_df[target_column], temp_groups))
                val_idx = temp_idx[val_rel]
                test_idx = temp_idx[test_rel]

    # 3. Stratified Splitting for Classification (with Rare Class Fallback)
    if policy_used in ("auto", "stratified") and is_classification:
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())

        # Stratified 3-way partition requires at least 3 instances per class,
        # and temp partition (30%) must have at least 2 instances of each class
        # to stratify into val and test.
        can_stratify = min_class_count >= 4 and len(class_counts) < n_samples * 0.4
        
        if can_stratify:
            try:
                # 70% Train, 30% Temp
                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=(1.0 - config.train_ratio),
                    random_state=config.random_state,
                    stratify=y
                )
                
                # Check temp partition class distribution
                temp_y = y.iloc[temp_idx]
                temp_counts = temp_y.value_counts()
                if (temp_counts >= 2).all():
                    val_idx, test_idx = train_test_split(
                        temp_idx,
                        test_size=0.50,
                        random_state=config.random_state,
                        stratify=temp_y
                    )
                else:
                    warnings.append("Rare classes in held-out partition prevented secondary stratification. Applied unstratified split on validation/test.")
                    val_idx, test_idx = train_test_split(
                        temp_idx,
                        test_size=0.50,
                        random_state=config.random_state,
                        stratify=None
                    )
                policy_used = "stratified"
            except Exception as e:
                warnings.append(f"Stratified split failed ({e}). Falling back to randomized partition.")
                policy_used = "random"

        else:
            warnings.append(
                f"Stratification infeasible: least populated class has only {min_class_count} sample(s). "
                f"Applied randomized partition to protect pipeline execution."
            )
            policy_used = "random"

    # 4. Standard Randomized Splitting (Regression or Classification Fallback)
    if policy_used == "random" or (policy_used == "auto" and not is_classification):
        policy_used = "random"
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(1.0 - config.train_ratio),
            random_state=config.random_state,
            shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=config.random_state,
            shuffle=True
        )

    # Convert to pure Python lists
    train_list = train_idx.tolist() if isinstance(train_idx, np.ndarray) else list(train_idx)
    val_list = val_idx.tolist() if isinstance(val_idx, np.ndarray) else list(val_idx)
    test_list = test_idx.tolist() if isinstance(test_idx, np.ndarray) else list(test_idx)

    # Post-split verification: ensure no overlap and valid sample counts
    s_train, s_val, s_test = set(train_list), set(val_list), set(test_list)
    assert len(s_train.intersection(s_val)) == 0, "Data leakage detected: train and validation sets overlap!"
    assert len(s_train.intersection(s_test)) == 0, "Data leakage detected: train and test sets overlap!"
    assert len(s_val.intersection(s_test)) == 0, "Data leakage detected: validation and test sets overlap!"

    split_metadata = {
        "policy_used": policy_used,
        "random_state": config.random_state,
        "sample_counts": {
            "total": n_samples,
            "train": len(train_list),
            "validation": len(val_list),
            "test": len(test_list)
        },
        "proportions": {
            "train": round(len(train_list) / n_samples, 4),
            "validation": round(len(val_list) / n_samples, 4),
            "test": round(len(test_list) / n_samples, 4)
        },
        "warnings": warnings
    }

    if is_classification:
        split_metadata["class_distributions"] = {
            "train": {str(k): int(v) for k, v in y.iloc[train_list].value_counts().items()},
            "validation": {str(k): int(v) for k, v in y.iloc[val_list].value_counts().items()},
            "test": {str(k): int(v) for k, v in y.iloc[test_list].value_counts().items()}
        }

    return train_list, val_list, test_list, split_metadata
