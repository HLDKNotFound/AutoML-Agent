"""
Dataset Profiling Service.
Computes comprehensive dataset-level, feature-level, and target-level statistics.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_dataset_profile(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyzes the dataframe and produces detailed structured statistics for profiling and LLM analysis.
    """
    total_rows, total_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())
    total_cells = total_rows * total_cols
    missing_ratio = float(total_missing / total_cells) if total_cells > 0 else 0.0
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())

    if feature_columns is None:
        if target_column and target_column in df.columns:
            feature_columns = [c for c in df.columns if c != target_column]
        else:
            feature_columns = list(df.columns)

    # Classify feature types
    numeric_features = []
    categorical_features = []
    datetime_features = []

    for col in feature_columns:
        if col not in df.columns:
            continue
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            numeric_features.append(col)
        elif pd.api.types.is_datetime64_any_dtype(col_type):
            datetime_features.append(col)
        else:
            categorical_features.append(col)

    # Numeric feature statistics
    num_stats = {}
    for col in numeric_features:
        series = df[col].dropna()
        n_unique = int(df[col].nunique())
        n_missing = int(df[col].isna().sum())
        missing_pct = float(n_missing / total_rows) if total_rows > 0 else 0.0

        if len(series) > 0:
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            var_val = float(series.var(ddof=1)) if len(series) > 1 else 0.0
            min_val = float(series.min())
            max_val = float(series.max())
            q25 = float(series.quantile(0.25))
            q50 = float(series.quantile(0.50))
            q75 = float(series.quantile(0.75))
            skew_val = float(series.skew()) if len(series) > 2 else 0.0

            # Outlier detection via IQR
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            n_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
            outlier_pct = float(n_outliers / len(series)) if len(series) > 0 else 0.0
        else:
            mean_val = median_val = std_val = var_val = min_val = max_val = q25 = q50 = q75 = skew_val = 0.0
            n_outliers = 0
            outlier_pct = 0.0

        num_stats[col] = {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "variance": var_val,
            "min": min_val,
            "max": max_val,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "skewness": skew_val,
            "unique_values": n_unique,
            "missing_count": n_missing,
            "missing_pct": missing_pct,
            "outliers_count": n_outliers,
            "outliers_pct": outlier_pct,
            "is_constant": bool(n_unique <= 1),
            "is_id_candidate": bool(n_unique == total_rows and total_rows > 50)
        }

    # Categorical feature statistics
    cat_stats = {}
    for col in categorical_features:
        series = df[col].dropna().astype(str)
        n_unique = int(df[col].nunique())
        n_missing = int(df[col].isna().sum())
        missing_pct = float(n_missing / total_rows) if total_rows > 0 else 0.0

        val_counts = series.value_counts(normalize=True)
        top_category = str(val_counts.index[0]) if not val_counts.empty else None
        top_freq = float(val_counts.iloc[0]) if not val_counts.empty else 0.0

        # Rare categories (< 1% frequency)
        rare_categories = [str(cat) for cat, freq in val_counts.items() if freq < 0.01][:10]

        # Top 10 frequency map for preview
        top_10_freq = {str(k): round(float(v), 4) for k, v in val_counts.head(10).items()}

        cat_stats[col] = {
            "unique_categories": n_unique,
            "top_category": top_category,
            "top_frequency": top_freq,
            "top_frequencies": top_10_freq,
            "rare_categories_sample": rare_categories,
            "missing_count": n_missing,
            "missing_pct": missing_pct,
            "high_cardinality": bool(n_unique > 50 and n_unique > 0.3 * total_rows),
            "is_constant": bool(n_unique <= 1)
        }

    # Correlation Matrix and Collinear Pairs (> 0.85)
    high_correlation_pairs = []
    correlation_matrix = {}
    if len(numeric_features) >= 2:
        try:
            corr_df = df[numeric_features].corr().fillna(0.0)
            correlation_matrix = {col: {k: round(float(v), 4) for k, v in row.items()} for col, row in corr_df.to_dict().items()}
            
            cols = corr_df.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1, c2 = cols[i], cols[j]
                    val = abs(corr_df.loc[c1, c2])
                    if val >= 0.85:
                        high_correlation_pairs.append({
                            "feature1": c1,
                            "feature2": c2,
                            "correlation": round(float(corr_df.loc[c1, c2]), 4)
                        })
        except Exception:
            pass

    # Target Analysis
    target_stats = {}
    problem_type_guess = "binary_classification"
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        n_unique_target = int(target_series.nunique())
        
        # Determine if target is numeric and continuous
        is_num = pd.api.types.is_numeric_dtype(target_series)
        
        if is_num and n_unique_target > 20:
            problem_type_guess = "regression"
            target_stats = {
                "type": "regression",
                "mean": float(target_series.mean()),
                "median": float(target_series.median()),
                "std": float(target_series.std(ddof=1)) if len(target_series) > 1 else 0.0,
                "variance": float(target_series.var(ddof=1)) if len(target_series) > 1 else 0.0,
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "skewness": float(target_series.skew()) if len(target_series) > 2 else 0.0,
                "missing_count": int(df[target_column].isna().sum())
            }
        else:
            val_counts = target_series.value_counts()
            class_dist = {str(k): int(v) for k, v in val_counts.items()}
            class_pcts = {str(k): round(float(v / len(target_series)), 4) for k, v in val_counts.items()}
            
            # Imbalance ratio: min_class / max_class
            imbalance_ratio = float(val_counts.min() / val_counts.max()) if val_counts.max() > 0 else 1.0
            
            if n_unique_target == 2:
                problem_type_guess = "binary_classification"
            else:
                problem_type_guess = "multiclass_classification"
                
            target_stats = {
                "type": problem_type_guess,
                "num_classes": n_unique_target,
                "class_counts": class_dist,
                "class_proportions": class_pcts,
                "imbalance_ratio": round(imbalance_ratio, 4),
                "is_imbalanced": bool(imbalance_ratio < 0.25),
                "missing_count": int(df[target_column].isna().sum())
            }

        # Check for potential target leakage in features
        potential_leakage = []
        for col in numeric_features:
            if col != target_column:
                try:
                    target_numeric = pd.to_numeric(target_series, errors="coerce")
                    feat_numeric = pd.to_numeric(df[col], errors="coerce")
                    valid_mask = target_numeric.notna() & feat_numeric.notna()
                    if valid_mask.sum() > 10:
                        corr_with_target = abs(float(np.corrcoef(feat_numeric[valid_mask], target_numeric[valid_mask])[0, 1]))
                        if corr_with_target > 0.96:
                            potential_leakage.append({
                                "feature": col,
                                "correlation_with_target": round(corr_with_target, 4),
                                "reason": "Extremely high linear correlation with target"
                            })
                except Exception:
                    pass

        # String identity match leakage
        for col in categorical_features:
            if col != target_column:
                if df[col].astype(str).equals(df[target_column].astype(str)):
                    potential_leakage.append({
                        "feature": col,
                        "correlation_with_target": 1.0,
                        "reason": "Exact identical values to target"
                    })
        target_stats["potential_leakage"] = potential_leakage

    return {
        "dataset": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": duplicate_rows,
            "missing_cells": total_missing,
            "missing_ratio": round(missing_ratio, 4),
            "memory_usage_kb": round(memory_usage_bytes / 1024, 2),
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "datetime_feature_count": len(datetime_features)
        },
        "feature_types": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "datetime": datetime_features
        },
        "numerical_stats": num_stats,
        "categorical_stats": cat_stats,
        "high_correlation_pairs": high_correlation_pairs,
        "correlation_matrix": correlation_matrix,
        "target_stats": target_stats,
        "suggested_problem_type": problem_type_guess
    }
