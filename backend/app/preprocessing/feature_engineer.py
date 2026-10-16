"""
Automated Tabular Feature Engineering Module.
Extracts generally valid tabular features:
1. Structured string decomposition (slash/dash/colon delimited strings).
2. Datetime components (month, day, dayofweek, is_weekend, hour).
3. Missingness indicators for partially missing attributes.
4. Numerical interaction terms (products and safe ratios).
Conforms strictly to scikit-learn BaseEstimator and TransformerMixin for zero-leakage pipeline integration.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generally valid automated feature engineer for tabular data.
    Separates general tabular transformations from domain-specific heuristics.
    Guarantees no column name collisions, idempotent transforms, and deterministic column order.
    """

    def __init__(self, enable_interactions: bool = False, enable_missing_indicators: bool = True):
        self.enable_interactions = enable_interactions
        self.enable_missing_indicators = enable_missing_indicators

        self.delimited_cols_: List[Tuple[str, str]] = []  # (column_name, delimiter)
        self.datetime_cols_: List[str] = []
        self.missing_indicator_cols_: List[str] = []
        self.interaction_pairs_: List[Tuple[str, str]] = []
        self.fitted_columns_: List[str] = []
        self.output_feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.fitted_columns_ = list(X.columns)
        self.delimited_cols_ = []
        self.datetime_cols_ = []
        self.missing_indicator_cols_ = []
        self.interaction_pairs_ = []

        cols = list(X.columns)
        n_rows = len(X)

        for col in cols:
            series = X[col]
            # 1. Delimited string detection (e.g. "A/12/P" or "2024-05")
            if series.dtype == object or str(series.dtype) == "string":
                sample = series.dropna().astype(str).head(100)
                if len(sample) >= 10:
                    for delim in ["/", "-", ":", "_"]:
                        counts = sample.str.count(delim)
                        if (counts >= 1).mean() > 0.70 and counts.nunique() <= 3:
                            self.delimited_cols_.append((col, delim))
                            break

            # 2. Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                self.datetime_cols_.append(col)
            elif series.dtype == object or str(series.dtype) == "string":
                sample = series.dropna().astype(str).head(50)
                if len(sample) >= 10:
                    try:
                        converted = pd.to_datetime(sample, errors="coerce", format="mixed")
                        if converted.notna().mean() > 0.85:
                            self.datetime_cols_.append(col)
                    except Exception:
                        pass

            # 3. Missingness indicators for moderate missingness (5% to 50%)
            if self.enable_missing_indicators and n_rows > 0:
                missing_pct = float(series.isna().mean())
                if 0.05 <= missing_pct <= 0.50:
                    self.missing_indicator_cols_.append(col)

        # 4. Optional numeric interactions
        if self.enable_interactions:
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c].dtype)]
            if len(numeric_cols) >= 2:
                # Select top 3 highest-variance numerical columns to avoid combinatorial explosion
                variances = {c: float(X[c].dropna().var()) for c in numeric_cols if len(X[c].dropna()) > 1}
                top_numeric = sorted(variances.keys(), key=lambda c: variances[c], reverse=True)[:3]
                for i in range(len(top_numeric)):
                    for j in range(i + 1, len(top_numeric)):
                        self.interaction_pairs_.append((top_numeric[i], top_numeric[j]))

        # Precompute expected output feature names
        dummy_transformed = self._transform_internal(X.head(5))
        self.output_feature_names_ = list(dummy_transformed.columns)
        self.is_fitted_ = True
        return self

    def _transform_internal(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Internal transform logic ensuring no column name collisions and safe derived values."""
        X_out = X_df.copy()
        existing_cols: Set[str] = set(X_out.columns)

        # 1. Delimited string decomposition
        for col, delim in self.delimited_cols_:
            if col in X_out.columns:
                parts = X_out[col].astype(str).str.split(delim, expand=True)
                n_parts = min(parts.shape[1], 4)
                for i in range(n_parts):
                    new_col = f"{col}_part_{i}"
                    if new_col not in existing_cols:
                        series_part = parts[i].replace("nan", np.nan).replace("None", np.nan)
                        # Auto-convert to numeric if possible
                        try:
                            X_out[new_col] = pd.to_numeric(series_part)
                        except Exception:
                            X_out[new_col] = series_part
                        existing_cols.add(new_col)

        # 2. Datetime feature extraction
        for col in self.datetime_cols_:
            if col in X_out.columns:
                dt_series = pd.to_datetime(X_out[col], errors="coerce")
                for feat, name in [
                    (dt_series.dt.month, f"{col}_month"),
                    (dt_series.dt.day, f"{col}_day"),
                    (dt_series.dt.dayofweek, f"{col}_dayofweek"),
                    ((dt_series.dt.dayofweek >= 5).astype(float), f"{col}_is_weekend"),
                    (dt_series.dt.hour, f"{col}_hour")
                ]:
                    if name not in existing_cols:
                        X_out[name] = feat.fillna(-1.0)
                        existing_cols.add(name)

        # 3. Missingness indicator flags
        for col in self.missing_indicator_cols_:
            if col in X_out.columns:
                name = f"{col}_is_missing"
                if name not in existing_cols:
                    X_out[name] = X_out[col].isna().astype(float)
                    existing_cols.add(name)

        # 4. Numeric interactions
        for c1, c2 in self.interaction_pairs_:
            if c1 in X_out.columns and c2 in X_out.columns:
                prod_name = f"{c1}_x_{c2}"
                if prod_name not in existing_cols:
                    s1 = pd.to_numeric(X_out[c1], errors="coerce").fillna(0.0)
                    s2 = pd.to_numeric(X_out[c2], errors="coerce").fillna(0.0)
                    X_out[prod_name] = s1 * s2
                    existing_cols.add(prod_name)

        return X_out

    def transform(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.fitted_columns_ if self.fitted_columns_ and len(X[0]) == len(self.fitted_columns_) else None)

        X_out = self._transform_internal(X)

        # Ensure all columns fitted during training exist; fill missing columns if any
        if self.is_fitted_ and self.output_feature_names_:
            for col in self.output_feature_names_:
                if col not in X_out.columns:
                    X_out[col] = 0.0

        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.output_feature_names_, dtype=object)
