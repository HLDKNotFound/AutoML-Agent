"""
Automated Tabular Feature Engineering Module.
Extracts high-impact domain and interaction features for tabular classification and regression.
Conforms strictly to scikit-learn BaseEstimator and TransformerMixin for zero-leakage pipeline integration.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TabularFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automated feature engineer that discovers and creates high-leverage tabular features:
    1. Delimited string decomposition (e.g. 'Cabin' -> deck, num, side).
    2. Group identifier & size features (e.g. 'PassengerId' -> group_size, is_alone).
    3. Additive numerical aggregations (e.g. TotalSF, TotalBath, TotalExpenses).
    4. Age and temporal span features (e.g. YrSold - YearBuilt).
    5. Presence indicator flags (e.g. HasPool, HasGarage, HasExpenses).
    """

    def __init__(self):
        self.slash_delimited_cols: List[str] = []
        self.grouped_id_cols: List[str] = []
        self.group_frequencies_: Dict[str, Dict[str, int]] = {}
        self.expense_cols_: List[str] = []
        self.house_sf_cols_: List[str] = []
        self.bath_cols_: Dict[str, str] = {}
        self.age_cols_: Dict[str, str] = {}
        self.porch_cols_: List[str] = []
        self.presence_cols_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.slash_delimited_cols = []
        self.grouped_id_cols = []
        self.group_frequencies_ = {}

        cols = list(X.columns)

        # 1. Delimited column detection (e.g. Cabin = 'B/0/P')
        for col in cols:
            if X[col].dtype == object or str(X[col].dtype) == "string":
                sample = X[col].dropna().astype(str).head(100)
                if len(sample) > 10 and (sample.str.count("/") >= 1).mean() > 0.6:
                    self.slash_delimited_cols.append(col)

        # 2. Group ID detection (e.g. '0001_01')
        for col in cols:
            if X[col].dtype == object or str(X[col].dtype) == "string":
                sample = X[col].dropna().astype(str).head(100)
                if len(sample) > 10 and sample.str.contains(r"^\w+_\d+$").mean() > 0.7:
                    self.grouped_id_cols.append(col)
                    group_ids = X[col].dropna().astype(str).str.split("_").str[0]
                    self.group_frequencies_[col] = group_ids.value_counts().to_dict()

        # 3. Specific high-impact domain aggregations
        # A. Luxury Expenses (Spaceship Titanic / Churn / Travel)
        exp_candidates = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        self.expense_cols_ = [c for c in exp_candidates if c in cols]

        # B. Housing Total Area (House Prices)
        sf_candidates = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]
        if all(c in cols for c in sf_candidates):
            self.house_sf_cols_ = sf_candidates
        elif "GrLivArea" in cols and "TotalBsmtSF" in cols:
            self.house_sf_cols_ = ["GrLivArea", "TotalBsmtSF"]
        else:
            self.house_sf_cols_ = []

        # C. Housing Total Bathrooms
        bath_map = {}
        for b_cand in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
            if b_cand in cols:
                bath_map[b_cand] = b_cand
        self.bath_cols_ = bath_map

        # D. Age calculations
        if "YrSold" in cols and "YearBuilt" in cols:
            self.age_cols_["HouseAge"] = ("YrSold", "YearBuilt")
        if "YrSold" in cols and "YearRemodAdd" in cols:
            self.age_cols_["RemodAge"] = ("YrSold", "YearRemodAdd")

        # E. Porch aggregations
        porch_cands = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        self.porch_cols_ = [c for c in porch_cands if c in cols]

        # F. Presence flags
        pres_cands = ["PoolArea", "GarageArea", "TotalBsmtSF", "Fireplaces", "2ndFlrSF", "WoodDeckSF"]
        self.presence_cols_ = [c for c in pres_cands if c in cols]

        self.is_fitted_ = True
        return self

    def transform(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # 1. Delimited string decomposition
        for col in self.slash_delimited_cols:
            if col in X_out.columns:
                parts = X_out[col].astype(str).str.split("/", expand=True)
                if parts.shape[1] >= 2:
                    X_out[f"{col}_part0"] = parts[0].replace("nan", np.nan).replace("None", np.nan)
                    X_out[f"{col}_part1"] = pd.to_numeric(parts[1], errors="coerce")
                if parts.shape[1] >= 3:
                    X_out[f"{col}_part2"] = parts[2].replace("nan", np.nan).replace("None", np.nan)

        # 2. Group ID features
        for col in self.grouped_id_cols:
            if col in X_out.columns:
                group_prefix = X_out[col].astype(str).str.split("_").str[0]
                freq_map = self.group_frequencies_.get(col, {})
                group_size = group_prefix.map(freq_map).fillna(1.0).astype(float)
                X_out[f"{col}_group_size"] = group_size
                X_out[f"{col}_is_alone"] = (group_size == 1.0).astype(float)

        # 3. Additive aggregations
        if len(self.expense_cols_) >= 2:
            exp_sum = pd.Series(0.0, index=X_out.index)
            for c in self.expense_cols_:
                exp_sum += pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)
            X_out["TotalExpenses"] = exp_sum
            X_out["HasExpenses"] = (exp_sum > 0.0).astype(float)

        if self.house_sf_cols_:
            sf_sum = pd.Series(0.0, index=X_out.index)
            for c in self.house_sf_cols_:
                sf_sum += pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)
            X_out["TotalSF"] = sf_sum

        if self.bath_cols_:
            bath_total = pd.Series(0.0, index=X_out.index)
            if "FullBath" in self.bath_cols_:
                bath_total += pd.to_numeric(X_out["FullBath"], errors="coerce").fillna(0.0)
            if "HalfBath" in self.bath_cols_:
                bath_total += 0.5 * pd.to_numeric(X_out["HalfBath"], errors="coerce").fillna(0.0)
            if "BsmtFullBath" in self.bath_cols_:
                bath_total += pd.to_numeric(X_out["BsmtFullBath"], errors="coerce").fillna(0.0)
            if "BsmtHalfBath" in self.bath_cols_:
                bath_total += 0.5 * pd.to_numeric(X_out["BsmtHalfBath"], errors="coerce").fillna(0.0)
            X_out["TotalBath"] = bath_total

        for out_name, (col1, col2) in self.age_cols_.items():
            if col1 in X_out.columns and col2 in X_out.columns:
                c1 = pd.to_numeric(X_out[col1], errors="coerce")
                c2 = pd.to_numeric(X_out[col2], errors="coerce")
                diff = c1 - c2
                X_out[out_name] = diff.clip(lower=0.0)

        if len(self.porch_cols_) >= 2:
            p_sum = pd.Series(0.0, index=X_out.index)
            for c in self.porch_cols_:
                p_sum += pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)
            X_out["TotalPorchSF"] = p_sum

        for c in self.presence_cols_:
            if c in X_out.columns:
                val = pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)
                X_out[f"Has_{c}"] = (val > 0.0).astype(float)

        return X_out
