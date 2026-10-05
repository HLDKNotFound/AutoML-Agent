"""
Numerical preprocessing templates conforming to scikit-learn official documentation.
"""

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, MaxAbsScaler
from sklearn.impute import SimpleImputer, KNNImputer


def get_numeric_scaler(scaler_type: str = "robust"):
    """
    Returns an un-fitted scikit-learn transformer for numeric scaling.
    Supports robust, standard, minmax, quantile, maxabs, or None.
    """
    scaler_type = (scaler_type or "").lower()
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    elif scaler_type == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "maxabs":
        return MaxAbsScaler()
    elif scaler_type == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    elif scaler_type in ("none", "passthrough", ""):
        return "passthrough"
    else:
        # Default fallback
        return RobustScaler()


def get_numeric_imputer(strategy: str = "median"):
    """
    Returns an un-fitted scikit-learn imputer for numeric values.
    Supports median, mean, most_frequent, or knn.
    """
    strategy = (strategy or "median").lower()
    if strategy in ("median", "mean", "most_frequent"):
        return SimpleImputer(strategy=strategy)
    elif strategy == "knn":
        return KNNImputer(n_neighbors=5)
    return SimpleImputer(strategy="median")
