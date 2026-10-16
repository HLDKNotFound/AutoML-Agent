"""
Missing value handling templates.
"""

from sklearn.impute import SimpleImputer, KNNImputer


def create_imputer(data_type: str = "numeric", strategy: str = "median", add_indicator: bool = False):
    """
    Returns an appropriate scikit-learn imputer based on feature type and desired strategy.
    Supports:
      - Numeric: median, mean, most_frequent, knn, iterative (MICE)
      - Categorical: most_frequent, constant
    """
    strategy = (strategy or "median").lower()
    if data_type == "numeric":
        if strategy == "knn":
            return KNNImputer(n_neighbors=5, add_indicator=add_indicator)
        elif strategy in ("iterative", "mice"):
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            return IterativeImputer(random_state=42, max_iter=10, add_indicator=add_indicator)
        elif strategy in ("mean", "median", "most_frequent"):
            return SimpleImputer(strategy=strategy, add_indicator=add_indicator)
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    else:
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value="missing")
        return SimpleImputer(strategy="most_frequent")
