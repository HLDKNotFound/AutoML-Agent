"""
Missing value handling templates.
"""

from sklearn.impute import SimpleImputer, KNNImputer


def create_imputer(data_type: str = "numeric", strategy: str = "median"):
    """
    Returns an appropriate scikit-learn imputer based on feature type and desired strategy.
    """
    if data_type == "numeric":
        if strategy == "knn":
            return KNNImputer(n_neighbors=5)
        elif strategy in ("mean", "median", "most_frequent"):
            return SimpleImputer(strategy=strategy)
        return SimpleImputer(strategy="median")
    else:
        if strategy == "constant":
            return SimpleImputer(strategy="constant", fill_value="missing")
        return SimpleImputer(strategy="most_frequent")
