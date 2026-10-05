"""
Categorical preprocessing templates conforming to scikit-learn official documentation.
"""

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


def get_categorical_encoder(encoder_type: str = "one_hot", handle_unknown: str = "ignore"):
    """
    Returns an un-fitted scikit-learn transformer for categorical encoding.
    Ensures safe handling of unknown categories during inference.
    """
    encoder_type = (encoder_type or "one_hot").lower()
    if encoder_type == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoder_type == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def get_categorical_imputer(strategy: str = "most_frequent", fill_value: str = "missing"):
    """
    Returns an un-fitted scikit-learn imputer for categorical values.
    """
    strategy = (strategy or "most_frequent").lower()
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=fill_value)
    return SimpleImputer(strategy="most_frequent")
