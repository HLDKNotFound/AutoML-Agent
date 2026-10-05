"""
Model-Specific Preprocessing Pipeline Builder.
Assembles tailored scikit-learn ColumnTransformer and Pipeline instances for each model.
Guarantees zero data leakage by ensuring all fitting occurs strictly on training partitions.
"""

from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from backend.templates.numerical.scaling import get_numeric_scaler, get_numeric_imputer
from backend.templates.categorical.encoding import get_categorical_encoder, get_categorical_imputer
from backend.app.schemas.ml_schemas import PreprocessingStrategy


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    strategy: Optional[PreprocessingStrategy] = None
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer tailored to the specific model's requirements.
    
    Tree-based models: Typically need imputation, but NOT scaling; can use one-hot or ordinal encoding.
    Linear / Kernel / Distance / MLP models: Require robust/standard scaling and one-hot encoding.
    """
    if strategy is None:
        strategy = PreprocessingStrategy()

    transformers = []

    # Numeric Pipeline
    if numeric_features:
        num_steps = [
            ("num_imputer", get_numeric_imputer(strategy.numeric_imputer))
        ]
        scaler = get_numeric_scaler(strategy.numeric_scaler)
        if scaler != "passthrough":
            num_steps.append(("num_scaler", scaler))
            
        numeric_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    # Categorical Pipeline
    if categorical_features:
        cat_steps = [
            ("cat_imputer", get_categorical_imputer(strategy.categorical_imputer)),
            ("cat_encoder", get_categorical_encoder(strategy.categorical_encoder, strategy.handle_unknown))
        ]
        categorical_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    return preprocessor


def build_full_model_pipeline(
    model_estimator,
    numeric_features: List[str],
    categorical_features: List[str],
    strategy: Optional[PreprocessingStrategy] = None
) -> Pipeline:
    """
    Returns an end-to-end Pipeline: ColumnTransformer -> Estimator.
    This guarantees atomic fit, transform, and predict operations.
    """
    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        strategy=strategy
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_estimator)
    ])
    
    return pipeline
