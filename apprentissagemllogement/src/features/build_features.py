# """Feature building helpers."""

# from __future__ import annotations

# from typing import Iterable, Tuple

# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler


# def _get_column_types(df: pd.DataFrame) -> Tuple[list, list]:
#     numerics = df.select_dtypes(include=["number"]).columns.tolist()
#     categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()

#     # Remove Id if present
#     for col in ["Id", "id", "ID"]:
#         numerics = [c for c in numerics if c != col]
#         categoricals = [c for c in categoricals if c != col]

#     return numerics, categoricals


# def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, Iterable[str]]:
#     num_cols, cat_cols = _get_column_types(df)

#     numeric_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#     ])

#     categorical_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
#     ])

#     preprocessor = ColumnTransformer([
#         ("num", numeric_pipeline, num_cols),
#         ("cat", categorical_pipeline, cat_cols),
#     ])

#     # noms bruts (approximatifs)
#     output_columns = num_cols + cat_cols
#     return preprocessor, output_columns


# def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
#     """
#     Prépare les features pour l'entraînement et la prédiction.
#     NE FAIT PLUS AUCUNE TRANSFORMATION ICI.
#     Le Pipeline dans train_model fera la transformation.
#     """
#     # Séparation cible
#     if "SalePrice" not in train_df.columns:
#         raise ValueError("train_df doit contenir la colonne SalePrice")

#     y_train = train_df["SalePrice"].copy()
#     X_train = train_df.drop(columns=["SalePrice"])

#     # Test
#     X_test = test_df.copy()
#     test_ids = X_test["Id"].copy() if "Id" in X_test.columns else None

#     # On enlève "Id" partout
#     for df in [X_train, X_test]:
#         if "Id" in df.columns:
#             df.drop(columns="Id", inplace=True)

#     # Preprocessor basé sur train+test (mêmes catégories)
#     full_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)
#     preprocessor, _ = build_preprocessor(full_df)

#     #   PAS transformer ici
#     return X_train, y_train.values, X_test, preprocessor, test_ids


# __all__ = ["build_preprocessor", "prepare_features"]
"""Feature building helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove Id and target if present
    for col in ["Id", "id", "ID", "SalePrice"]:
        if col in numerics:
            numerics.remove(col)
        if col in categoricals:
            categoricals.remove(col)

    return numerics, categoricals


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, Iterable[str]]:
    num_cols, cat_cols = _get_column_types(df)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])

    # Get output feature names after preprocessing
    preprocessor.fit(df)
    output_columns = preprocessor.get_feature_names_out()
    
    return preprocessor, output_columns


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prépare les features pour l'entraînement et la prédiction.
    Retourne les données brutes mais avec des transformations préparatoires minimales.
    """
    # Séparation cible
    if "SalePrice" not in train_df.columns:
        raise ValueError("train_df doit contenir la colonne SalePrice")

    y_train = train_df["SalePrice"].copy()
    X_train = train_df.drop(columns=["SalePrice"])

    # Test
    X_test = test_df.copy()
    test_ids = X_test["Id"].copy() if "Id" in X_test.columns else None

    # On enlève "Id" partout
    for df in [X_train, X_test]:
        if "Id" in df.columns:
            df.drop(columns="Id", inplace=True)

    # Preprocessor basé sur train uniquement (pour éviter data leakage)
    preprocessor, _ = build_preprocessor(X_train)

    # Important : NE PAS transformer ici, mais retourner les données
    # avec les types appropriés pour le pipeline
    return X_train, y_train.values, X_test, preprocessor, test_ids


__all__ = ["build_preprocessor", "prepare_features"]