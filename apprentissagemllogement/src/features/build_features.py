"""Feature building helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    numerics = df.select_dtypes(include=["number"]).columns.tolist()
    categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove Id if present
    for col in ["Id", "id", "ID"]:
        numerics = [c for c in numerics if c != col]
        categoricals = [c for c in categoricals if c != col]

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

    # Pour info/debug : noms approximatifs des colonnes après transformation
    output_columns = num_cols + cat_cols
    return preprocessor, output_columns


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prépare les features pour entraînement et prédiction.
    Retourne X_train et X_test déjà transformés + le preprocessor fitté.
    """
    # Séparation cible
    if "SalePrice" not in train_df.columns:
        raise ValueError("train_df doit contenir la colonne SalePrice")
    y_train = train_df["SalePrice"].copy()
    X_train = train_df.drop(columns=["SalePrice"])

    X_test = test_df.copy()
    X_test_ids = X_test["Id"].copy() if "Id" in X_test.columns else None

    # Suppression de Id des features
    for df in [X_train, X_test]:
        if "Id" in df.columns:
            df.drop(columns="Id", inplace=True)

    # On fit le preprocessor sur train + test concaténés → même traitement des catégories
    full_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    preprocessor, _ = build_preprocessor(full_df)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    return X_train_t, y_train.values, X_test_t, preprocessor, X_test_ids


__all__ = ["build_preprocessor", "prepare_features"]