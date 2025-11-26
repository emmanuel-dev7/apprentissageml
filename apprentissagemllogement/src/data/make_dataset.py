"""Simple dataset loader for the housing dataset.

This module contains helpers to load the train and test CSV files located
under data/raw/ and return pandas DataFrames for downstream processing.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def load_raw_data(root: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load train and test files under data/raw.

	Parameters
	----------
	root:
		Project root directory where `data/raw/train.csv` and `data/raw/test.csv`
		live. If None, the current working directory is used.

	Returns
	-------
	(train_df, test_df)
	"""
	base = root or os.getcwd()
	train_path = os.path.join(base, "data", "raw", "train.csv")
	test_path = os.path.join(base, "data", "raw", "test.csv")

	train = pd.read_csv(train_path)
	test = pd.read_csv(test_path)

	return train, test


def load_train_test_split(train_df: pd.DataFrame):
	"""Split a train dataframe into X (features) and y (target SalePrice).

	Returns
	-------
	X, y
	"""
	if "SalePrice" not in train_df.columns:
		raise ValueError("train_df must contain a SalePrice column")
	X = train_df.drop(columns=["SalePrice"])
	y = train_df["SalePrice"].copy()
	return X, y


__all__ = ["load_raw_data", "load_train_test_split"]

