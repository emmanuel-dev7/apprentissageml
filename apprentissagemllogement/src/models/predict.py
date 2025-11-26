# empty
"""Prediction code to load trained pipeline and create submission CSV.

This module expects a joblib file that contains a dict with keys
`preprocessor` and `estimator` (the latter must implement predict).
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def predict_and_save(
	root: str | Path = None, model_path: str | Path = None, output_csv: str | Path = None
):
	"""Load the pipeline + estimator and use it to predict on data/raw/test.csv

	The output CSV saved has columns [Id, SalePrice] matching Kaggle submission.
	"""
	base = Path(root) if root else Path.cwd()
	model_file = Path(model_path) if model_path else base / "models" / "model.joblib"
	out = Path(output_csv) if output_csv else base / "sample_submission.csv"

	wrapper = joblib.load(model_file)
	preprocessor = wrapper["preprocessor"]
	estimator = wrapper["model"]

	test_df = pd.read_csv(base / "data" / "raw" / "test.csv")

	test_ids = test_df["Id"].copy() if "Id" in test_df.columns else pd.Series(range(len(test_df)))
	X_test = test_df.drop(columns=[c for c in ("Id",) if c in test_df.columns])

	X_test_t = preprocessor.transform(X_test)
	preds = estimator.predict(X_test_t)

	submission = pd.DataFrame({"Id": test_ids.values.flatten(), "SalePrice": preds})
	submission.to_csv(out, index=False)

	return out


if __name__ == "__main__":
	print("Loading model and predicting on data/raw/test.csv -> sample_submission.csv")
	outp = predict_and_save()
	print("Saved predictions to", outp)


__all__ = ["predict_and_save"]
