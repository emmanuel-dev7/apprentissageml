import os
from pathlib import Path

import pandas as pd


def test_raw_files_exist_and_load():
    base = Path(__file__).resolve().parents[1]
    train_path = base / "data" / "raw" / "train.csv"
    test_path = base / "data" / "raw" / "test.csv"

    assert train_path.exists(), f"train.csv not found at {train_path}"
    assert test_path.exists(), f"test.csv not found at {test_path}"

    # can we load a small portion?
    df = pd.read_csv(train_path, nrows=5)
    assert not df.empty

