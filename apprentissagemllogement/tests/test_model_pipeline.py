import os
from pathlib import Path

import joblib


from src.data.make_dataset import load_raw_data
from src.features.build_features import prepare_features


def test_prepare_features_shapes():
    base = Path(__file__).resolve().parents[1]
    train_df, test_df = load_raw_data(base)

    X_train_t, y_train, X_test_t, preprocessor, test_ids = prepare_features(train_df, test_df)

    # the transformed train/test arrays should have same number of rows as inputs
    assert X_train_t.shape[0] == train_df.shape[0]
    assert X_test_t.shape[0] == test_df.shape[0]
    assert y_train.shape[0] == train_df.shape[0]


def test_train_and_predict_end_to_end(tmp_path):
    base = Path(__file__).resolve().parents[1]
    # train the model and save into a temp model file
    from src.models.train_model import train_and_save
    from apprentissagemllogement.src.models.predict import predict_and_save

    model_file = tmp_path / "model_test.joblib"
    out_submission = tmp_path / "submission_test.csv"

    saved = train_and_save(root=base, output_path=str(model_file))
    assert Path(saved).exists()

    predicted = predict_and_save(root=base, model_path=str(model_file), output_csv=str(out_submission))
    assert Path(predicted).exists()

    # check output has Id and SalePrice columns
    import pandas as pd

    df = pd.read_csv(predicted)
    assert "Id" in df.columns and "SalePrice" in df.columns
    # number of predictions should match number of rows in test.csv
    _, test_df = load_raw_data(base)
    assert df.shape[0] == test_df.shape[0]
