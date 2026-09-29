"""
test_project.py
Automated tests for the Student Performance Prediction project (roadmap Step 10).

Run from the project root:  pytest tests/ -v
"""

import json
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import (
    PROJECT_ROOT,
    TARGET,
    build_preprocessor,
    load_clean_data,
    split_data,
)
from train_models import build_model_pipeline, get_models, regression_metrics

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "student_mat_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
APP_PATH = PROJECT_ROOT / "app" / "app.py"


def _skip_if_missing(path, step_name):
    if not path.exists():
        pytest.skip(f"{path.relative_to(PROJECT_ROOT)} not found. Run {step_name} first.")


# ======================================================================
# 1. DATASET LOADING
# ======================================================================
def test_01_dataset_loading():
    _skip_if_missing(RAW_PATH, "Step 1 (download_data.py)")
    df = pd.read_csv(RAW_PATH, sep=";")
    assert df.shape[1] == 33, f"Expected 33 columns, got {df.shape[1]}"
    assert len(df) > 0, "Raw file loaded but has 0 rows"
    assert TARGET in df.columns, f"'{TARGET}' column is missing"
    print(f"\n  Raw data loaded: {df.shape[0]} rows x {df.shape[1]} columns")


# ======================================================================
# 2. MISSING VALUES
# ======================================================================
def test_02_missing_values():
    _skip_if_missing(CLEAN_PATH, "clean_data.py")
    df = pd.read_csv(CLEAN_PATH)
    n_missing = int(df.isna().sum().sum())
    fully_empty_cols = [c for c in df.columns if df[c].isna().all()]
    assert not fully_empty_cols, f"These columns are entirely empty: {fully_empty_cols}"
    print(f"\n  Total missing values in cleaned data: {n_missing}")
    print("  (0 is expected per the dataset documentation, but any number is fine "
          "as long as it was REPORTED, not silently created.)")


# ======================================================================
# 3. DATA PREPROCESSING (the pipeline, not a model)
# ======================================================================
def test_03_preprocessing_pipeline():
    _skip_if_missing(CLEAN_PATH, "clean_data.py")
    df = load_clean_data(drop_zero_grades=False)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    for use_prior in (True, False):
        pre = build_preprocessor(use_prior_grades=use_prior)
        Xt = pre.fit_transform(X, y)
        assert not np.isnan(Xt).any(), "Transformed data contains NaN"
        names = list(pre.named_steps["columns"].get_feature_names_out())
        bare = [n.split("__", 1)[1] for n in names]
        assert TARGET not in bare, "Target leaked into the transformed features!"
        if not use_prior:
            assert not {"G1", "G2"}.intersection(bare), "G1/G2 leaked into the no-prior-grades pipeline!"
        print(f"\n  use_prior_grades={use_prior}: {Xt.shape[1]} output features, no NaN, no leakage")


# ======================================================================
# 4. TRAIN/TEST SPLIT
# ======================================================================
def test_04_train_test_split():
    _skip_if_missing(CLEAN_PATH, "clean_data.py")
    df = load_clean_data(drop_zero_grades=False)
    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df), "Rows lost or duplicated by the split"
    assert set(X_train.index).isdisjoint(X_test.index), "Train and test rows overlap"
    assert TARGET not in X_train.columns and TARGET not in X_test.columns, "Target leaked into X"
    print(f"\n  Train: {X_train.shape}   Test: {X_test.shape}   "
          f"Split ratio: {len(X_test) / len(df):.1%} test")


# ======================================================================
# 5. MODEL TRAINING
# ======================================================================
def test_05_model_training():
    _skip_if_missing(CLEAN_PATH, "clean_data.py")
    df = load_clean_data(drop_zero_grades=False)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_model_pipeline(get_models()["Linear Regression"], use_prior_grades=True)
    pipeline.fit(X_train, y_train)          # must not raise
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test), "Prediction count does not match test set size"
    assert np.isfinite(preds).all(), "Predictions contain NaN or infinity"
    print(f"\n  Linear Regression trained and predicted {len(preds)} test rows without error")


# ======================================================================
# 6. MODEL EVALUATION (the metric formulas themselves)
# ======================================================================
def test_06_model_evaluation_metrics():
    # Hand-computable example: errors are 1, 1, 1, 3 -> known MAE/MSE/RMSE/R2
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([11.0, 9.0, 11.0, 13.0])   # errors: 1, 1, 1, 3

    m = regression_metrics(y_true, y_pred)

    assert m["MAE"] == pytest.approx((1 + 1 + 1 + 3) / 4)          # 1.5
    assert m["MSE"] == pytest.approx((1 + 1 + 1 + 9) / 4)          # 3.0
    assert m["RMSE"] == pytest.approx(np.sqrt(3.0))                # ~1.732
    assert m["RMSE"] >= m["MAE"], "RMSE should never be smaller than MAE"
    assert m["R2"] <= 1.0
    print(f"\n  Hand-checked metrics correct: {m}")


# ======================================================================
# 7. MODEL SAVING
# ======================================================================
def test_07_model_saving():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    assert MODEL_PATH.stat().st_size > 0, "Model file exists but is empty"
    _skip_if_missing(METADATA_PATH, "save_model.py")
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    required_keys = {"model_name", "scenario", "input_columns", "input_schema", "test_metrics"}
    assert required_keys.issubset(meta), f"Metadata is missing keys: {required_keys - set(meta)}"
    print(f"\n  Model file: {MODEL_PATH.stat().st_size / 1024:.1f} KB. Metadata has all required keys.")


# ======================================================================
# 8. MODEL LOADING
# ======================================================================
def test_08_model_loading():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    import joblib
    from sklearn.pipeline import Pipeline

    model = joblib.load(MODEL_PATH)
    assert isinstance(model, Pipeline), "Loaded object is not a scikit-learn Pipeline"
    assert list(model.named_steps) == ["preprocess", "model"], "Pipeline is missing expected steps"
    print(f"\n  Loaded pipeline steps: {list(model.named_steps)}")


# ======================================================================
# 9. PREDICTION FUNCTION
# ======================================================================
def test_09_prediction_function():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    from predict import get_metadata, predict_student

    meta = get_metadata()
    example = {
        col: (rule["values"][0] if rule["type"] == "categorical" else rule["min"])
        for col, rule in meta["input_schema"].items()
    }
    grade = predict_student(example)
    assert isinstance(grade, float), f"Expected a float, got {type(grade)}"
    assert 0 <= grade <= 20, f"Prediction {grade} is outside the 0-20 grade scale"
    print(f"\n  predict_student(example) = {grade}  (within 0-20 as required)")


# ======================================================================
# 10. STREAMLIT APP (smoke test: does it even compile? full UI is manual)
# ======================================================================
def test_10_streamlit_app_compiles():
    _skip_if_missing(APP_PATH, "the Streamlit app step")
    try:
        py_compile.compile(str(APP_PATH), doraise=True)
    except py_compile.PyCompileError as error:
        pytest.fail(f"app/app.py has a syntax error:\n{error}")
    print("\n  app/app.py compiles with no syntax errors. "
          "(Manually run 'streamlit run app/app.py' to test the actual UI.)")


# ======================================================================
# 11. INVALID USER INPUTS
# ======================================================================
def test_11_invalid_inputs_missing_field():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    from predict import InvalidStudentInput, get_metadata, predict_student

    meta = get_metadata()
    example = {
        col: (rule["values"][0] if rule["type"] == "categorical" else rule["min"])
        for col, rule in meta["input_schema"].items()
    }
    incomplete = dict(example)
    del incomplete[meta["input_columns"][0]]   # remove one required field

    with pytest.raises(InvalidStudentInput):
        predict_student(incomplete)
    print("\n  A missing required field correctly raised InvalidStudentInput")


def test_11b_invalid_inputs_bad_type():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    from predict import InvalidStudentInput, get_metadata, predict_student

    meta = get_metadata()
    example = {
        col: (rule["values"][0] if rule["type"] == "categorical" else rule["min"])
        for col, rule in meta["input_schema"].items()
    }
    numeric_cols = [c for c, r in meta["input_schema"].items() if r["type"] == "numeric"]
    if not numeric_cols:
        pytest.skip("No numeric columns in this model's schema")
    bad = dict(example)
    bad[numeric_cols[0]] = "not_a_number"

    with pytest.raises(InvalidStudentInput):
        predict_student(bad)
    print(f"\n  A non-numeric value in '{numeric_cols[0]}' correctly raised InvalidStudentInput")


# ======================================================================
# 12. MISSING MODEL FILE
# ======================================================================
def test_12_missing_model_file(monkeypatch, tmp_path):
    import predict

    fake_path = tmp_path / "does_not_exist.pkl"
    monkeypatch.setattr(predict, "MODEL_PATH", fake_path)
    predict._load_model.cache_clear()   # clear the cached real model for this test only

    with pytest.raises(FileNotFoundError):
        predict._load_model()

    predict._load_model.cache_clear()   # restore: next call reloads the REAL model
    print("\n  A missing model file correctly raised FileNotFoundError, "
          "and the cache was reset afterwards")


# ======================================================================
# 13. WRONG FEATURE INPUT (unexpected / typo'd field)
# ======================================================================
def test_13_wrong_feature_input():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    from predict import InvalidStudentInput, get_metadata, predict_student

    meta = get_metadata()
    example = {
        col: (rule["values"][0] if rule["type"] == "categorical" else rule["min"])
        for col, rule in meta["input_schema"].items()
    }
    example["this_column_does_not_exist"] = 123   # a typo / made-up field

    with pytest.raises(InvalidStudentInput):
        predict_student(example)
    print("\n  An unexpected/unknown field correctly raised InvalidStudentInput")


# ======================================================================
# 14. REPRODUCIBILITY
# ======================================================================
def test_14_reproducibility_split():
    _skip_if_missing(CLEAN_PATH, "clean_data.py")
    df = load_clean_data(drop_zero_grades=False)

    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(df)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(df)

    assert X_train_1.index.equals(X_train_2.index), "Train split changed between runs"
    assert X_test_1.index.equals(X_test_2.index), "Test split changed between runs"
    print("\n  Running split_data() twice gives identical train/test rows")


def test_14b_reproducibility_prediction():
    _skip_if_missing(MODEL_PATH, "save_model.py")
    from predict import get_metadata, predict_student

    meta = get_metadata()
    example = {
        col: (rule["values"][0] if rule["type"] == "categorical" else rule["min"])
        for col, rule in meta["input_schema"].items()
    }
    grade_1 = predict_student(example)
    grade_2 = predict_student(example)
    assert grade_1 == grade_2, "Same input gave different predictions on two calls"
    print(f"\n  Same input predicted twice: {grade_1} == {grade_2}")