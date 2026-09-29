"""
verify_model.py
Loads models/student_performance_model.pkl in a FRESH Python process and checks that
it works: structure, test metrics, single raw rows, unusual input.

Run from the project root:  python src/verify_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from features import PROJECT_ROOT, TARGET, load_clean_data, split_data
from train_models import regression_metrics

MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise SystemExit("Model files not found. Run 'python src/save_model.py' first.")

meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
checks_passed = 0


def ok(message):
    """Count and print one passed check."""
    global checks_passed
    checks_passed += 1
    print(f"  OK  {message}")


# ----------------------------------------------------------------------
print("1. LOAD THE SAVED FILE")
# ----------------------------------------------------------------------
print(f"File: {MODEL_PATH.relative_to(PROJECT_ROOT)}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
if meta["versions"]["scikit-learn"] != sklearn.__version__:
    print(f"  WARNING: saved with scikit-learn {meta['versions']['scikit-learn']}, "
          f"now running {sklearn.__version__}. Retrain and resave if you see errors.")

model = joblib.load(MODEL_PATH)
ok("file loaded with joblib")

# ----------------------------------------------------------------------
print("\n2. PREPROCESSING AND MODEL ARE SAVED TOGETHER")
# ----------------------------------------------------------------------
assert isinstance(model, Pipeline), "The saved object is not a Pipeline!"
assert list(model.named_steps) == ["preprocess", "model"], "Unexpected pipeline steps!"
assert list(model.named_steps["preprocess"].named_steps) == ["engineer", "columns"], \
    "Preprocessing steps are missing!"
ok("object is a single Pipeline with steps: preprocess -> model")
ok("preprocess contains: engineer (new features) -> columns (imputing, scaling, encoding)")
print(f"      final model: {type(model.named_steps['model']).__name__}")
print(f"      scenario   : {meta['scenario']}")
n_out = len(model.named_steps["preprocess"].named_steps["columns"].get_feature_names_out())
print(f"      the model receives {n_out} processed features")

# ----------------------------------------------------------------------
print("\n3. TEST-SET METRICS REPRODUCE THE SAVED ONES")
# ----------------------------------------------------------------------
df = load_clean_data(drop_zero_grades=meta["drop_zero_grades"])   # same setting as when saving
X_train, X_test, y_train, y_test = split_data(df)
assert TARGET not in X_test.columns, "G3 leaked into X!"
assert list(X_test.columns) == meta["input_columns"], "Input columns differ from the saved ones!"

pred = model.predict(X_test)
now = regression_metrics(y_test, pred)
comparison = pd.DataFrame({"saved_in_metadata": meta["test_metrics"], "recomputed_now": now})
print(comparison.round(4).to_string())
for name, value in meta["test_metrics"].items():
    assert abs(value - now[name]) < 1e-9, f"Test {name} does not match the saved value!"
ok("recomputed test MAE, MSE, RMSE and R2 equal the values stored at saving time")

# ----------------------------------------------------------------------
print("\n4. ONE RAW ROW, LIKE A STREAMLIT INPUT")
# ----------------------------------------------------------------------
row = X_test.iloc[0].to_dict()
row.update({"studytime": 3, "absences": 4, "failures": 0, "higher": "yes"})   # pretend a user typed these
new_student = pd.DataFrame([row])[meta["input_columns"]]

single = float(model.predict(new_student)[0])
print(f"Prediction for the example student: {single:.2f}  (grade scale 0-20)")
assert np.isfinite(single), "Prediction is not a finite number!"
ok("one raw row goes in, one number comes out (no manual preprocessing needed)")

shuffled = new_student[list(reversed(meta["input_columns"]))]
assert np.isclose(float(model.predict(shuffled)[0]), single), "Column order changes the result!"
ok("column order does not matter (columns are selected by name)")

both = pd.concat([X_test.iloc[[0]], new_student], ignore_index=True)
assert np.isclose(model.predict(both)[1], single), "Single-row and batch predictions differ!"
ok("predicting one row gives the same value as predicting it inside a batch")

# ----------------------------------------------------------------------
print("\n5. UNUSUAL INPUT")
# ----------------------------------------------------------------------
odd = new_student.copy()
odd["Mjob"] = "astronaut"   # a category never seen in training
odd_value = float(model.predict(odd)[0])
assert np.isfinite(odd_value)
ok(f"an unseen category does not crash (handle_unknown='ignore'), prediction = {odd_value:.2f}")

low, high = meta["prediction_range"]
outside = int(((pred < low) | (pred > high)).sum())
print(f"  INFO  test predictions outside {low}-{high}: {outside} of {len(pred)} "
      f"(min {pred.min():.2f}, max {pred.max():.2f}). "
      f"The app should clip predictions to {low}-{high}.")

if not meta["use_prior_grades"]:
    try:
        model.predict(new_student.drop(columns=["G1", "G2"]))
        print("  INFO  Scenario B model also accepts a row WITHOUT G1 and G2.")
    except Exception as error:
        print(f"  INFO  Scenario B model needs ALL {len(meta['input_columns'])} raw columns, "
              f"even though it ignores G1 and G2 ({type(error).__name__}).")
        print("        In Step 9 the app will fill G1 and G2 with placeholder values.")

# ----------------------------------------------------------------------
print(f"\nALL {checks_passed} CHECKS PASSED. The saved model is ready for the Streamlit app.")