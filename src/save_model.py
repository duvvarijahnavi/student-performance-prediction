"""
save_model.py
Builds the FINAL pipeline (preprocessing + model), fits it on the training set,
evaluates it on the test set, and saves it with joblib.

Saves:
  models/student_performance_model.pkl   the whole Pipeline (preprocessing AND model together)
  models/model_metadata.json             scenario, parameters, versions, input columns, metrics

Run from the project root:  python src/save_model.py
"""

import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from features import (
    PROJECT_ROOT,
    RANDOM_STATE,
    SCENARIOS,
    TARGET,
    TEST_SIZE,
    load_clean_data,
    split_data,
)
from train_models import BASELINE_NAME, build_model_pipeline, get_models, regression_metrics

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# Which scenario should the saved model serve?
#   "A_late_prediction" : uses G1, G2 (the app must ask for earlier grades)
#   "B_early_warning"   : does NOT use G1, G2 (usable before any grade exists)
# Decide this by the PURPOSE of your app, not by the scores.
FINAL_SCENARIO = "B_early_warning"

# Must be the SAME value as in train_models.py and tune_model.py.
DROP_ZERO_GRADES = False

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "student_performance_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
TUNED_PARAMS_PATH = PROJECT_ROOT / "reports" / "tuned_params.json"
SELECTION_PATH = PROJECT_ROOT / "reports" / "best_model_selection.csv"


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def choose_model_spec(scenario):
    """
    Decide WHICH model and WHICH settings to save, using results from earlier steps.
    Returns (model_name, params_dict, explanation).
    """
    # 1. Preferred source: the tuning step's verdict
    if TUNED_PARAMS_PATH.exists():
        info = json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))
        if scenario in info:
            entry = info[scenario]
            if entry["adopt_tuned"]:
                return (entry["tuned_model"], entry["best_params"],
                        "tuned settings (adopted by the cross-validation rule in tune_model.py)")
            return (entry["selected_model"], {},
                    "starting settings (tuning gave no clear gain, so the simpler original is kept)")

    # 2. Fallback: the model selected in Step 10, with starting settings
    if SELECTION_PATH.exists():
        selection = pd.read_csv(SELECTION_PATH)
        row = selection[selection["scenario"] == scenario]
        if not row.empty and str(row["selected_model"].iloc[0]) != "none":
            return (str(row["selected_model"].iloc[0]), {},
                    "starting settings (no tuning result found for this scenario)")
        raise SystemExit(f"No model was selected for {scenario} in best_model_selection.csv.")

    raise SystemExit(
        "Neither reports/tuned_params.json nor reports/best_model_selection.csv exists.\n"
        "Run train_models.py, select_best_model.py and tune_model.py first."
    )


def build_input_schema(inputs):
    """Allowed values per input column. Step 9 (Streamlit) will use this for its widgets."""
    schema = {}
    for col in inputs.columns:
        if pd.api.types.is_numeric_dtype(inputs[col]):
            schema[col] = {"type": "numeric",
                           "min": float(inputs[col].min()), "max": float(inputs[col].max())}
        else:
            schema[col] = {"type": "categorical",
                           "values": sorted(inputs[col].dropna().astype(str).unique())}
    return schema


# ----------------------------------------------------------------------
# 1. LOAD AND SPLIT (same function and seed as every other script)
# ----------------------------------------------------------------------
if FINAL_SCENARIO not in SCENARIOS:
    raise SystemExit(f"FINAL_SCENARIO must be one of {list(SCENARIOS)}")

df = load_clean_data(drop_zero_grades=DROP_ZERO_GRADES)
X_train, X_test, y_train, y_test = split_data(df)
assert TARGET not in X_train.columns and TARGET not in X_test.columns, "G3 leaked into X!"
print(f"Rows: {len(df)}  |  train: {X_train.shape}  test: {X_test.shape}"
      f"  (DROP_ZERO_GRADES = {DROP_ZERO_GRADES})")

# ----------------------------------------------------------------------
# 2. DECIDE WHAT TO SAVE
# ----------------------------------------------------------------------
use_prior = SCENARIOS[FINAL_SCENARIO]["use_prior_grades"]
model_name, params, source = choose_model_spec(FINAL_SCENARIO)

if model_name == BASELINE_NAME or model_name not in get_models():
    raise SystemExit(f"'{model_name}' is not a valid model to save.")

print(f"\nScenario       : {FINAL_SCENARIO}")
print(SCENARIOS[FINAL_SCENARIO]["description"])
print(f"Model          : {model_name}")
print(f"Settings source: {source}")
print(f"Parameters     : {params if params else '(starting settings)'}")

# ----------------------------------------------------------------------
# 3. BUILD AND FIT THE PIPELINE (training rows only)
# ----------------------------------------------------------------------
model = get_models()[model_name]
if params:
    model.set_params(**params)

pipeline = build_model_pipeline(model, use_prior)   # preprocessing + model in ONE object
pipeline.fit(X_train, y_train)                      # every learned step sees TRAIN rows only

# ----------------------------------------------------------------------
# 4. EVALUATE ONCE (for the record; do not tune based on these numbers)
# ----------------------------------------------------------------------
train_metrics = {k: float(v) for k, v in regression_metrics(y_train, pipeline.predict(X_train)).items()}
test_pred = pipeline.predict(X_test)
test_metrics = {k: float(v) for k, v in regression_metrics(y_test, test_pred).items()}

print("\nMetrics of the model that will be saved:")
print(pd.DataFrame({"train": train_metrics, "test": test_metrics}).round(3).to_string())
print("(The test set is the final check. Do not change the model to improve these numbers.)")

# ----------------------------------------------------------------------
# 5. SAVE
# ----------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\nSaved model    -> {MODEL_PATH.relative_to(PROJECT_ROOT)}"
      f"  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")

# ----------------------------------------------------------------------
# 6. ROUND-TRIP CHECK: reload the file and compare predictions
# ----------------------------------------------------------------------
reloaded = joblib.load(MODEL_PATH)
assert list(reloaded.named_steps) == ["preprocess", "model"], "Saved object has unexpected steps!"
assert np.allclose(reloaded.predict(X_test), test_pred, rtol=0, atol=1e-12), \
    "Reloaded model gives different predictions!"
print("Round-trip check: the reloaded pipeline gives identical predictions.  OK")

# ----------------------------------------------------------------------
# 7. SAVE METADATA
# ----------------------------------------------------------------------
metadata = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "scenario": FINAL_SCENARIO,
    "use_prior_grades": use_prior,
    "model_name": model_name,
    "settings_source": source,
    "model_params": pipeline.named_steps["model"].get_params(),
    "target": TARGET,
    "prediction_range": [0, 20],   # the app should clip predictions to the grade scale
    "drop_zero_grades": DROP_ZERO_GRADES,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "input_columns": list(X_train.columns),
    "input_schema": build_input_schema(df.drop(columns=[TARGET])),
    "versions": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
    "note": "Educational project. Not for making real decisions about students.",
}
METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
print(f"Saved metadata -> {METADATA_PATH.relative_to(PROJECT_ROOT)}")

print("\nDone. Next: python src/verify_model.py")