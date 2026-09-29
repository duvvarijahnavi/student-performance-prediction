"""
predict.py
Loads the saved model ONCE and exposes predict_student(), which turns a plain
dictionary of student data into a final-grade prediction.

Uses ONLY the feature names recorded in models/model_metadata.json
(itself produced from the real training data by save_model.py). No feature
name is invented here.

Run directly for a demo:  python src/predict.py
Import from elsewhere (e.g. the Streamlit app):
    from predict import predict_student, get_required_fields, get_metadata
"""

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.features import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


class InvalidStudentInput(ValueError):
    """Raised when the input dictionary does not match what the model expects."""


# ----------------------------------------------------------------------
# LOAD ONCE, REUSE (lru_cache keeps the same object across every call)
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 'python src/save_model.py' first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata():
    """Scenario, expected columns, allowed values, and metrics recorded when the model was saved."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}. Run 'python src/save_model.py' first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def get_required_fields():
    """The EXACT list of keys predict_student() needs, taken from the saved model. Nothing invented."""
    return list(get_metadata()["input_columns"])


# ----------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------
def _validate_and_order(student, meta):
    """
    Check `student` against the model's real input columns and schema.
    Returns (ordered_dict, list_of_warning_strings). Raises InvalidStudentInput
    for problems that would make the prediction meaningless or crash the model.
    """
    required = meta["input_columns"]
    schema = meta["input_schema"]
    warnings = []

    if not isinstance(student, dict):
        raise InvalidStudentInput(f"Expected a dict of student data, got {type(student).__name__}.")

    missing = [c for c in required if c not in student]
    if missing:
        raise InvalidStudentInput(
            f"Missing {len(missing)} required field(s): {missing}\n"
            f"All required fields ({len(required)}): {required}"
        )

    unexpected = [c for c in student if c not in required]
    if unexpected:
        raise InvalidStudentInput(
            f"Unexpected field(s) not used by this model: {unexpected}\n"
            f"This model only accepts: {required}"
        )

    ordered = {}
    for col in required:   # fixed order: the same order the pipeline was trained with
        value = student[col]
        rule = schema[col]

        if rule["type"] == "numeric":
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise InvalidStudentInput(f"'{col}' must be a number, got {value!r}.")
            if not (rule["min"] <= value <= rule["max"]):
                warnings.append(
                    f"'{col}' = {value} is outside the range seen in training "
                    f"({rule['min']:.0f} to {rule['max']:.0f}). Prediction may be unreliable."
                )
        else:  # categorical
            value = str(value)
            if value not in rule["values"]:
                warnings.append(
                    f"'{col}' = {value!r} was never seen in training. "
                    f"Known values: {rule['values']}. The model will treat it as unknown."
                )
        ordered[col] = value

    return ordered, warnings


# ----------------------------------------------------------------------
# THE PREDICTION FUNCTION
# ----------------------------------------------------------------------
def predict_student(student, return_details=False):
    """
    Predict the final grade (G3) for ONE student.

    Parameters
    ----------
    student : dict
        Keys must match get_required_fields() exactly (see get_metadata()["input_schema"]
        for the allowed values of each field). No feature name is invented: this list
        comes straight from the trained model's own metadata.
    return_details : bool, default False
        If False (default): returns just the predicted grade, e.g. 12.4
        If True: returns a dict with the grade, any input warnings, and the model info.

    Returns
    -------
    float, or dict if return_details=True

    Raises
    ------
    InvalidStudentInput
        If a required field is missing, an extra field is given, or a numeric
        field cannot be converted to a number.
    """
    meta = get_metadata()
    model = _load_model()

    # 1. validate the input and put it in the exact column order the model needs
    ordered, warnings = _validate_and_order(student, meta)

    # 2. one-row DataFrame, exact column names and order from metadata (nothing invented)
    row = pd.DataFrame([ordered], columns=meta["input_columns"])

    # 3 & 4. the loaded pipeline runs preprocessing AND the model, then predicts
    raw_prediction = float(model.predict(row)[0])

    # 5. clip to the real grade scale (a regression model can output outside 0-20)
    low, high = meta["prediction_range"]
    prediction = max(low, min(high, raw_prediction))

    if not return_details:
        return round(prediction, 2)

    return {
        "predicted_grade": round(prediction, 2),
        "raw_prediction": round(raw_prediction, 2),
        "clipped": raw_prediction != prediction,
        "warnings": warnings,
        "scenario": meta["scenario"],
        "model_name": meta["model_name"],
        "expected_test_mae": meta["test_metrics"]["MAE"],
    }


# ----------------------------------------------------------------------
# DEMO (runs only when this file is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    meta = get_metadata()
    print(f"Scenario           : {meta['scenario']}")
    print(f"Model               : {meta['model_name']}")
    print(f"Required fields ({len(meta['input_columns'])}): {meta['input_columns']}")
    print(f"Test MAE when saved : {meta['test_metrics']['MAE']:.2f} grade points "
          f"(a rough idea of how far predictions typically are from the real grade)")

    # ---- Example 1: a complete, valid student ---------------------------
    # Values are taken from the schema itself, so this example is GUARANTEED
    # to use only real column names and real category values, whatever they are.
    print("\n--- Example 1: a valid student built from the model's own schema ---")
    example = {}
    for col, rule in meta["input_schema"].items():
        example[col] = rule["values"][0] if rule["type"] == "categorical" else rule["min"]
    print("Input:", example)

    result = predict_student(example, return_details=True)
    print("Result:", result)
    print(f"\nPredicted final grade (G3): {result['predicted_grade']} out of 20")
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(" -", w)

    # Same call without details, exactly as Requirement 5 asks: return the grade
    grade_only = predict_student(example)
    print(f"\npredict_student(example)  ->  {grade_only}")

    # ---- Example 2: a missing field (should raise a clear error) --------
    print("\n--- Example 2: a required field is missing (on purpose, to show the error) ---")
    broken = dict(example)
    first_key = meta["input_columns"][0]
    del broken[first_key]
    try:
        predict_student(broken)
    except InvalidStudentInput as error:
        print(f"Correctly rejected. InvalidStudentInput: {error}")

    # ---- Example 3: an unexpected field (a typo, say) --------------------
    print("\n--- Example 3: an extra/unexpected field (on purpose) ---")
    broken2 = dict(example)
    broken2["not_a_real_column"] = 5
    try:
        predict_student(broken2)
    except InvalidStudentInput as error:
        print(f"Correctly rejected. InvalidStudentInput: {error}")