"""
features.py
Reusable feature-engineering code for the Student Performance project.

This file is IMPORTED by other scripts (build_features.py, and later the model
training script and the Streamlit app). You do not run it directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "student_mat_clean.csv"

TARGET = "G3"
PASS_MARK = 10        # OUR assumption for the pass/fail version. State it in the README.
PRIOR_GRADES = ["G1", "G2"]
RANDOM_STATE = 42     # fixed, so every script gets the SAME train/test split
TEST_SIZE = 0.2

# Two scenarios, explained in the README:
#   A = predict late in the term (earlier grades available)
#   B = early warning (earlier grades NOT available)
SCENARIOS = {
    "A_late_prediction": {
        "use_prior_grades": True,
        "description": "Scenario A (late prediction): G1, G2 and grade_trend are used.",
    },
    "B_early_warning": {
        "use_prior_grades": False,
        "description": "Scenario B (early warning): G1, G2 and grade_trend are NOT used.",
    },
}

# ----------------------------------------------------------------------
# COLUMN GROUPS
# ----------------------------------------------------------------------
BINARY_CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
]
NOMINAL_CATEGORICAL = ["Mjob", "Fjob", "reason", "guardian"]
CATEGORICAL = BINARY_CATEGORICAL + NOMINAL_CATEGORICAL

# Numeric columns kept as they are. Ordinal codes (1-5 ratings, education levels)
# stay numeric because their ORDER is meaningful.
NUMERIC_BASE = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
]

# Created by add_engineered_features(). Raw 'absences' is replaced by 'absences_log'.
ENGINEERED_NUMERIC = ["absences_log", "parent_edu_avg", "alcohol_avg", "has_past_failure"]


# ----------------------------------------------------------------------
# DATA LOADING AND SPLITTING
# ----------------------------------------------------------------------
def load_clean_data(drop_zero_grades=False):
    """Load the cleaned CSV. Rows are removed ONLY if drop_zero_grades=True (and we say so)."""
    if not CLEAN_PATH.exists():
        raise FileNotFoundError("Cleaned data not found. Run 'python src/clean_data.py' first.")
    df = pd.read_csv(CLEAN_PATH)  # the cleaned file uses commas
    if drop_zero_grades:
        n_zero = int((df[TARGET] == 0).sum())
        df = df[df[TARGET] > 0].reset_index(drop=True)
        print(f"NOTE: drop_zero_grades=True removed {n_zero} rows with {TARGET} == 0.")
    return df


def make_pass_fail(y):
    """Classification target: 1 = pass (G3 >= PASS_MARK), 0 = fail. Built from y only."""
    return (y >= PASS_MARK).astype(int)


def split_data(df):
    """
   Split into train and test sets BEFORE any learning happens.
    X never contains G3. We stratify on pass/fail so both sets have a similar
    share of passing students (stratification uses the label only to balance the split).
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    strat = make_pass_fail(y)
    if strat.value_counts().min() < 2:  # stratifying needs at least 2 rows per class
        strat = None
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )



# ----------------------------------------------------------------------
# FEATURE ENGINEERING (row by row: no target, no statistics from other rows)
# ----------------------------------------------------------------------
def add_engineered_features(data):
    """Return a copy of the DataFrame with the new columns added."""
    out = data.copy()
    out["absences_log"] = np.log1p(out["absences"])            # tames the long right tail
    out["parent_edu_avg"] = (out["Medu"] + out["Fedu"]) / 2
    out["alcohol_avg"] = (out["Dalc"] + out["Walc"]) / 2
    # keep a missing 'failures' as missing (NaN) instead of silently calling it "no failure"
    out["has_past_failure"] = (out["failures"] > 0).astype(float).where(out["failures"].notna())
    if {"G1", "G2"}.issubset(out.columns):                      # only when earlier grades exist
        out["grade_trend"] = out["G2"] - out["G1"]
    return out


# ----------------------------------------------------------------------
# THE PREPROCESSING PIPELINE
# ----------------------------------------------------------------------
def get_feature_lists(use_prior_grades):
    """Which columns go to the numeric branch and which to the categorical branch."""
    numeric_cols = NUMERIC_BASE + ENGINEERED_NUMERIC
    if use_prior_grades:
        numeric_cols = numeric_cols + PRIOR_GRADES + ["grade_trend"]
    return numeric_cols, CATEGORICAL


def build_preprocessor(use_prior_grades=True):
    """
    Build an UNFITTED preprocessing pipeline.
    Call .fit(X_train) once, then .transform(...) on train, test or new data.
    """
    numeric_cols, categorical_cols = get_feature_lists(use_prior_grades)

    numeric_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # safety net: fills gaps if any appear
        ("scale", StandardScaler()),                    # mean 0, spread 1 (learned from train only)
    ])
    categorical_branch = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])

    column_step = ColumnTransformer(
        transformers=[
            ("num", numeric_branch, numeric_cols),
            ("cat", categorical_branch, categorical_cols),
        ],
        remainder="drop",   # any column NOT listed above (G3, raw absences, ...) is dropped
    )

    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
        ("columns", column_step),
    ])


def get_feature_names(fitted_pipeline):
    """Names of the columns produced by a FITTED pipeline, e.g. 'num__age', 'cat__sex_M'."""
    return list(fitted_pipeline.named_steps["columns"].get_feature_names_out())