"""
build_features.py
Builds the feature-engineering pipeline for both scenarios and CHECKS it.
It trains no model and does not change any data file.

Saves:
  reports/feature_engineering_report.txt
  reports/feature_names_A_late_prediction.csv
  reports/feature_names_B_early_warning.csv

Run from the project root:  python src/build_features.py
"""

import numpy as np
import pandas as pd

from features import (
    PASS_MARK,
    PROJECT_ROOT,
    SCENARIOS,
    TARGET,
    add_engineered_features,
    build_preprocessor,
    get_feature_names,
    load_clean_data,
    make_pass_fail,
    split_data,
)

# Decision about G3 == 0 rows is still open, so nothing is removed by default.
DROP_ZERO_GRADES = False

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "feature_engineering_report.txt"

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", None)

log_lines = []


def log(text=""):
    """Print a message and keep a copy for the report file."""
    print(text)
    log_lines.append(str(text))


def section(title):
    log("\n" + "=" * 64)
    log(title)
    log("=" * 64)


# ----------------------------------------------------------------------
section("1. LOAD CLEANED DATA AND SPLIT INTO TRAIN / TEST")
# ----------------------------------------------------------------------
df = load_clean_data(drop_zero_grades=DROP_ZERO_GRADES)
log(f"Rows: {len(df)}   Columns: {df.shape[1]}   (DROP_ZERO_GRADES = {DROP_ZERO_GRADES})")

X_train, X_test, y_train, y_test = split_data(df)
log(f"Training rows: {len(X_train)}    Test rows: {len(X_test)}")
log(f"Share passing (G3 >= {PASS_MARK}) in train: {make_pass_fail(y_train).mean():.1%}"
    f"    in test: {make_pass_fail(y_test).mean():.1%}")

# Leakage guard number 1: the target must not be in the inputs
assert TARGET not in X_train.columns and TARGET not in X_test.columns, "G3 leaked into X!"
log(f"Leakage check: '{TARGET}' is not among the {X_train.shape[1]} input columns.  OK")

# ----------------------------------------------------------------------
section("2. ENGINEERED FEATURES (first 8 training rows, so you can verify by eye)")
# ----------------------------------------------------------------------
engineered = add_engineered_features(X_train)
preview_cols = [
    "absences", "absences_log", "Medu", "Fedu", "parent_edu_avg",
    "Dalc", "Walc", "alcohol_avg", "failures", "has_past_failure",
    "G1", "G2", "grade_trend",
]
log(engineered[preview_cols].head(8).round(2).to_string())

log("\nSkewness of absences (training data). Closer to 0 = more symmetric:")
log(f"  raw absences        : {X_train['absences'].skew():.2f}")
log(f"  log(1 + absences)   : {np.log1p(X_train['absences']).skew():.2f}")

# ----------------------------------------------------------------------
# 3. BUILD, FIT AND CHECK THE PIPELINE FOR EACH SCENARIO
# ----------------------------------------------------------------------
for scenario_name, cfg in SCENARIOS.items():
    section(f"3. PIPELINE CHECK: {scenario_name}")
    log(cfg["description"])

    pre = build_preprocessor(use_prior_grades=cfg["use_prior_grades"])

    Xt_train = pre.fit_transform(X_train)   # LEARN from training rows only
    Xt_test = pre.transform(X_test)         # only APPLY to test rows (nothing learned)
    names = get_feature_names(pre)

    # ---- automatic checks -------------------------------------------
    assert Xt_train.shape[1] == Xt_test.shape[1] == len(names), "Column counts do not match!"
    assert not np.isnan(Xt_train).any(), "NaN found in transformed training data!"
    assert not np.isnan(Xt_test).any(), "NaN found in transformed test data!"

    bare_names = [n.split("__", 1)[1] for n in names]
    assert TARGET not in bare_names, "G3 leaked into the model features!"
    if not cfg["use_prior_grades"]:
        forbidden = {"G1", "G2", "grade_trend"}
        assert not forbidden.intersection(bare_names), "Earlier grades leaked into Scenario B!"

    n_num = sum(n.startswith("num__") for n in names)
    n_cat = sum(n.startswith("cat__") for n in names)
    log(f"\nTransformed training shape : {Xt_train.shape}")
    log(f"Transformed test shape     : {Xt_test.shape}")
    log(f"Feature columns            : {len(names)}  ({n_num} numeric + {n_cat} one-hot/binary)")
    log("Automatic checks           : no NaN, column counts match, no G3, no forbidden grades.  OK")

    numeric_names = [n for n in names if n.startswith("num__")]
    categorical_names = [n for n in names if n.startswith("cat__")]
    log("\nNumeric features (scaled):")
    log("  " + ", ".join(numeric_names))
    log("\nCategorical features (encoded):")
    log("  " + ", ".join(categorical_names))

    # ---- proof that the scaler learned from TRAIN only ---------------
    train_df = pd.DataFrame(Xt_train, columns=names)
    test_df = pd.DataFrame(Xt_test, columns=names)
    check_cols = ["num__age", "num__absences_log", "num__studytime"]
    scaling_check = pd.DataFrame({
        "train_mean": train_df[check_cols].mean(),
        "train_std": train_df[check_cols].std(),
        "test_mean": test_df[check_cols].mean(),
        "test_std": test_df[check_cols].std(),
    }).round(3)
    log("\nScaling check (train mean should be about 0 and std about 1;")
    log("test values will NOT be exactly 0 and 1, because the scaler never saw the test rows):")
    log(scaling_check.to_string())

    # ---- can the pipeline handle ONE raw row? (this is what Streamlit will do) ----
    one_row = pre.transform(X_test.iloc[[0]])
    log(f"\nSingle-row test (like a Streamlit input): output shape = {one_row.shape}")
    assert one_row.shape == (1, len(names)), "Single-row transform failed!"

    # ---- save the list of feature names -----------------------------
    feature_table = pd.DataFrame({
        "feature_name": names,
        "type": ["numeric" if n.startswith("num__") else "categorical" for n in names],
    })
    out_path = REPORTS_DIR / f"feature_names_{scenario_name}.csv"
    feature_table.to_csv(out_path, index=False)
    log(f"Feature names saved -> {out_path.relative_to(PROJECT_ROOT)}")

# ----------------------------------------------------------------------
section("4. DONE")
# ----------------------------------------------------------------------
log("The pipelines are built and verified. No model was trained and no data file was changed.")
log(f"Report saved -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")