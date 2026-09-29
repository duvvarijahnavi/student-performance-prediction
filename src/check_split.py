"""
check_split.py
Creates the train/test split used by the whole project, prints its shapes,
and runs automatic leakage checks. It trains no model and changes no data file.

Saves: reports/split_report.txt

Run from the project root:  python src/check_split.py
"""

import pandas as pd

from features import (
    PASS_MARK,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    load_clean_data,
    make_pass_fail,
    split_data,
)

# The decision about G3 == 0 rows is still open, so nothing is removed.
# IMPORTANT: use the SAME value in every script that loads the data.
DROP_ZERO_GRADES = False

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "split_report.txt"

pd.set_option("display.width", 120)

log_lines = []  # everything printed is also collected here for the report file


def log(text=""):
    """Print a message and keep a copy for the report file."""
    print(text)
    log_lines.append(str(text))


def section(title):
    log("\n" + "=" * 64)
    log(title)
    log("=" * 64)


# ----------------------------------------------------------------------
section("1. LOAD THE CLEANED DATA")
# ----------------------------------------------------------------------
df = load_clean_data(drop_zero_grades=DROP_ZERO_GRADES)
log(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}   (DROP_ZERO_GRADES = {DROP_ZERO_GRADES})")

# ----------------------------------------------------------------------
section("2. SEPARATE X (INPUTS) AND y (TARGET)")
# ----------------------------------------------------------------------
X = df.drop(columns=[TARGET])   # every column EXCEPT the final grade
y = df[TARGET]                  # only the final grade

log(f"Target column        : {TARGET}  (final grade, whole numbers from 0 to 20)")
log(f"X shape (all rows)   : {X.shape}")
log(f"y shape (all rows)   : {y.shape}")
log(f"Is '{TARGET}' inside X? {TARGET in X.columns}   (must be False)")
assert TARGET not in X.columns, "The target is inside X. That is data leakage!"

# ----------------------------------------------------------------------
section(f"3. SPLIT: {int((1 - TEST_SIZE) * 100)}% TRAIN / {int(TEST_SIZE * 100)}% TEST (random_state={RANDOM_STATE})")
# ----------------------------------------------------------------------
# split_data() lives in features.py and is used by every script in the project.
X_train, X_test, y_train, y_test = split_data(df)

log(f"X_train shape: {X_train.shape}")
log(f"X_test  shape: {X_test.shape}")
log(f"y_train shape: {y_train.shape}")
log(f"y_test  shape: {y_test.shape}")
log(f"Share of rows in test set: {len(X_test) / len(df):.1%}")

# ----------------------------------------------------------------------
section("4. AUTOMATIC LEAKAGE AND SANITY CHECKS")
# ----------------------------------------------------------------------
# Check 1: no student was lost or duplicated by the split
assert len(X_train) + len(X_test) == len(df), "Rows were lost or duplicated!"
assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "X and y sizes differ!"
log("Check 1: train rows + test rows = total rows.                       OK")

# Check 2: no row appears in both sets (the row labels must be disjoint)
assert set(X_train.index).isdisjoint(set(X_test.index)), "Some rows are in BOTH sets!"
log("Check 2: no row appears in both train and test.                    OK")

# Check 3: each X row is still paired with its own y value
assert X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index), "X and y are misaligned!"
log("Check 3: every X row is still matched with its own y value.        OK")

# Check 4: the target is not among the inputs of either set
assert TARGET not in X_train.columns and TARGET not in X_test.columns, "G3 leaked into X!"
log(f"Check 4: '{TARGET}' is not among the input columns of either set.       OK")

# Check 5: identical-looking students in both sets (information, not an error).
# The dataset has no student ID, so identical rows might be two real students.
train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
test_hashes = pd.util.hash_pandas_object(X_test, index=False)
n_shared = int(test_hashes.isin(train_hashes).sum())
log(f"Check 5: test rows with an identical input row in train: {n_shared}   (informational)")

# Check 6: the split is reproducible (running it again gives the same rows)
X_train_again, X_test_again, _, _ = split_data(df)
assert X_train.index.equals(X_train_again.index) and X_test.index.equals(X_test_again.index), \
    "The split changed between two runs!"
log("Check 6: running the split a second time gives identical rows.     OK")

# ----------------------------------------------------------------------
section("5. DO TRAIN AND TEST LOOK SIMILAR? (target G3)")
# ----------------------------------------------------------------------
summary = pd.DataFrame({"train": y_train.describe(), "test": y_test.describe()}).round(2)
log(summary.to_string())

log(f"\nShare passing (G3 >= {PASS_MARK}, our own pass mark):")
log(f"  all data : {make_pass_fail(y).mean():.1%}")
log(f"  train    : {make_pass_fail(y_train).mean():.1%}")
log(f"  test     : {make_pass_fail(y_test).mean():.1%}")

log("\nRows with G3 == 0:")
log(f"  all data : {int((y == 0).sum())}")
log(f"  train    : {int((y_train == 0).sum())}")
log(f"  test     : {int((y_test == 0).sum())}")

# ----------------------------------------------------------------------
section("6. DONE")
# ----------------------------------------------------------------------
log("The split is created and verified. Nothing was fitted and no data file was changed.")
log("Reminder: the test set stays untouched until the final evaluation.")
log(f"Report saved -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")
REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")