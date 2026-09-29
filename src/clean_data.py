"""
clean_data.py
Data-cleaning step for the Student Performance project.

What this script does:
  - Checks missing values, duplicates, data types, unique values,
    invalid values and outliers.
  - NEVER silently deletes rows. Every suspicious row is written to
    reports/flagged_rows.csv and every check is written to
    reports/cleaning_log.txt.
  - Saves the cleaned dataset to data/processed/student_mat_clean.csv

Run from the project root:  python src/clean_data.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots to files only (avoids pop-up window errors)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw"/ "student-mat.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "student_mat_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOG_PATH = REPORTS_DIR / "cleaning_log.txt"
FLAGS_PATH = REPORTS_DIR / "flagged_rows.csv"
DUPLICATES_PATH = REPORTS_DIR / "removed_duplicates.csv"

TARGET = "G3"

# Duplicates are only removed if you deliberately change this to True.
# Default is False because the dataset has no student ID, so two identical
# rows could be two different real students.
DROP_DUPLICATES = False

YES_NO = {"yes", "no"}
JOBS = {"teacher", "health", "services", "at_home", "other"}

# Allowed categories (taken from the UCI dataset documentation)
ALLOWED_CATEGORIES = {
    "school": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"U", "R"},
    "famsize": {"LE3", "GT3"},
    "Pstatus": {"T", "A"},
    "Mjob": JOBS,
    "Fjob": JOBS,
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": YES_NO,
    "famsup": YES_NO,
    "paid": YES_NO,
    "activities": YES_NO,
    "nursery": YES_NO,
    "higher": YES_NO,
    "internet": YES_NO,
    "romantic": YES_NO,
}

# Allowed numeric ranges (min, max), from the same documentation
NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 3),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}

# Outlier screening only makes sense for columns that behave like quantities.
# Codes such as 1-5 ratings are not screened.
OUTLIER_COLS = ["age", "absences", "G1", "G2", "G3"]

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
log_lines = []  # everything we print is also stored here, then saved to a file
flags = []      # every suspicious value is recorded here


def log(text=""):
    """Print a message and keep a copy for the log file."""
    print(text)
    log_lines.append(str(text))


def section(title):
    log("\n" + "=" * 64)
    log(title)
    log("=" * 64)


def add_flag(row_index, column, issue, value=None):
    """Record one suspicious value. Nothing is deleted."""
    flags.append(
        {
            "row_index": row_index,          # index inside the pandas DataFrame
            "csv_line": row_index + 2,       # line number if you open the CSV (header = line 1)
            "column": column,
            "issue": issue,
            "value": value,
        }
    )


# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
if not RAW_PATH.exists():
    raise SystemExit("Raw dataset NOT found. Expected: data/raw/student-mat.csv")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(RAW_PATH, sep=";")
df = raw.copy()  # we clean the COPY; the raw file on disk is never touched
n_raw = len(df)

section("LOADED RAW DATA")
log(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

expected_cols = set(NUMERIC_RANGES) | set(ALLOWED_CATEGORIES)
missing_cols = expected_cols - set(df.columns)
extra_cols = set(df.columns) - expected_cols
log(f"Expected columns missing from file : {sorted(missing_cols) if missing_cols else 'None'}")
log(f"Unexpected extra columns in file   : {sorted(extra_cols) if extra_cols else 'None'}")
if missing_cols:
    raise SystemExit("Some expected columns are missing. Check that you downloaded student-mat.csv.")

# ----------------------------------------------------------------------
# STEP 0: STANDARDISE TEXT (trim spaces, treat empty text as missing)
# Why: "yes " and "yes" would be seen as two different categories, and an
# empty string hides a missing value from isna().
# ----------------------------------------------------------------------
section("STEP 0: TEXT STANDARDISATION")
text_cols = df.select_dtypes(exclude="number").columns.tolist()
n_trimmed = 0
n_empty = 0
for col in text_cols:
    stripped = df[col].str.strip()
    n_trimmed += int((stripped.fillna("") != df[col].fillna("")).sum())
    df[col] = stripped
    empty_mask = df[col] == ""
    n_empty += int(empty_mask.sum())
    df.loc[empty_mask, col] = pd.NA

log(f"Text cells that had leading/trailing spaces (fixed): {n_trimmed}")
log(f"Empty text cells converted to missing              : {n_empty}")

# ----------------------------------------------------------------------
# STEP 1: MISSING VALUES
# Why: many models cannot handle NaN. We only REPORT here. If anything is
# missing, imputation must be done inside the modelling pipeline (Step 4)
# so that it is learned from training data only.
# ----------------------------------------------------------------------
section("STEP 1: MISSING VALUES")
missing_per_col = df.isna().sum()
missing_per_col = missing_per_col[missing_per_col > 0]

if missing_per_col.empty:
    log("No missing values found. Nothing to fill and nothing to drop.")
else:
    log("Missing values per column:")
    log(missing_per_col.to_string())
    for col in missing_per_col.index:
        for idx in df.index[df[col].isna()]:
            add_flag(idx, col, "missing value")
    log("Rows were flagged, NOT deleted or filled. Decision is postponed to Step 4.")

# ----------------------------------------------------------------------
# STEP 2: DUPLICATE ROWS
# Why: exact duplicates can appear in both train and test sets and make
# results look better than they are. But without a student ID we cannot be
# sure they are true duplicates, so we look before deciding.
# ----------------------------------------------------------------------
section("STEP 2: DUPLICATE ROWS")
n_extra_copies = int(df.duplicated().sum())
log(f"Duplicate rows (extra copies of an earlier row): {n_extra_copies}")

if n_extra_copies == 0:
    log("No duplicates found.")
else:
    in_dup_group = df.duplicated(keep=False)
    for idx in df.index[in_dup_group]:
        add_flag(idx, "ALL", "duplicate row (part of a duplicate group)")
    log("Rows involved in duplicate groups (first 10 shown):")
    log(df[in_dup_group].head(10).to_string())

    if DROP_DUPLICATES:
        removed = df[df.duplicated()]
        removed.to_csv(DUPLICATES_PATH, index=True)
        df = df.drop_duplicates()
        log(f"DROP_DUPLICATES=True: removed {len(removed)} rows. Copy saved to {DUPLICATES_PATH.name}")
    else:
        log("DROP_DUPLICATES=False: duplicates were flagged and KEPT.")

# ----------------------------------------------------------------------
# STEP 3: DATA TYPES
# Why: numbers stored as text cannot be used in calculations or models.
# We convert only if needed and we log every value that fails to convert.
# ----------------------------------------------------------------------
section("STEP 3: DATA TYPES")
for col in NUMERIC_RANGES:
    if pd.api.types.is_numeric_dtype(df[col]):
        continue
    converted = pd.to_numeric(df[col], errors="coerce")
    failed = converted.isna() & df[col].notna()
    log(f"Column '{col}' was not numeric. Converting. Values that failed: {int(failed.sum())}")
    for idx in df.index[failed]:
        add_flag(idx, col, "could not convert to number", df.at[idx, col])
    df[col] = converted

not_numeric = [c for c in NUMERIC_RANGES if not pd.api.types.is_numeric_dtype(df[c])]
log(f"Numeric columns that are still not numeric: {not_numeric if not_numeric else 'None'}")

log("\nData types now:")
log(df.dtypes.astype(str).value_counts().to_string())
log("\nNote: categorical columns stay as text. Encoding them is done in Step 4.")

# ----------------------------------------------------------------------
# STEP 4: UNIQUE VALUES
# Why: reveals typos, unexpected categories, and constant columns.
# ----------------------------------------------------------------------
section("STEP 4: UNIQUE VALUES")
log("Categorical columns: values found (unexpected values are marked):")
for col, allowed in ALLOWED_CATEGORIES.items():
    found = set(df[col].dropna().unique())
    unexpected = found - allowed
    log(f"  {col:11s} {sorted(found)}" + (f"   <-- UNEXPECTED: {sorted(unexpected)}" if unexpected else ""))
    if unexpected:
        for idx in df.index[df[col].isin(unexpected)]:
            add_flag(idx, col, "unexpected category", df.at[idx, col])

log("\nNumeric columns: number of unique values and observed min / max:")
for col in NUMERIC_RANGES:
    log(f"  {col:11s} unique={df[col].nunique():3d}   min={df[col].min()}   max={df[col].max()}")

constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
log(f"\nConstant columns (no information): {constant_cols if constant_cols else 'None'}")

# ----------------------------------------------------------------------
# STEP 5: INVALID VALUES
# Why: values outside the documented range are impossible and would mislead
# a model. Ordinal codes must also be whole numbers.
# ----------------------------------------------------------------------
section("STEP 5: INVALID VALUES")
n_invalid = 0
for col, (low, high) in NUMERIC_RANGES.items():
    out_of_range = df[col].notna() & ((df[col] < low) | (df[col] > high))
    not_whole = df[col].notna() & (df[col] % 1 != 0)
    for idx in df.index[out_of_range]:
        add_flag(idx, col, f"outside allowed range {low}-{high}", df.at[idx, col])
    for idx in df.index[not_whole]:
        add_flag(idx, col, "not a whole number", df.at[idx, col])
    count = int(out_of_range.sum() + not_whole.sum())
    n_invalid += count
    if count:
        log(f"  {col}: {int(out_of_range.sum())} out of range, {int(not_whole.sum())} not whole numbers")

if n_invalid == 0:
    log("All numeric values are inside the documented ranges and are whole numbers.")
else:
    log(f"Total invalid numeric values flagged (NOT deleted): {n_invalid}")

# ----------------------------------------------------------------------
# STEP 6: OUTLIERS (IQR rule, screening only)
# Why: extreme values can distort some models. The IQR rule marks values
# beyond 1.5 x IQR from the quartiles. A flagged value is a candidate to
# LOOK AT, not automatically an error. We keep all of them.
# ----------------------------------------------------------------------
section("STEP 6: OUTLIERS (IQR rule, nothing removed)")
for col in OUTLIER_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    log(f"  {col:9s} bounds=[{lower:.2f}, {upper:.2f}]   outliers={int(mask.sum())}")
    if mask.any():
        log(f"            outlier values range: {df.loc[mask, col].min()} to {df.loc[mask, col].max()}")
    for idx in df.index[mask]:
        add_flag(idx, col, "possible outlier (IQR rule)", df.at[idx, col])

fig, axes = plt.subplots(1, len(OUTLIER_COLS), figsize=(15, 4))
for ax, col in zip(axes, OUTLIER_COLS):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "outlier_boxplots.png", dpi=150)
plt.close(fig)
log(f"\nBox plots saved to: reports/figures/outlier_boxplots.png")

# ----------------------------------------------------------------------
# STEP 7: ZERO FINAL GRADES (special case)
# Why: G3 == 0 may mean "did not take the final exam" rather than "scored 0".
# We only investigate. We do NOT add a helper column made from G3 to the
# dataset, because that would leak the target into the features.
# ----------------------------------------------------------------------
section("STEP 7: ZERO FINAL GRADES (G3 == 0)")
zero_mask = df[TARGET] == 0
log(f"Rows with {TARGET} == 0: {int(zero_mask.sum())} of {len(df)}")
if zero_mask.any():
    log("\nSummary of G1, G2 and absences for those rows:")
    log(df.loc[zero_mask, ["G1", "G2", "absences"]].describe().round(2).to_string())
    log(f"\nOf these, rows with G2 == 0      : {int((df.loc[zero_mask, 'G2'] == 0).sum())}")
    log(f"Of these, rows with absences == 0: {int((df.loc[zero_mask, 'absences'] == 0).sum())}")
    for idx in df.index[zero_mask]:
        add_flag(idx, TARGET, "final grade is 0 (possible missed exam)", 0)
    log("\nThese rows were flagged and KEPT. We decide how to handle them after EDA (Step 3).")

# ----------------------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------------------
section("SAVING RESULTS")
flags_df = pd.DataFrame(
    flags, columns=["row_index", "csv_line", "column", "issue", "value"]
)
flags_df.to_csv(FLAGS_PATH, index=False)

# The cleaned file uses a COMMA separator (the raw file used semicolons).
df.to_csv(CLEAN_PATH, index=False)

# Safety check: reload the saved file and confirm the shape matches
check = pd.read_csv(CLEAN_PATH)
assert check.shape == df.shape, "Saved file does not match the cleaned DataFrame!"

if not DROP_DUPLICATES:
    assert len(df) == n_raw, "Rows were lost even though no deletion was requested!"

log(f"Rows in raw file         : {n_raw}")
log(f"Rows in cleaned file     : {len(df)}")
log(f"Rows removed             : {n_raw - len(df)}")
log(f"Total flags recorded     : {len(flags_df)}")
log(f"Rows with at least 1 flag: {flags_df['row_index'].nunique() if not flags_df.empty else 0}")
if not flags_df.empty:
    log("\nFlags by issue:")
    log(flags_df["issue"].value_counts().to_string())

log(f"\nCleaned data  -> {CLEAN_PATH.relative_to(PROJECT_ROOT)}")
log(f"Flagged rows  -> {FLAGS_PATH.relative_to(PROJECT_ROOT)}")
log(f"Cleaning log  -> {LOG_PATH.relative_to(PROJECT_ROOT)}")

LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")
print("\nDone.")
