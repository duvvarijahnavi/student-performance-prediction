"""
understand_data.py
READ-ONLY inspection of the raw dataset.
This script does NOT modify or save any data.

Run from the project root:  python src/understand_data.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" /  "student-mat.csv"
TARGET = "G3"

if not DATA_PATH.exists():
    raise SystemExit("Dataset NOT found. Put student-mat.csv inside data/raw/")

df = pd.read_csv(DATA_PATH, sep=";")

# Show all columns and use a wider console
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

# ---------------------------------------------------------------
print("=" * 60)
print("1. SHAPE")
print("=" * 60)
print("Rows, columns:", df.shape)

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("2. NUMERIC vs CATEGORICAL (as detected by pandas)")
print("=" * 60)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}):")
print(numeric_cols)
print(f"\nCategorical columns ({len(categorical_cols)}):")
print(categorical_cols)

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("3. COLUMN SUMMARY (type, unique values, missing values, examples)")
print("=" * 60)
summary = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "example_values": [df[c].unique()[:5].tolist() for c in df.columns],
    }
)
print(summary)

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("4. DATA QUALITY CHECKS")
print("=" * 60)
print("Total missing values      :", int(df.isna().sum().sum()))
print("Duplicate rows            :", int(df.duplicated().sum()))
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
print("Constant columns          :", constant_cols if constant_cols else "None")
print(f"Rows where {TARGET} == 0       :", int((df[TARGET] == 0).sum()))

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("5. TARGET SUMMARY")
print("=" * 60)
print(df[TARGET].describe())

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("6. CATEGORY COUNTS (spot rare categories)")
print("=" * 60)
for col in categorical_cols:
    print(f"{col:12s}", df[col].value_counts().to_dict())

# ---------------------------------------------------------------
print("\n" + "=" * 60)
print(f"7. CORRELATION OF NUMERIC COLUMNS WITH {TARGET}")
print("=" * 60)
print("(A first look only. Correlation is not proof of cause.)\n")
corr_with_target = (
    df[numeric_cols]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)
print(corr_with_target.round(3))

print("\nDone. No data was modified or saved.")