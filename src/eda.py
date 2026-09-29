"""
eda.py
Exploratory Data Analysis for the Student Performance project.

Reads  : data/processed/student_mat_clean.csv
Saves  : charts in reports/figures/
Prints : summary tables in the terminal
Does NOT modify any data file.

Run from the project root:  python src/eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats  # installed automatically with scikit-learn

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "student_mat_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET = "G3"

if not CLEAN_PATH.exists():
    raise SystemExit(
        "Cleaned data not found. Run 'python src/clean_data.py' first."
    )

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CLEAN_PATH)  # the cleaned file uses commas

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: reports/figures/{name}")


def group_summary(column):
    """Count, mean, median and std of G3 for each value of `column`."""
    return df.groupby(column)[TARGET].agg(
        n="count", mean="mean", median="median", std="std"
    ).round(2)


def box_and_mean_plot(column, xlabel, title, filename):
    """
    Left panel : box plot + individual students (dots) of G3 per group.
    Right panel: average G3 per group, with the group size (n) on top.
    """
    summary = group_summary(column)
    print(summary.to_string())

    # Treat the column as text just for plotting, so the groups are categories
    plot_df = df[[column, TARGET]].copy()
    plot_df[column] = plot_df[column].astype(str)
    order = [str(v) for v in summary.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.boxplot(data=plot_df, x=column, y=TARGET, order=order,
                showfliers=False, ax=axes[0])
    sns.stripplot(data=plot_df, x=column, y=TARGET, order=order,
                  color="black", alpha=0.35, size=3, ax=axes[0])
    axes[0].set_title(f"{title}: distribution")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Final grade (G3)")

    axes[1].bar(order, summary["mean"].values)
    axes[1].axhline(df[TARGET].mean(), color="red", linestyle="--",
                    label=f"overall mean = {df[TARGET].mean():.1f}")
    for i, (mean, n) in enumerate(zip(summary["mean"], summary["n"])):
        axes[1].text(i, mean + 0.2, f"n={n}", ha="center", fontsize=9)
    axes[1].set_title(f"{title}: average G3")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Mean G3")
    axes[1].legend()

    save(fig, filename)


print(f"Loaded cleaned data: {df.shape[0]} rows x {df.shape[1]} columns")


# ======================================================================
# 1. TARGET DISTRIBUTION
# ======================================================================
section("1. TARGET DISTRIBUTION (G3)")
print(df[TARGET].describe().round(2).to_string())
print(f"\nSkewness           : {df[TARGET].skew():.2f}")
n_zero = int((df[TARGET] == 0).sum())
print(f"Rows with G3 == 0  : {n_zero} ({n_zero / len(df):.1%})")
print(f"Rows with G3 >= 10 : {(df[TARGET] >= 10).mean():.1%}  (pass mark is our own assumption)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
panels = [
    (axes[0], df[TARGET], "All students"),
    (axes[1], df.loc[df[TARGET] > 0, TARGET], "Excluding G3 == 0"),
]
for ax, data, title in panels:
    sns.histplot(data, discrete=True, ax=ax)  # one bar per grade value
    ax.axvline(data.mean(), color="red", linestyle="--", label=f"mean = {data.mean():.1f}")
    ax.axvline(data.median(), color="green", linestyle=":", label=f"median = {data.median():.1f}")
    ax.axvline(9.5, color="black", linewidth=1, label="pass mark (G3 >= 10)")
    ax.set_title(title)
    ax.set_xlabel("Final grade (G3)")
    ax.set_ylabel("Number of students")
    ax.legend()
save(fig, "01_target_distribution.png")


# ======================================================================
# 2. STUDY TIME vs FINAL GRADE
# studytime: 1 = <2h, 2 = 2-5h, 3 = 5-10h, 4 = >10h per week
# ======================================================================
section("2. STUDY TIME vs G3")
box_and_mean_plot("studytime", "Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)",
                  "Study time", "02_studytime_vs_g3.png")


# ======================================================================
# 3. ABSENCES vs FINAL GRADE
# ======================================================================
section("3. ABSENCES vs G3")
pearson = df["absences"].corr(df[TARGET])
spearman = df["absences"].corr(df[TARGET], method="spearman")
print(f"Pearson correlation  (straight-line relationship): {pearson:.3f}")
print(f"Spearman correlation (rank / any steady trend)   : {spearman:.3f}")
print(f"Absences: min={df['absences'].min()}  median={df['absences'].median()}  max={df['absences'].max()}")

abs_labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
abs_bins = [-1, 0, 2, 5, 10, 20, np.inf]
abs_group = pd.cut(df["absences"], bins=abs_bins, labels=abs_labels)
abs_summary = df.groupby(abs_group, observed=True)[TARGET].agg(
    n="count", mean="mean", median="median"
).round(2)
print("\nG3 by absence group:")
print(abs_summary.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.regplot(data=df, x="absences", y=TARGET,
            scatter_kws={"alpha": 0.4}, line_kws={"color": "red"}, ax=axes[0])
axes[0].set_title("Absences vs G3 (red = straight-line trend)")
axes[0].set_xlabel("Number of absences")
axes[0].set_ylabel("Final grade (G3)")

binned = pd.DataFrame({"absence_group": abs_group.astype(str), TARGET: df[TARGET]})
sns.boxplot(data=binned, x="absence_group", y=TARGET, order=abs_labels,
            showfliers=False, ax=axes[1])
axes[1].set_title("G3 by absence group")
axes[1].set_xlabel("Number of absences (grouped)")
axes[1].set_ylabel("Final grade (G3)")
save(fig, "03_absences_vs_g3.png")


# ======================================================================
# 4. PREVIOUS FAILURES vs FINAL GRADE
# failures: 0, 1, 2, or 3 (3 means "3 or more")
# ======================================================================
section("4. PREVIOUS FAILURES vs G3")
box_and_mean_plot("failures", "Past class failures (3 = 3 or more)",
                  "Past failures", "04_failures_vs_g3.png")


# ======================================================================
# 5. GENDER vs FINAL GRADE
# ======================================================================
section("5. GENDER (sex) vs G3")
box_and_mean_plot("sex", "Sex (F = female, M = male)",
                  "Sex", "05_sex_vs_g3.png")

female = df.loc[df["sex"] == "F", TARGET]
male = df.loc[df["sex"] == "M", TARGET]
stat, p_value = stats.mannwhitneyu(female, male, alternative="two-sided")
print(f"\nMean difference (M - F): {male.mean() - female.mean():.2f} grade points")
print(f"Mann-Whitney U test p-value: {p_value:.4f}")
print("(A small p-value, e.g. below 0.05, means the difference is unlikely to be pure chance.")
print(" It does NOT tell us WHY the groups differ, and it does not mean the difference is large.)")


# ======================================================================
# 6. RELATIONSHIPS BETWEEN IMPORTANT NUMERICAL VARIABLES
# ======================================================================
section("6. PAIRWISE RELATIONSHIPS")
pair_cols = ["G1", "G2", TARGET, "absences", "studytime", "failures"]
g = sns.pairplot(df[pair_cols], corner=True, diag_kind="hist",
                 plot_kws={"alpha": 0.4, "s": 18})
g.fig.suptitle("Pairwise relationships of important numeric variables", y=1.02)
g.savefig(FIGURES_DIR / "06_pairplot.png", dpi=120)
plt.close("all")
print("\n  Saved: reports/figures/06_pairplot.png")

# Extra: earlier grades vs final grade, with G3 == 0 students highlighted.
# The highlight column exists only for this plot; it is NOT saved to any file.
plot_df = df.copy()
plot_df["G3 is zero"] = plot_df[TARGET] == 0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, ["G1", "G2"]):
    sns.scatterplot(data=plot_df, x=col, y=TARGET, hue="G3 is zero", alpha=0.6, ax=ax)
    ax.set_title(f"{col} vs G3 (G3 = 0 highlighted)")
    ax.set_xlabel(f"{col} (earlier grade)")
    ax.set_ylabel("Final grade (G3)")
save(fig, "06b_earlier_grades_vs_g3.png")


# ======================================================================
# 7. CORRELATION MATRIX
# ======================================================================
section("7. CORRELATION MATRIX")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones(corr.shape, dtype=bool))  # hide the mirrored upper half
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, annot_kws={"size": 7},
            linewidths=0.5, ax=ax)
ax.set_title("Correlation matrix (numeric columns, Pearson)")
save(fig, "07_correlation_matrix.png")

target_corr = corr[TARGET].drop(TARGET).sort_values()
print("\nCorrelation of each numeric column with G3 (weakest to strongest, by sign):")
print(target_corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
colors = np.where(target_corr.values > 0, "tab:blue", "tab:red")
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlation with the final grade (G3)")
ax.set_xlabel("Pearson correlation")
save(fig, "07b_correlation_with_target.png")

# Pairs of input columns that are strongly related to each other (multicollinearity check)
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
strong_pairs = upper[upper.abs() >= 0.7].sort_values(key=abs, ascending=False)
print("\nPairs of columns with |correlation| >= 0.7:")
print(strong_pairs.round(3).to_string() if not strong_pairs.empty else "None")

print("\nEDA finished. Open the PNG files in reports/figures/ to study each chart.")