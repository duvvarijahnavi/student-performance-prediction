"""
visualize.py
Publication-quality charts for the Student Performance project.

Reads  : data/processed/student_mat_clean.csv
Saves  : PNG charts in reports/figures/eda/  (plus figure_index.csv)
Does NOT modify any data file. Charts are saved, never displayed.

Run from the project root:  python src/visualize.py
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # "Agg" draws to files only, so no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "student_mat_clean.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"   # change here if you want another folder
INDEX_PATH = OUTPUT_DIR / "figure_index.csv"

TARGET = "G3"
PASS_MARK = 10   # OUR assumption (not part of the dataset). State it in the README.
DPI = 300
SOURCE_NOTE = "Data: UCI Student Performance (Math), Cortez & Silva (2008)"

BLUE, ORANGE, RED = "#4C72B0", "#DD8452", "#C44E52"
LIGHT_BLUE = "#9ecae1"

# Readable names for the chart labels
LABELS = {
    "school": "School (GP or MS)",
    "sex": "Sex",
    "address": "Home address (U = urban, R = rural)",
    "famsize": "Family size (LE3 = up to 3, GT3 = over 3)",
    "Pstatus": "Parents living together (T) or apart (A)",
    "schoolsup": "Extra school support",
    "famsup": "Family educational support",
    "paid": "Paid extra classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing the school",
    "guardian": "Main guardian",
    "Medu": "Mother's education (0-4)",
    "Fedu": "Father's education (0-4)",
    "traveltime": "Travel time to school (1-4)",
    "famrel": "Family relationship quality (1-5)",
    "freetime": "Free time after school (1-5)",
    "goout": "Going out with friends (1-5)",
    "Dalc": "Weekday alcohol use (1-5)",
    "Walc": "Weekend alcohol use (1-5)",
    "health": "Current health status (1-5)",
    "age": "Age (years)",
}

BINARY_COLS = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup",
               "paid", "activities", "nursery", "higher", "internet", "romantic"]
NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]
ORDINAL_COLS = ["Medu", "Fedu", "traveltime", "famrel", "freetime", "goout",
                "Dalc", "Walc", "health", "age"]

# ----------------------------------------------------------------------
# LOAD AND VALIDATE
# ----------------------------------------------------------------------
if not CLEAN_PATH.exists():
    raise SystemExit("Cleaned data not found. Run 'python src/clean_data.py' first.")

df = pd.read_csv(CLEAN_PATH)  # the cleaned file uses commas

required = [TARGET, "G1", "G2", "absences", "studytime", "failures", "sex"]
required += BINARY_COLS + NOMINAL_COLS + ORDINAL_COLS
missing_columns = sorted(set(required) - set(df.columns))
if missing_columns:
    raise SystemExit(f"These expected columns are missing from the data: {missing_columns}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent look for every chart. set_theme must come BEFORE rcParams.update.
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})

saved_figures = []  # filled by save_figure(); becomes figure_index.csv


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def save_figure(fig, filename, description, note=None):
    """Add a source footer, save the figure as PNG, close it, and record it."""
    fig.tight_layout(rect=[0, 0.04, 1, 1])  # leave room at the bottom for the footer
    fig.text(0.99, 0.008, f"{SOURCE_NOTE}, n = {len(df)}",
             ha="right", va="bottom", fontsize=8, color="dimgray")
    if note:
        fig.text(0.01, 0.008, note, ha="left", va="bottom", fontsize=8, color="dimgray")
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)  # free memory
    saved_figures.append({"filename": filename, "description": description})
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


def grade_by_group(groups, xlabel, title, filename, description,
                   order=None, tick_names=None):
    """
    Box plot of G3 for each group, with every student as a dot, the group mean
    as a red diamond, and the group size (n) in the axis label.
    `groups` is a column (Series) with the same rows as df.
    """
    tick_names = tick_names or {}
    data = pd.DataFrame({"group": groups.astype(object), TARGET: df[TARGET]})
    counts = data["group"].value_counts()
    means = data.groupby("group")[TARGET].mean()

    if order is None:
        order = sorted(data["group"].dropna().unique())
    order = [g for g in order if g in counts.index]  # keep only groups that exist

    label_map = {g: f"{tick_names.get(g, g)}\n(n={counts[g]})" for g in order}
    data["label"] = data["group"].map(label_map)
    label_order = [label_map[g] for g in order]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.boxplot(data=data, x="label", y=TARGET, order=label_order,
                color=LIGHT_BLUE, showfliers=False, ax=ax)
    sns.stripplot(data=data, x="label", y=TARGET, order=label_order,
                  color="black", alpha=0.3, size=3, jitter=0.2, ax=ax)
    ax.scatter(range(len(order)), [means[g] for g in order], marker="D", s=70,
               color=RED, edgecolor="white", zorder=10, label="Group mean")
    ax.axhline(df[TARGET].mean(), color="gray", linestyle="--", linewidth=1.2,
               label=f"Overall mean = {df[TARGET].mean():.2f}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final grade (G3, scale 0-20)")
    ax.set_ylim(-1, 24)               # extra headroom for the legend
    ax.set_yticks(range(0, 21, 2))
    ax.legend(loc="upper center", ncol=2)
    save_figure(fig, filename, description, note="Box = middle 50% of students; dots = individual students")


def mean_grade_grid(columns, title, filename, description, ncols):
    """
    Small-multiples chart: for each column, the mean G3 per category as bars,
    with approximate 95% confidence-interval whiskers and the group size (n).
    """
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    overall = df[TARGET].mean()

    for ax, col in zip(axes_flat, columns):
        stats = df.groupby(col)[TARGET].agg(["mean", "std", "count"])
        ci = 1.96 * (stats["std"] / np.sqrt(stats["count"])).fillna(0)  # about 95% interval
        x = np.arange(len(stats))
        ax.bar(x, stats["mean"], yerr=ci, capsize=3, color=BLUE, edgecolor="white")
        ax.axhline(overall, color=RED, linestyle="--", linewidth=1)
        for xi, m, e, n in zip(x, stats["mean"], ci, stats["count"]):
            ax.text(xi, m + e + 0.3, f"n={n}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in stats.index])
        ax.set_ylim(0, 20)
        ax.set_title(textwrap.fill(LABELS.get(col, col), 34), fontsize=11)
        ax.set_xlabel(f"Value of '{col}'")
        ax.set_ylabel("Mean G3")

    for ax in axes_flat[len(columns):]:
        ax.set_visible(False)  # hide unused panels

    fig.suptitle(title)
    save_figure(fig, filename, description,
                note="Bars = mean G3; whiskers = approx. 95% CI; red dashed = overall mean")


# ----------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------
def chart_target_distribution():
    counts = df[TARGET].value_counts().reindex(range(0, 21), fill_value=0)
    colours = [RED if g == 0 else ORANGE if g < PASS_MARK else BLUE for g in counts.index]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(counts.index, counts.values, color=colours, edgecolor="white")
    for grade, count in counts.items():
        if count > 0:
            ax.text(grade, count + 0.4, str(count), ha="center", va="bottom", fontsize=8)

    ax.axvline(df[TARGET].mean(), color="black", linestyle="--", linewidth=1.3,
               label=f"Mean = {df[TARGET].mean():.2f}")
    ax.axvline(df[TARGET].median(), color="dimgray", linestyle=":", linewidth=1.8,
               label=f"Median = {df[TARGET].median():.1f}")
    ax.axvline(PASS_MARK - 0.5, color="green", linewidth=1.3,
               label=f"Pass boundary (G3 >= {PASS_MARK})")

    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color=RED, label="G3 = 0"),
                Patch(color=ORANGE, label=f"G3 = 1 to {PASS_MARK - 1}"),
                Patch(color=BLUE, label=f"G3 >= {PASS_MARK}")]
    ax.legend(handles=handles, loc="upper right")

    ax.set_title("Distribution of final grades (G3)")
    ax.set_xlabel("Final grade (G3, scale 0-20)")
    ax.set_ylabel("Number of students")
    ax.set_xticks(range(0, 21))
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "01_g3_distribution.png",
                "Histogram of the final grade G3 with mean, median and pass boundary")


def chart_pass_fail_balance():
    passed = int((df[TARGET] >= PASS_MARK).sum())
    failed = len(df) - passed

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bars = ax.bar([f"Fail\n(G3 < {PASS_MARK})", f"Pass\n(G3 >= {PASS_MARK})"],
                  [failed, passed], color=[ORANGE, BLUE], width=0.55)
    for bar, n in zip(bars, [failed, passed]):
        ax.text(bar.get_x() + bar.get_width() / 2, n + len(df) * 0.01,
                f"{n} ({n / len(df):.1%})", ha="center", va="bottom", fontsize=11)
    ax.set_title("Pass / fail balance (classification target)")
    ax.set_xlabel(f"Result (pass mark of {PASS_MARK} is our own assumption)")
    ax.set_ylabel("Number of students")
    ax.set_ylim(0, max(failed, passed) * 1.15)
    save_figure(fig, "02_pass_fail_balance.png",
                "Number of students below and above the assumed pass mark")


def chart_absences_scatter():
    pearson = df["absences"].corr(df[TARGET])
    spearman = df["absences"].corr(df[TARGET], method="spearman")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    sns.regplot(data=df, x="absences", y=TARGET, scatter_kws={"alpha": 0.45, "s": 30},
                line_kws={"color": RED, "linewidth": 2}, ax=ax)
    ax.text(0.98, 0.05, f"Pearson r = {pearson:.2f}\nSpearman rho = {spearman:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_title("Absences vs final grade (G3)")
    ax.set_xlabel("Number of school absences")
    ax.set_ylabel("Final grade (G3, scale 0-20)")
    save_figure(fig, "04_absences_vs_g3_scatter.png",
                "Scatter plot of absences against G3 with a straight-line trend",
                note="Red line = straight-line trend with confidence band")


def chart_absence_groups():
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    bins = [-1, 0, 2, 5, 10, 20, np.inf]
    groups = pd.cut(df["absences"], bins=bins, labels=labels)
    grade_by_group(groups, "Number of absences (grouped)",
                   "Final grade (G3) by number of absences",
                   "05_g3_by_absence_group.png",
                   "Box plot of G3 for grouped absence counts", order=labels)


def chart_earlier_grades():
    rng = np.random.default_rng(42)  # fixed seed -> identical chart every run
    zero = df[TARGET] == 0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax, col in zip(axes, ["G1", "G2"]):
        x_jit = df[col] + rng.normal(0, 0.12, len(df))   # jitter is for display only
        y_jit = df[TARGET] + rng.normal(0, 0.12, len(df))
        ax.scatter(x_jit[~zero], y_jit[~zero], alpha=0.5, s=28, color=BLUE, label="G3 > 0")
        ax.scatter(x_jit[zero], y_jit[zero], alpha=0.85, s=38, color=RED, label="G3 = 0")
        ax.plot([0, 20], [0, 20], color="gray", linestyle="--", label="G3 equal to earlier grade")
        r = df[col].corr(df[TARGET])
        ax.set_title(f"{col} vs final grade (Pearson r = {r:.2f})")
        ax.set_xlabel(f"{col} (earlier grade, scale 0-20)")
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Final grade (G3, scale 0-20)")
    fig.suptitle("Earlier grades vs final grade")
    save_figure(fig, "08_earlier_grades_vs_g3.png",
                "G1 and G2 against G3, with G3 = 0 students highlighted",
                note="A small random jitter is added so overlapping points are visible")


def chart_correlation_matrix():
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool))  # hide the mirrored upper half

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                center=0, annot_kws={"size": 7}, linewidths=0.5,
                cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set_title("Correlation matrix of numeric columns")
    save_figure(fig, "09_correlation_matrix.png",
                "Pearson correlation heatmap of all numeric columns",
                note="Ordinal codes (1-5 ratings) are treated as numbers: rough guide only")

    target_corr = corr[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = [BLUE if v > 0 else RED for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colours)
    for y, v in enumerate(target_corr.values):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of each numeric column with the final grade (G3)")
    ax.set_xlabel("Pearson correlation with G3")
    ax.set_ylabel("Column")
    ax.margins(x=0.12)
    save_figure(fig, "10_correlation_with_g3.png",
                "Ranked Pearson correlation of each numeric column with G3",
                note="Blue = positive, red = negative. Correlation is not causation.")


def chart_zero_grade_diagnostic():
    zero = df[TARGET] == 0
    n_zero = int(zero.sum())
    if n_zero == 0:
        print("  Skipped 14_zero_grade_diagnostic.png: no rows with G3 = 0 in this dataset.")
        return

    zero_label = f"G3 = 0\n(n={n_zero})"
    other_label = f"G3 > 0\n(n={len(df) - n_zero})"
    data = df.copy()  # this helper column exists only for this plot, never saved
    data["group"] = np.where(zero, zero_label, other_label)
    order = [zero_label, other_label]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, col in zip(axes, ["G1", "G2", "absences"]):
        sns.boxplot(data=data, x="group", y=col, order=order, color=LIGHT_BLUE,
                    showfliers=False, ax=ax)
        sns.stripplot(data=data, x="group", y=col, order=order, color="black",
                      alpha=0.3, size=3, jitter=0.2, ax=ax)
        ax.set_title(f"{col}: G3 = 0 vs the rest")
        ax.set_xlabel("Group")
        ax.set_ylabel(LABELS.get(col, f"{col} (scale 0-20)" if col != "absences" else "Number of absences"))
    fig.suptitle("Do students with a final grade of 0 look different?")
    save_figure(fig, "14_zero_grade_diagnostic.png",
                "Comparison of G1, G2 and absences for G3 = 0 students vs the rest",
                note="Diagnostic only: G3 is used to form groups, never as a model feature")


# ----------------------------------------------------------------------
# RUN EVERYTHING
# ----------------------------------------------------------------------
print(f"Loaded cleaned data: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Saving charts to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}\n")

chart_target_distribution()
chart_pass_fail_balance()

grade_by_group(df["studytime"], "Weekly study time",
               "Final grade (G3) by weekly study time", "03_g3_by_studytime.png",
               "Box plot of G3 for each study-time category",
               tick_names={1: "< 2 h", 2: "2-5 h", 3: "5-10 h", 4: "> 10 h"})

chart_absences_scatter()
chart_absence_groups()

grade_by_group(df["failures"], "Number of past class failures",
               "Final grade (G3) by number of past class failures", "06_g3_by_past_failures.png",
               "Box plot of G3 for each past-failures category",
               tick_names={0: "0", 1: "1", 2: "2", 3: "3 or more"})

grade_by_group(df["sex"], "Sex",
               "Final grade (G3) by sex", "07_g3_by_sex.png",
               "Box plot of G3 for female and male students",
               tick_names={"F": "Female", "M": "Male"})

chart_earlier_grades()
chart_correlation_matrix()

mean_grade_grid(BINARY_COLS, "Mean final grade (G3) by two-value features",
                "11_mean_g3_binary_features.png",
                "Mean G3 for each yes/no or two-category feature", ncols=4)
mean_grade_grid(NOMINAL_COLS, "Mean final grade (G3) by job, reason and guardian",
                "12_mean_g3_nominal_features.png",
                "Mean G3 for the multi-category text features", ncols=2)
mean_grade_grid(ORDINAL_COLS, "Mean final grade (G3) by rating and education features",
                "13_mean_g3_ordinal_features.png",
                "Mean G3 for the ordinal (1-5 style) features and age", ncols=4)

chart_zero_grade_diagnostic()

# ----------------------------------------------------------------------
# FIGURE INDEX + VERIFICATION
# ----------------------------------------------------------------------
index_df = pd.DataFrame(saved_figures)
index_df.to_csv(INDEX_PATH, index=False)

for row in saved_figures:
    file_path = OUTPUT_DIR / row["filename"]
    assert file_path.exists() and file_path.stat().st_size > 0, f"{row['filename']} was not saved!"

print(f"\nDone. {len(saved_figures)} charts saved and verified.")
print(f"Index of charts -> {INDEX_PATH.relative_to(PROJECT_ROOT)}")