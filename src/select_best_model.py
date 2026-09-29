"""
select_best_model.py
Reads reports/model_results.csv (made by train_models.py), applies a selection
rule that was fixed BEFORE looking at results, and draws comparison charts.
It trains nothing and changes no data file.

Saves:
  reports/best_model_selection.csv
  reports/best_model_report.txt
  reports/figures/models/best_model_cv_rmse.png
  reports/figures/models/best_model_metric_grid.png
  reports/figures/models/best_model_train_cv_test.png

Run from the project root:  python src/select_best_model.py
"""

import textwrap

import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from features import PROJECT_ROOT, SCENARIOS
from train_models import BASELINE_NAME, CV_FOLDS

# ----------------------------------------------------------------------
# SETTINGS (the rule is fixed here, before looking at any result)
# ----------------------------------------------------------------------
USE_ONE_SE_RULE = True
COMPLEXITY_ORDER = [
    "Linear Regression", "Ridge Regression", "Decision Tree",
    "Random Forest", "Gradient Boosting",
]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "models"
RESULTS_PATH = REPORTS_DIR / "model_results.csv"
SELECTION_PATH = REPORTS_DIR / "best_model_selection.csv"
REPORT_PATH = REPORTS_DIR / "best_model_report.txt"

GREEN, BLUE, GRAY = "#55A868", "#4C72B0", "#B0B0B0"

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", None)

log_lines = []


def log(text=""):
    """Print a message and keep a copy for the report file."""
    print(text)
    log_lines.append(str(text))


def section(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ----------------------------------------------------------------------
# LOAD AND VALIDATE
# ----------------------------------------------------------------------
if not RESULTS_PATH.exists():
    raise SystemExit("reports/model_results.csv not found. Run 'python src/train_models.py' first.")

results = pd.read_csv(RESULTS_PATH)

required = [
    "scenario", "model", "cv_MAE", "cv_MSE", "cv_RMSE", "cv_R2", "cv_RMSE_std",
    "train_MAE", "train_RMSE", "train_R2", "test_MAE", "test_MSE", "test_RMSE", "test_R2",
]
missing = sorted(set(required) - set(results.columns))
if missing:
    raise SystemExit(f"model_results.csv is missing columns: {missing}. Rerun train_models.py.")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# SELECTION LOGIC
# ----------------------------------------------------------------------
def analyse_scenario(part):
    """Return (table of the real models with extra columns, dict describing the decision)."""
    baseline = part[part["model"] == BASELINE_NAME]
    assert len(baseline) == 1, "Baseline row is missing or duplicated!"
    baseline_rmse = float(baseline["cv_RMSE"].iloc[0])

    real = part[part["model"] != BASELINE_NAME].copy()
    assert len(real) > 0, "No real models found in the results!"

    real["cv_RMSE_se"] = real["cv_RMSE_std"] / np.sqrt(CV_FOLDS)   # standard error across folds
    real["beats_baseline"] = real["cv_RMSE"] < baseline_rmse
    real["cv_minus_train_RMSE"] = real["cv_RMSE"] - real["train_RMSE"]   # big gap = overfitting sign

    lowest = real.loc[real["cv_RMSE"].idxmin()]
    threshold = float(lowest["cv_RMSE"] + lowest["cv_RMSE_se"])
    real["within_1se_of_best"] = real["cv_RMSE"] <= threshold

    decision = {
        "baseline_rmse": baseline_rmse,
        "lowest_name": lowest["model"],
        "threshold": threshold,
        "selected_name": None,
        "rule": "",
    }

    if not real["beats_baseline"].any():
        decision["rule"] = "No real model beat the baseline, so no model is selected."
        return real, decision

    if USE_ONE_SE_RULE:
        candidates = real[real["within_1se_of_best"] & real["beats_baseline"]].copy()
        candidates["complexity"] = candidates["model"].map(
            lambda m: COMPLEXITY_ORDER.index(m) if m in COMPLEXITY_ORDER else len(COMPLEXITY_ORDER)
        )
        chosen = candidates.sort_values(["complexity", "cv_RMSE"]).iloc[0]
        decision["rule"] = ("One-standard-error rule: simplest model whose cv_RMSE is within "
                            "one standard error of the lowest cv_RMSE.")
    else:
        chosen = lowest
        decision["rule"] = "Lowest cross-validated RMSE."

    decision["selected_name"] = chosen["model"]
    return real, decision


# ----------------------------------------------------------------------
# CHARTS
# ----------------------------------------------------------------------
def bar_colors(models, selected):
    return [GRAY if m == BASELINE_NAME else GREEN if m == selected else BLUE for m in models]


def legend_handles(has_selection):
    handles = []
    if has_selection:
        handles.append(Patch(color=GREEN, label="Selected model"))
    handles += [Patch(color=BLUE, label="Other models"), Patch(color=GRAY, label="Baseline (predict mean)")]
    return handles


def plot_cv_rmse(decisions, path):
    """The deciding chart: cross-validated RMSE per model, whiskers = standard error."""
    scenarios = list(decisions)
    xmax = (results["cv_RMSE"] + results["cv_RMSE_std"] / np.sqrt(CV_FOLDS)).max() * 1.18
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.5 * len(scenarios), 5.4), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, sc in zip(axes, scenarios):
        part = results[results["scenario"] == sc].copy()
        part["se"] = part["cv_RMSE_std"] / np.sqrt(CV_FOLDS)
        part = part.sort_values("cv_RMSE", ascending=False)   # best model ends up at the top
        sel = decisions[sc]["selected_name"]

        ax.barh(part["model"], part["cv_RMSE"], xerr=part["se"], capsize=3,
                color=bar_colors(part["model"], sel))
        for y, (v, e) in enumerate(zip(part["cv_RMSE"], part["se"])):
            ax.text(v + e + xmax * 0.01, y, f"{v:.3f}", va="center", fontsize=9)
        if USE_ONE_SE_RULE:
            ax.axvline(decisions[sc]["threshold"], color="black", linestyle="--", linewidth=1)
            ax.text(decisions[sc]["threshold"], len(part) - 0.45, " best + 1 SE",
                    fontsize=8, va="bottom", ha="left")
        ax.set_xlim(0, xmax)
        ax.set_title(sc)
        ax.set_xlabel("Cross-validated RMSE (grade points, lower is better)")
        ax.set_ylabel("Model")
        ax.legend(handles=legend_handles(sel is not None), loc="lower right", fontsize=8)

    fig.suptitle(f"Model comparison: {CV_FOLDS}-fold cross-validation on the training set",
                 fontweight="bold")
    fig.text(0.99, 0.005, "Whiskers = standard error across folds (std / sqrt(k)). "
             "Dashed line = lowest cv_RMSE + 1 SE.", ha="right", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_grid(decisions, path):
    """Cross-validated MAE, RMSE and R2 side by side, one row per scenario."""
    metrics = [
        ("cv_MAE", "MAE (grade points, lower is better)"),
        ("cv_RMSE", "RMSE (grade points, lower is better)"),
        ("cv_R2", "R\u00b2 (higher is better)"),
    ]
    scenarios = list(decisions)
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 4.6 * len(scenarios)), squeeze=False)

    for r, sc in enumerate(scenarios):
        part = results[results["scenario"] == sc]
        sel = decisions[sc]["selected_name"]
        for c, (col, label) in enumerate(metrics):
            ax = axes[r][c]
            p = part.sort_values(col, ascending=(col == "cv_R2"))   # best model at the top
            ax.barh(p["model"], p[col], color=bar_colors(p["model"], sel))
            for y, v in enumerate(p[col]):
                ax.text(v, y, f" {v:.3f} " if v >= 0 else f" {v:.3f} ", va="center",
                        ha="left" if v >= 0 else "right", fontsize=8)
            if col == "cv_R2":
                ax.axvline(0, color="black", linewidth=0.8)
            ax.margins(x=0.18)
            ax.set_title(f"{sc}\n{label}", fontsize=10)
            ax.set_xlabel("Cross-validated value")
            ax.set_ylabel("Model")
    fig.suptitle("Cross-validated MAE, RMSE and R\u00b2 for every model", fontweight="bold")
    fig.legend(handles=legend_handles(any(d["selected_name"] for d in decisions.values())),
               loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_train_cv_test(path):
    """Train vs cross-validated vs test RMSE: a quick look at overfitting."""
    scenarios = list(results["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(8.5 * len(scenarios), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.26

    for ax, sc in zip(axes, scenarios):
        part = results[results["scenario"] == sc]
        x = np.arange(len(part))
        ax.bar(x - width, part["train_RMSE"], width, label="Train", color="#9ecae1")
        ax.bar(x, part["cv_RMSE"], width, label="Cross-validation (used to choose)", color=BLUE)
        ax.bar(x + width, part["test_RMSE"], width, label="Test (final check only)", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(m, 12) for m in part["model"]], fontsize=9)
        ax.set_title(sc)
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE (grade points, lower is better)")
        ax.legend(fontsize=8)
    fig.suptitle("Train vs cross-validated vs test RMSE", fontweight="bold")
    fig.text(0.99, 0.005, "A much lower train bar than CV bar suggests overfitting.",
             ha="right", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
decisions = {}
selection_rows = []

for scenario in results["scenario"].unique():
    part = results[results["scenario"] == scenario]
    real, decision = analyse_scenario(part)
    decisions[scenario] = decision
    name = decision["selected_name"]

    section(f"SCENARIO: {scenario}")
    log(SCENARIOS.get(scenario, {}).get("description", ""))

    log(f"\nBaseline (predict mean) cv_RMSE: {decision['baseline_rmse']:.3f}")
    log("\nReal models, sorted by cross-validated RMSE:")
    table = real.sort_values("cv_RMSE").set_index("model")[[
        "cv_MAE", "cv_RMSE", "cv_RMSE_se", "cv_R2", "train_RMSE",
        "cv_minus_train_RMSE", "within_1se_of_best", "beats_baseline",
    ]].round(3)
    log(table.to_string())

    lowest_row = real[real["model"] == decision["lowest_name"]].iloc[0]
    log(f"\nLowest cv_RMSE          : {decision['lowest_name']} ({lowest_row['cv_RMSE']:.3f})")
    log(f"Lowest cv_RMSE + 1 SE   : {decision['threshold']:.3f}")
    log(f"Rule used               : {decision['rule']}")

    if name is None:
        log("\nSELECTED MODEL: none. Investigate the features and the data before choosing anything.")
        selection_rows.append({"scenario": scenario, "selected_model": "none",
                               "lowest_cv_rmse_model": decision["lowest_name"],
                               "selection_rule": decision["rule"],
                               "baseline_cv_RMSE": decision["baseline_rmse"]})
        continue

    chosen = real[real["model"] == name].iloc[0]
    improvement = 1 - chosen["cv_RMSE"] / decision["baseline_rmse"]
    log(f"\nSELECTED MODEL: {name}")
    log(f"  cv_RMSE = {chosen['cv_RMSE']:.3f} (SE {chosen['cv_RMSE_se']:.3f}),  "
        f"cv_MAE = {chosen['cv_MAE']:.3f},  cv_R2 = {chosen['cv_R2']:.3f}")
    log(f"  Its cv_RMSE is {improvement:.1%} lower than the baseline's ({decision['baseline_rmse']:.3f}).")

    if name != decision["lowest_name"]:
        log(f"  {decision['lowest_name']} had the lowest cv_RMSE ({lowest_row['cv_RMSE']:.3f}), but "
            f"{name} is within one standard error of it and is simpler, so it is preferred.")
    else:
        log("  It also has the lowest cv_RMSE of all real models.")

    tied = real[real["within_1se_of_best"] & (real["model"] != name)]["model"].tolist()
    log(f"  Other models statistically indistinguishable from the best: {tied if tied else 'none'}")

    log(f"  Train RMSE = {chosen['train_RMSE']:.3f} vs cv RMSE = {chosen['cv_RMSE']:.3f} "
        f"(gap {chosen['cv_minus_train_RMSE']:.3f}).")
    if chosen["train_RMSE"] < 0.5 * chosen["cv_RMSE"]:
        log("  WARNING (rule of thumb): train error is less than half of the cv error. "
            "Overfitting is likely.")

    full = part[part["model"] == name].iloc[0]
    log(f"\n  Test-set metrics for the selected model (shown AFTER choosing, never used to choose):")
    log(f"    MAE = {full['test_MAE']:.3f}   MSE = {full['test_MSE']:.3f}   "
        f"RMSE = {full['test_RMSE']:.3f}   R2 = {full['test_R2']:.3f}")
    log(f"    (cv_RMSE was {full['cv_RMSE']:.3f}. With only a small test set, some difference is normal.)")

    selection_rows.append({
        "scenario": scenario, "selected_model": name,
        "lowest_cv_rmse_model": decision["lowest_name"],
        "selection_rule": decision["rule"],
        "baseline_cv_RMSE": decision["baseline_rmse"],
        "cv_MAE": full["cv_MAE"], "cv_MSE": full["cv_MSE"],
        "cv_RMSE": full["cv_RMSE"], "cv_R2": full["cv_R2"],
        "test_MAE": full["test_MAE"], "test_MSE": full["test_MSE"],
        "test_RMSE": full["test_RMSE"], "test_R2": full["test_R2"],
    })

# ----------------------------------------------------------------------
section("CHARTS AND FILES")
# ----------------------------------------------------------------------
plot_cv_rmse(decisions, FIGURES_DIR / "best_model_cv_rmse.png")
plot_metric_grid(decisions, FIGURES_DIR / "best_model_metric_grid.png")
plot_train_cv_test(FIGURES_DIR / "best_model_train_cv_test.png")

for fname in ["best_model_cv_rmse.png", "best_model_metric_grid.png", "best_model_train_cv_test.png"]:
    p = FIGURES_DIR / fname
    assert p.exists() and p.stat().st_size > 0, f"{fname} was not saved!"
    log(f"Saved: {p.relative_to(PROJECT_ROOT)}")

pd.DataFrame(selection_rows).to_csv(SELECTION_PATH, index=False)
log(f"Saved: {SELECTION_PATH.relative_to(PROJECT_ROOT)}")

log("\nReminders:")
log("  - This selection is among UNTUNED models. Rerun it after hyperparameter tuning.")
log("  - Choosing scenario A or B depends on your goal (late prediction vs early warning), not on the scores.")
REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
log(f"Saved: {REPORT_PATH.relative_to(PROJECT_ROOT)}")