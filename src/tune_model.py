"""
tune_model.py
Hyperparameter tuning with GridSearchCV for the model selected in Step 10.

For each scenario (A = with G1/G2, B = without) it:
  1. reads the selected model from reports/best_model_selection.csv
  2. runs GridSearchCV (5-fold cross-validation, TRAINING set only)
  3. prints the best parameters and the best cross-validated score
  4. evaluates the tuned model and the original model in exactly the same way
     (cross-validation, train, test) and compares them
The test set is NOT used by the search. It is only scored afterwards, for information.

Saves:
  reports/tuning_results.csv               original vs tuned metrics
  reports/tuned_params.json                best parameters (used later when saving the final model)
  reports/tuning_grid_<scenario>.csv       every combination tried, with its scores
  reports/tuning_report.txt                the full printed output
  reports/figures/models/tuning_comparison.png

Run from the project root:  python src/tune_model.py
"""

import json
import math
import textwrap

import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from features import (
    PROJECT_ROOT,
    RANDOM_STATE,
    SCENARIOS,
    TARGET,
    load_clean_data,
    split_data,
)
from train_models import CV_FOLDS, build_model_pipeline, evaluate_model, get_models

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# Must be the SAME value as in train_models.py and the other scripts.
DROP_ZERO_GRADES = False

# Normally leave this empty: the model chosen by select_best_model.py is tuned.
# To force a model for a scenario, write e.g. {"A_late_prediction": "Gradient Boosting"}.
# Choose it from CROSS-VALIDATION results, never from test scores.
MANUAL_MODEL_CHOICE = {}

# Linear Regression has no setting worth tuning, so we tune its regularised sibling instead.
TUNING_SUBSTITUTE = {"Linear Regression": "Ridge Regression"}

# Every grid includes the starting values used in train_models.py.
PARAM_GRIDS = {
    "Ridge Regression": {
        "model__alpha": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
    },
    "Decision Tree": {
        "model__max_depth": [2, 3, 4, 5, 6, 8, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
    },
    "Random Forest": {
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [1.0, 0.5, "sqrt"],
    },
    "Gradient Boosting": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
    },
}

# The search scores every combination on all three metrics; RMSE decides the winner.
SCORING = {
    "RMSE": "neg_root_mean_squared_error",   # scikit-learn reports errors as negatives
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "models"
SELECTION_PATH = REPORTS_DIR / "best_model_selection.csv"
RESULTS_PATH = REPORTS_DIR / "tuning_results.csv"
PARAMS_PATH = REPORTS_DIR / "tuned_params.json"
REPORT_PATH = REPORTS_DIR / "tuning_report.txt"

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
# HELPERS
# ----------------------------------------------------------------------
def read_selected_models():
    """Return {scenario: model name} from Step 10, with optional manual overrides."""
    if not SELECTION_PATH.exists():
        raise SystemExit(
            "reports/best_model_selection.csv not found. "
            "Run 'python src/train_models.py' and then 'python src/select_best_model.py' first."
        )
    selection = pd.read_csv(SELECTION_PATH)
    chosen = dict(zip(selection["scenario"], selection["selected_model"]))
    chosen.update(MANUAL_MODEL_CHOICE)
    unknown = sorted(set(chosen) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown scenario names: {unknown}")
    return chosen


def strip_prefix(params):
    """{'model__alpha': 10} -> {'alpha': 10}  (the prefix only exists inside the pipeline)."""
    return {key.split("__", 1)[1]: value for key, value in params.items()}


def edge_notes(grid, best_params):
    """Warn when a numeric best value sits at the edge of its grid (the true best may lie beyond it)."""
    notes = []
    for key, values in grid.items():
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(values) < 3 or len(numeric) != len(values):
            continue
        best = best_params[key]
        if best == min(values) or best == max(values):
            notes.append(f"'{key.split('__', 1)[1]}' = {best} is at the edge of its grid "
                         f"{values}. A better value may lie outside the grid.")
    return notes


def summary_row(scenario, stage, name, params_text, metrics):
    """One line of the comparison table."""
    return {"scenario": scenario, "stage": stage, "params": params_text, **metrics}


# ----------------------------------------------------------------------
# TUNE ONE SCENARIO
# ----------------------------------------------------------------------
def tune_scenario(scenario, selected_name, X_train, y_train, X_test, y_test, folds):
    use_prior = SCENARIOS[scenario]["use_prior_grades"]
    tuned_name = TUNING_SUBSTITUTE.get(selected_name, selected_name)

    section(f"SCENARIO: {scenario}")
    log(SCENARIOS[scenario]["description"])
    log(f"\nModel selected in Step 10 : {selected_name}")
    if tuned_name != selected_name:
        log(f"Model that will be tuned  : {tuned_name}")
        log(f"  ({selected_name} has no setting worth tuning. {tuned_name} is Linear Regression plus a "
            f"penalty 'alpha'; a tiny alpha behaves almost like {selected_name}.)")
    if tuned_name not in PARAM_GRIDS:
        raise SystemExit(f"No parameter grid is defined for '{tuned_name}'. Add one to PARAM_GRIDS.")

    grid = PARAM_GRIDS[tuned_name]
    n_combos = math.prod(len(v) for v in grid.values())
    log(f"\nParameter grid ({n_combos} combinations x {CV_FOLDS} folds = {n_combos * CV_FOLDS} fits):")
    for key, values in grid.items():
        log(f"  {key.split('__', 1)[1]:18s} {values}")

    # ---- 1. the grid search (training data only) -----------------------
    pipeline = build_model_pipeline(get_models()[tuned_name], use_prior)
    search = GridSearchCV(
        pipeline, grid, cv=folds, scoring=SCORING, refit="RMSE",
        n_jobs=1, error_score="raise",   # stop with a clear error instead of hiding a failed fit
    )
    log("\nRunning GridSearchCV (this can take a while for Random Forest) ...")
    search.fit(X_train, y_train)

    best_params = strip_prefix(search.best_params_)
    idx = search.best_index_
    best_rmse = -search.cv_results_["mean_test_RMSE"][idx]
    best_mae = -search.cv_results_["mean_test_MAE"][idx]
    best_r2 = search.cv_results_["mean_test_R2"][idx]

    log("\nBEST PARAMETERS FOUND:")
    for key, value in best_params.items():
        log(f"  {key} = {value}")
    log(f"\nBEST CROSS-VALIDATED SCORE: RMSE = {best_rmse:.3f}   MAE = {best_mae:.3f}   R2 = {best_r2:.3f}")
    log("  (This score picked the winner out of many combinations, so it is slightly optimistic.")
    log("   The test set below is the honest check.)")
    for note in edge_notes(grid, search.best_params_):
        log(f"  NOTE: {note}")

    # ---- 2. save and show the full grid --------------------------------
    cv = search.cv_results_
    grid_table = pd.DataFrame({
        "params": [json.dumps(strip_prefix(p), default=str) for p in cv["params"]],
        "cv_RMSE": -cv["mean_test_RMSE"],
        "cv_RMSE_std": cv["std_test_RMSE"],
        "cv_MAE": -cv["mean_test_MAE"],
        "cv_R2": cv["mean_test_R2"],
        "rank_by_RMSE": cv["rank_test_RMSE"],
    }).sort_values("rank_by_RMSE")
    grid_path = REPORTS_DIR / f"tuning_grid_{scenario}.csv"
    grid_table.to_csv(grid_path, index=False)
    log("\nTop 5 combinations by cross-validated RMSE:")
    log(grid_table.head(5).round(3).to_string(index=False))
    log(f"All {len(grid_table)} combinations saved -> {grid_path.relative_to(PROJECT_ROOT)}")

    # ---- 3. evaluate original and tuned models in EXACTLY the same way ---
    rows = []
    original = evaluate_model(selected_name, get_models()[selected_name], use_prior,
                              X_train, y_train, X_test, y_test)
    rows.append(summary_row(scenario, "original selected", selected_name,
                            "starting settings", original))

    if tuned_name != selected_name:
        defaults = evaluate_model(tuned_name, get_models()[tuned_name], use_prior,
                                  X_train, y_train, X_test, y_test)
        rows.append(summary_row(scenario, "starting settings", tuned_name,
                                "starting settings", defaults))

    tuned_model = get_models()[tuned_name]
    tuned_model.set_params(**best_params)
    tuned = evaluate_model(tuned_name, tuned_model, use_prior, X_train, y_train, X_test, y_test)
    rows.append(summary_row(scenario, "tuned", tuned_name,
                            json.dumps(best_params, default=str), tuned))

    # Consistency check: the re-evaluation must reproduce the grid-search score
    if abs(tuned["cv_RMSE"] - best_rmse) > 1e-6:
        log(f"\nWARNING: re-evaluated cv_RMSE ({tuned['cv_RMSE']:.6f}) differs from the grid-search "
            f"score ({best_rmse:.6f}). Tell me about this.")
    else:
        log("\nConsistency check: re-evaluating the tuned model reproduces the grid-search score.  OK")

    # ---- 4. the comparison table ---------------------------------------
    table = pd.DataFrame(rows).set_index(["stage", "model"])
    cols = ["cv_MAE", "cv_RMSE", "cv_R2", "train_RMSE", "test_MAE", "test_RMSE", "test_R2"]
    log("\nCOMPARISON (lower MAE/RMSE is better, higher R2 is better):")
    log(table[cols].round(3).to_string())

    # ---- 5. the verdict, based on cross-validation only ------------------
    original_se = original["cv_RMSE_std"] / math.sqrt(CV_FOLDS)
    gain = original["cv_RMSE"] - tuned["cv_RMSE"]
    adopt = bool(gain > original_se)

    log(f"\nCross-validated RMSE: original {original['cv_RMSE']:.3f}  ->  tuned {tuned['cv_RMSE']:.3f}"
        f"   (change {gain:+.3f}; standard error of the original = {original_se:.3f})")
    if adopt:
        log("VERDICT (cross-validation): the improvement is larger than one standard error, "
            "so the TUNED model is adopted.")
    else:
        log("VERDICT (cross-validation): the improvement is NOT larger than one standard error, "
            "so tuning gave no clear gain. Keep the SIMPLER original model.")
    log(f"For information only: test RMSE original {original['test_RMSE']:.3f} -> tuned {tuned['test_RMSE']:.3f}, "
        f"test R2 original {original['test_R2']:.3f} -> tuned {tuned['test_R2']:.3f}. "
        f"With a small test set some difference is normal; do NOT re-tune to improve it.")
    if tuned["train_RMSE"] < 0.5 * tuned["cv_RMSE"]:
        log("WARNING (rule of thumb): tuned train error is less than half of the cv error. "
            "Overfitting is likely.")

    info = {
        "selected_model": selected_name,
        "tuned_model": tuned_name,
        "best_params": best_params,
        "cv_RMSE_original": float(original["cv_RMSE"]),
        "cv_RMSE_tuned": float(tuned["cv_RMSE"]),
        "adopt_tuned": adopt,
    }
    return rows, info


# ----------------------------------------------------------------------
# CHART
# ----------------------------------------------------------------------
def plot_comparison(results, path):
    scenarios = list(results["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.5 * len(scenarios), 5.6),
                             sharey=True, squeeze=False)
    width = 0.36
    for ax, sc in zip(axes[0], scenarios):
        part = results[results["scenario"] == sc].reset_index(drop=True)
        x = np.arange(len(part))
        cv_se = part["cv_RMSE_std"] / math.sqrt(CV_FOLDS)
        b1 = ax.bar(x - width / 2, part["cv_RMSE"], width, yerr=cv_se, capsize=3,
                    color="#4C72B0", label="Cross-validation (used to decide)")
        b2 = ax.bar(x + width / 2, part["test_RMSE"], width,
                    color="#DD8452", label="Test (final check only)")
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{textwrap.fill(m, 14)}\n({s})"
                            for m, s in zip(part["model"], part["stage"])], fontsize=9)
        ax.set_title(sc)
        ax.set_xlabel("Model and stage")
        ax.set_ylabel("RMSE (grade points, lower is better)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Original vs tuned model", fontweight="bold")
    fig.text(0.99, 0.005, "Whiskers = standard error across folds (std / sqrt(k)).",
             ha="right", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = read_selected_models()

    df = load_clean_data(drop_zero_grades=DROP_ZERO_GRADES)
    X_train, X_test, y_train, y_test = split_data(df)   # the SAME split as every other script
    assert TARGET not in X_train.columns and TARGET not in X_test.columns, "G3 leaked into X!"
    log(f"Rows: {len(df)}  |  train: {X_train.shape}  test: {X_test.shape}"
        f"  (DROP_ZERO_GRADES = {DROP_ZERO_GRADES})")
    log(f"Selected models read from {SELECTION_PATH.name}: {selected}")

    # Same folds as train_models.py, so original and tuned are compared on identical splits
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_rows, all_info = [], {}
    for scenario in SCENARIOS:
        name = selected.get(scenario, "none")
        if name in ("none", None) or (isinstance(name, float) and math.isnan(name)):
            section(f"SCENARIO: {scenario}")
            log("No model was selected in Step 10 (none beat the baseline). Nothing to tune here.")
            continue
        rows, info = tune_scenario(scenario, name, X_train, y_train, X_test, y_test, folds)
        all_rows += rows
        all_info[scenario] = info

    if not all_rows:
        raise SystemExit("Nothing was tuned. Check reports/best_model_selection.csv.")

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_PATH, index=False)
    PARAMS_PATH.write_text(json.dumps(all_info, indent=2, default=str), encoding="utf-8")
    plot_comparison(results, FIGURES_DIR / "tuning_comparison.png")

    section("FILES SAVED")
    log(f"Comparison table -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    log(f"Best parameters  -> {PARAMS_PATH.relative_to(PROJECT_ROOT)}")
    log(f"Chart            -> {(FIGURES_DIR / 'tuning_comparison.png').relative_to(PROJECT_ROOT)}")
    log("\nReminders:")
    log("  - The search used ONLY the training set. The test scores above did not influence any choice.")
    log("  - Do not change the grid or the split to improve the test score. That would be leakage.")
    log("  - Next step: final evaluation and saving the model (roadmap Step 8).")
    REPORT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Report saved -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()