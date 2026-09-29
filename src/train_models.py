"""
train_models.py
Reusable training and evaluation code for the Student Performance project.

For each scenario (A = with G1/G2, B = without) and each model it computes:
  - 5-fold cross-validation metrics on the TRAINING set   (use these to compare models)
  - metrics on the training set itself                     (overfitting check)
  - metrics on the held-out test set                       (final honest check)
Metrics: MAE, MSE, RMSE, R2.

Saves:
  reports/model_results.csv
  reports/figures/models/model_comparison_cv_rmse.png

Run from the project root:  python src/train_models.py
Other scripts can also import build_model_pipeline, evaluate_model, get_models.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from features import (
    PROJECT_ROOT,
    RANDOM_STATE,
    SCENARIOS,
    TARGET,
    build_preprocessor,
    load_clean_data,
    split_data,
)

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
# The G3 == 0 decision is still open. Use the SAME value in every script.
DROP_ZERO_GRADES = False
CV_FOLDS = 5

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "models"
RESULTS_PATH = REPORTS_DIR / "model_results.csv"

BASELINE_NAME = "Baseline (predict mean)"


# ----------------------------------------------------------------------
# REUSABLE FUNCTIONS
# ----------------------------------------------------------------------
def get_models():
    """Return a fresh dictionary {name: unfitted model}. Settings are starting values, not tuned."""
    return {
        BASELINE_NAME: DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def build_model_pipeline(model, use_prior_grades):
    """
    Preprocessing (feature engineering, imputing, scaling, encoding) + one model.
    Because both live in ONE pipeline, every learned step is fitted on training rows only,
    and raw data can be passed straight in (this is what the Streamlit app will do).
    """
    return Pipeline([
        ("preprocess", build_preprocessor(use_prior_grades=use_prior_grades)),
        ("model", model),
    ])


def regression_metrics(y_true, y_pred):
    """The four metrics. RMSE is the square root of MSE."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_model(name, model, use_prior_grades, X_train, y_train, X_test, y_test):
    """Cross-validate on train, then fit on all of train and score train and test. Returns one row (dict)."""
    pipeline = build_model_pipeline(model, use_prior_grades)

    # ---- 1. cross-validation on the TRAINING set only ----------------
    # shuffle=True mixes the rows before making folds; the fixed seed makes it repeatable.
    # cross_validate clones the pipeline for each fold, so preprocessing is re-learned
    # from that fold's training part only.
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_out = cross_validate(
        pipeline, X_train, y_train, cv=folds,
        scoring={"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error", "R2": "r2"},
        error_score="raise",  # if a fold fails, stop and show the error instead of hiding it
    )
    fold_mae = -cv_out["test_MAE"]        # scikit-learn returns errors as negatives, so flip the sign
    fold_mse = -cv_out["test_MSE"]
    fold_rmse = np.sqrt(fold_mse)         # RMSE per fold, then averaged
    fold_r2 = cv_out["test_R2"]

    # ---- 2. fit on the full training set, score train and test ---------
    pipeline.fit(X_train, y_train)
    train_m = regression_metrics(y_train, pipeline.predict(X_train))
    test_m = regression_metrics(y_test, pipeline.predict(X_test))

    row = {
        "model": name,
        "cv_MAE": fold_mae.mean(),
        "cv_MSE": fold_mse.mean(),
        "cv_RMSE": fold_rmse.mean(),
        "cv_R2": fold_r2.mean(),
        "cv_RMSE_std": fold_rmse.std(),   # how much the score moves between folds
        "cv_R2_std": fold_r2.std(),
    }
    row.update({f"train_{k}": v for k, v in train_m.items()})
    row.update({f"test_{k}": v for k, v in test_m.items()})
    return row


def train_all(X_train, y_train, X_test, y_test):
    """Run every model in every scenario. Returns ONE DataFrame with all results."""
    rows = []
    for scenario_name, cfg in SCENARIOS.items():
        print(f"\n--- Training scenario: {scenario_name} ---")
        for model_name, model in get_models().items():
            print(f"  {model_name} ...")
            row = evaluate_model(
                model_name, model, cfg["use_prior_grades"], X_train, y_train, X_test, y_test
            )
            rows.append({"scenario": scenario_name, **row})
    return pd.DataFrame(rows)


def plot_cv_rmse(results, path):
    """Bar chart of cross-validated RMSE per model, one panel per scenario (whiskers = std across folds)."""
    scenarios = list(results["scenario"].unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, scenarios):
        part = results[results["scenario"] == scenario].sort_values("cv_RMSE", ascending=False)
        ax.barh(part["model"], part["cv_RMSE"], xerr=part["cv_RMSE_std"], capsize=3, color="#4C72B0")
        ax.set_title(scenario)
        ax.set_xlabel("Cross-validated RMSE (grade points, lower is better)")
        ax.set_ylabel("Model")
    fig.suptitle(f"Model comparison: {CV_FOLDS}-fold cross-validation on the training set",
                 fontweight="bold")
    fig.text(0.99, 0.005, "Whiskers = standard deviation across folds", ha="right", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# MAIN PROGRAM (runs only when you execute this file directly)
# ----------------------------------------------------------------------
def main():
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and split (same function, same seed as every other script)
    df = load_clean_data(drop_zero_grades=DROP_ZERO_GRADES)
    X_train, X_test, y_train, y_test = split_data(df)
    assert TARGET not in X_train.columns and TARGET not in X_test.columns, "G3 leaked into X!"
    print(f"Rows: {len(df)}  |  train: {X_train.shape}  test: {X_test.shape}"
          f"  (DROP_ZERO_GRADES = {DROP_ZERO_GRADES})")

    # 2. Train and evaluate everything
    results = train_all(X_train, y_train, X_test, y_test)

    # 3. Sanity checks on the results table
    expected_rows = len(SCENARIOS) * len(get_models())
    assert len(results) == expected_rows, "Some model results are missing!"
    assert not results.drop(columns=["scenario", "model"]).isna().any().any(), "NaN in results!"
    for prefix in ["cv", "train", "test"]:
        assert (results[f"{prefix}_RMSE"] >= results[f"{prefix}_MAE"] - 1e-9).all(), \
            f"{prefix}: RMSE should never be smaller than MAE"

    # 4. Save everything (full precision)
    results.to_csv(RESULTS_PATH, index=False)
    plot_cv_rmse(results, FIGURES_DIR / "model_comparison_cv_rmse.png")

    # 5. Print readable tables, one scenario at a time (sorted by cross-validated RMSE)
    for scenario_name, cfg in SCENARIOS.items():
        part = results[results["scenario"] == scenario_name].sort_values("cv_RMSE").set_index("model")

        print("\n" + "=" * 78)
        print(f"SCENARIO: {scenario_name}")
        print(cfg["description"])
        print("=" * 78)

        print(f"\n{CV_FOLDS}-fold CROSS-VALIDATION on the training set (use THIS to compare models):")
        print(part[["cv_MAE", "cv_MSE", "cv_RMSE", "cv_R2", "cv_RMSE_std", "cv_R2_std"]].round(3).to_string())

        print("\nTRAINING-set metrics (compare with the cv_ table to spot overfitting):")
        print(part[["train_MAE", "train_MSE", "train_RMSE", "train_R2"]].round(3).to_string())

        print("\nTEST-set metrics (final check, do NOT use to choose or tune models):")
        print(part[["test_MAE", "test_MSE", "test_RMSE", "test_R2"]].round(3).to_string())

        real = part.drop(index=BASELINE_NAME)
        print(f"\nLowest cv_RMSE among the five real models: {real['cv_RMSE'].idxmin()}"
              f"  (cv_RMSE = {real['cv_RMSE'].min():.3f})")

    print("\n" + "=" * 78)
    print("SCENARIO COMPARISON: cross-validated RMSE (grade points, lower is better)")
    print("=" * 78)
    pivot = results.pivot(index="model", columns="scenario", values="cv_RMSE").round(3)
    print(pivot.to_string())

    print(f"\nAll results saved     -> {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Comparison chart      -> {(FIGURES_DIR / 'model_comparison_cv_rmse.png').relative_to(PROJECT_ROOT)}")
    print("Done. Nothing was written to data/ and no test-based tuning was done.")


if __name__ == "__main__":
    main()
    