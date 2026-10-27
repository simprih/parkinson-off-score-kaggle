# %% [markdown]
# # Experiment 04 — hgb_smooth
#
# **Key:** 04_hgb_smooth  |  **Date:** 2026-10-08  |  **Kept:** yes, replaced by 05
# **Idea:** Per-patient linear smoothing of 03 predictions (PatientSmoothedRegressor).
# **CV (GroupKFold 5):** 4.19
# Smoothing the per-patient trajectory reduces visit-level noise slightly.
# **Skore Hub:** report key `04_hgb_smooth` (id not listed in the report).
# **Run:** `python experiments/04_hgb_smooth.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/04_hgb_smooth.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see the report (report/rapport_SPS.pdf), How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login, ComparisonReport, CrossValidationReport
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "04_hgb_smooth"

# %% [markdown]
# ## Open skore Hub project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Feature engineering helpers (identical to 03_hgb_patient)

# %%
def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """OLS slope and intercept of y on x; (NaN, NaN) if < 2 paired points."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan, np.nan
    xm, ym = x[mask], y[mask]
    xc = xm - xm.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        return np.nan, np.nan
    slope = (xc * (ym - ym.mean())).sum() / denom
    intercept = ym.mean() - slope * xm.mean()
    return slope, intercept


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append patient-level aggregated features (identical to experiment 03)."""
    df = df.copy()
    for col in ["off", "on", "ledd"]:
        stats = df.groupby("patient_id")[col].agg(
            **{
                f"{col}_pt_mean": "mean",
                f"{col}_pt_median": "median",
                f"{col}_pt_std": "std",
                f"{col}_pt_count": lambda s: s.notna().sum(),
            }
        )
        df = df.join(stats, on="patient_id")

    for score_col in ["off", "on"]:
        slopes, intercepts, fitted = {}, {}, {}
        for pid, grp in df.groupby("patient_id"):
            x = grp["time_since_diagnosis"].values.astype(float)
            y = grp[score_col].values.astype(float)
            s, b = _linreg_slope_intercept(x, y)
            slopes[pid] = s
            intercepts[pid] = b
            fitted_vals = np.where(np.isnan(s), np.nan, s * x + b)
            for idx, fv in zip(grp.index, fitted_vals):
                fitted[idx] = fv
        df[f"{score_col}_pt_slope"] = df["patient_id"].map(slopes)
        df[f"{score_col}_pt_intercept"] = df["patient_id"].map(intercepts)
        df[f"{score_col}_pt_fitted"] = pd.Series(fitted)

    df["off_dev"] = df["off"] - df["off_pt_mean"]
    df["on_dev"] = df["on"] - df["on_pt_mean"]
    df["pt_n_visits"] = df.groupby("patient_id")["patient_id"].transform("count")
    df["pt_visit_rank"] = df.groupby("patient_id")["time_since_diagnosis"].rank(
        method="first", na_option="keep"
    )
    return df


def prepare_features_with_pid(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix keeping patient_id for PatientSmoothedRegressor."""
    X = df.copy()
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    # patient_id kept — PatientSmoothedRegressor drops it before base.predict
    return X


# %% [markdown]
# ## PatientSmoothedRegressor

# %%
class PatientSmoothedRegressor(BaseEstimator, RegressorMixin):
    """Wrap a base regressor with per-patient OLS smoothing on predictions.

    Parameters
    ----------
    base :
        sklearn-compatible regressor. Fitted on X with patient_id dropped.
    alpha :
        Blend weight toward the per-patient linear fit of predictions.
        output = alpha * fitted_line + (1 - alpha) * raw_prediction.
        Patients with fewer than 3 visits receive raw_prediction.
    """

    def __init__(self, base, alpha: float = 0.8):
        self.base = base
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        """Fit base model on X with patient_id dropped."""
        self.base_ = clone(self.base)
        X_feat = X.drop(columns=["patient_id"])
        self.base_.fit(X_feat, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and optionally smooth along each patient's trajectory."""
        X_feat = X.drop(columns=["patient_id"])
        p = self.base_.predict(X_feat)

        if self.alpha == 0:
            return p

        output = p.copy()
        time = X["time_since_diagnosis"].values.astype(float)
        pid = X["patient_id"].values

        for patient in np.unique(pid):
            mask = pid == patient
            n = mask.sum()
            if n < 3:
                continue
            t = time[mask]
            pv = p[mask]
            s, b = _linreg_slope_intercept(t, pv)
            if np.isnan(s):
                continue
            fitted = s * t + b
            output[mask] = self.alpha * fitted + (1 - self.alpha) * pv

        return output


# %% [markdown]
# ## Load and prepare data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

groups_train = X_train_raw["patient_id"]

X_train = prepare_features_with_pid(X_train_raw)
X_test = prepare_features_with_pid(X_test_raw)

X_train.shape, X_test.shape

# %% [markdown]
# ## Precompute grouped CV splits

# %%
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
f"Precomputed {len(cv_splits)} folds"

# %% [markdown]
# ## Alpha sweep — evaluate each alpha as a CrossValidationReport
#
# ComparisonReport requires a collection of CrossValidationReport objects,
# not estimators.  We evaluate each alpha separately, then combine.

# %%
base_model = HistGradientBoostingRegressor(random_state=0)
alphas = [0, 0.5, 0.8, 1.0]

cv_reports: dict[str, CrossValidationReport] = {}
for a in alphas:
    label = f"alpha={a}"
    print(f"Evaluating {label}...")
    est = PatientSmoothedRegressor(base=base_model, alpha=a)
    cv_reports[label] = skore.evaluate(est, X_train, y_train, splitter=cv_splits)

"All CV evaluations complete"

# %% [markdown]
# ## Build ComparisonReport and print RMSE table

# %%
comparison = ComparisonReport(cv_reports)
metrics_df = comparison.metrics.summarize().frame()
metrics_df

# %% [markdown]
# ## RMSE mean ± std per alpha

# %%
rmse_table = {}
for label in cv_reports:
    m = cv_reports[label].metrics.summarize().frame()
    # columns are named after the estimator; rmse is an index value
    mean_col = [c for c in m.columns if c.endswith("_mean")][0]
    std_col = [c for c in m.columns if c.endswith("_std")][0]
    rmse_means = m.loc["rmse", mean_col]
    rmse_stds = m.loc["rmse", std_col]
    rmse_table[label] = {"rmse_mean": rmse_means, "rmse_std": rmse_stds}

print("\nRMSE per alpha (5-fold GroupKFold):")
print(f"{'alpha':<12} {'RMSE mean':>12} {'± std':>10}")
print("-" * 36)
best_label, best_rmse = None, float("inf")
for label, vals in rmse_table.items():
    m, s = vals["rmse_mean"], vals["rmse_std"]
    flag = ""
    if m < best_rmse:
        best_rmse, best_label = m, label
    print(f"{label:<12} {m:>12.4f} {s:>10.4f}{flag}")

print(f"\nBest: {best_label} (RMSE {best_rmse:.4f})")

# %% [markdown]
# ## Push best-alpha report to Hub under key 04_hgb_smooth

# %%
best_report = cv_reports[best_label]
project.put(EXPERIMENT_KEY, best_report)

# %% [markdown]
# ## Kaggle submission — best alpha refit on all training rows

# %%
import re as _re
_match = _re.search(r"alpha=([0-9.]+)", best_label)
best_alpha = float(_match.group(1)) if _match else 0.8
print(f"Fitting final model with alpha={best_alpha}")

final = PatientSmoothedRegressor(
    base=HistGradientBoostingRegressor(random_state=0), alpha=best_alpha
)
final.fit(X_train, y_train)
predictions = final.predict(X_test)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission = pd.DataFrame({"Index": X_test.index, "target": predictions})
submission.to_csv(submissions_dir / f"{EXPERIMENT_KEY}.csv", index=False)

assert len(submission) == 11013, f"Expected 11013 rows, got {len(submission)}"
f"Submission written: {len(submission)} rows"
