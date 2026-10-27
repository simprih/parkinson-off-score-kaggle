# %% [markdown]
# # Experiment 03 — hgb_patient
#
# **Key:** 03_hgb_patient  |  **Date:** 2026-10-08  |  **Kept:** yes, replaced by 04
# **Idea:** Add per-patient off/on mean and trend features to 02_hgb.
# **CV (GroupKFold 5):** 4.26
# Large drop from 7.36 shows that patient-level aggregates carry most of the signal.
# **Skore Hub:** report key `03_hgb_patient` (id not listed in the report).
# **Run:** `python experiments/03_hgb_patient.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/03_hgb_patient.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see the report (report/rapport_SPS.pdf), How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "03_hgb_patient"

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
# ## Helper: patient-level feature engineering

# %%
def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept) for a simple OLS fit of y on x.

    Returns (NaN, NaN) when fewer than 2 non-null paired observations exist.
    """
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
    """Append patient-level aggregated features to a visit DataFrame.

    All aggregations are computed from columns already in df (never from
    the target).  df must contain: patient_id, off, on, ledd,
    time_since_diagnosis, age.

    Parameters
    ----------
    df :
        Visit-level DataFrame with patient_id as a regular column.

    Returns
    -------
    pd.DataFrame
        Original df with extra columns added in-place (copy returned).
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Per-patient summary stats for off, on, ledd
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. Per-patient linear trend of off and on vs time_since_diagnosis
    # ------------------------------------------------------------------ #
    for score_col in ["off", "on"]:
        slopes, intercepts, fitted = {}, {}, {}
        for pid, grp in df.groupby("patient_id"):
            x = grp["time_since_diagnosis"].values.astype(float)
            y = grp[score_col].values.astype(float)
            s, b = _linreg_slope_intercept(x, y)
            slopes[pid] = s
            intercepts[pid] = b
            # fitted value at each visit's time_since_diagnosis
            fitted_vals = np.where(
                np.isnan(s),
                np.nan,
                s * x + b,
            )
            for idx, fv in zip(grp.index, fitted_vals):
                fitted[idx] = fv

        df[f"{score_col}_pt_slope"] = df["patient_id"].map(slopes)
        df[f"{score_col}_pt_intercept"] = df["patient_id"].map(intercepts)
        df[f"{score_col}_pt_fitted"] = pd.Series(fitted)

    # ------------------------------------------------------------------ #
    # 3. Deviation from patient mean
    # ------------------------------------------------------------------ #
    df["off_dev"] = df["off"] - df["off_pt_mean"]
    df["on_dev"] = df["on"] - df["on_pt_mean"]

    # ------------------------------------------------------------------ #
    # 4. Number of visits per patient; visit rank in time_since_diagnosis
    # ------------------------------------------------------------------ #
    df["pt_n_visits"] = df.groupby("patient_id")["patient_id"].transform("count")
    df["pt_visit_rank"] = df.groupby("patient_id")["time_since_diagnosis"].rank(
        method="first", na_option="keep"
    )

    return df


# %% [markdown]
# ## Load and prepare data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix: raw + derived + patient-level."""
    X = df.copy()
    # Derived scalar feature
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    # Patient-level features (uses patient_id internally, then we drop it)
    X = add_patient_features(X)
    # Categorical encoding for HGB
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    # Drop identifier columns
    X = X.drop(columns=["patient_id"])
    return X


groups_train = X_train_raw["patient_id"]

X_train = prepare_features(X_train_raw)
X_test = prepare_features(X_test_raw)

X_train.shape, X_test.shape

# %% [markdown]
# ## Precompute grouped CV splits

# %%
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))
f"Precomputed {len(cv_splits)} folds; fold 0 test size = {len(cv_splits[0][1])}"

# %% [markdown]
# ## Evaluate

# %%
model = HistGradientBoostingRegressor(random_state=0)
report = skore.evaluate(model, X_train, y_train, splitter=cv_splits)
report

# %% [markdown]
# ## Persist to Hub

# %%
project.put(EXPERIMENT_KEY, report)

# %% [markdown]
# ## Kaggle submission — refit on all training rows

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = HistGradientBoostingRegressor(random_state=0)
final.fit(X_train, y_train)
predictions = final.predict(X_test)

submission = pd.DataFrame({"Index": X_test.index, "target": predictions})
submission.to_csv(submissions_dir / f"{EXPERIMENT_KEY}.csv", index=False)

assert len(submission) == 11013, f"Expected 11013 rows, got {len(submission)}"
f"Submission written: {len(submission)} rows"
