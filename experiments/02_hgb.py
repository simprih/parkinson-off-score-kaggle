# %% [markdown]
# # Experiment 02 — hgb
#
# **Key:** 02_hgb  |  **Date:** 2026-10-08  |  **Kept:** yes, replaced by 03
# **Idea:** HGB on raw columns, first patient-honest grouped CV (GroupKFold 5).
# **CV (GroupKFold 5):** 7.36
# First trained model; establishes that HGB with native NaN handling beats the mean.
# **Skore Hub:** report key `02_hgb` (id not listed in the report).
# **Run:** `python experiments/02_hgb.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/02_hgb.py`).
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
EXPERIMENT_KEY = "02_hgb"

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
# ## Load and prepare data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a raw visit DataFrame.

    Adds time_since_diagnosis, casts cohort/gene to category,
    and drops Index-level identifiers. NaNs are left as-is.
    """
    X = df.copy()
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


# Keep patient_id as a separate series for GroupKFold groups.
groups_train = X_train_raw["patient_id"]

X_train = prepare_features(X_train_raw)
X_test = prepare_features(X_test_raw)

X_train.shape, X_test.shape

# %% [markdown]
# ## Precompute grouped CV splits
#
# GroupKFold(n_splits=5) on patient_id.  We precompute the split indices
# and pass the list directly to skore.evaluate as splitter=cv_splits so
# skore receives concrete (train_idx, test_idx) pairs.

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
