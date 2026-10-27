# %% [markdown]
# # Experiment 01 — dummy
#
# **Key:** 01_dummy  |  **Date:** 2026-10-08  |  **Kept:** baseline, replaced by 02
# **Idea:** Mean prediction — always predicts the training average (DummyRegressor).
# **CV (GroupKFold 5):** 16.48
# Establishes the RMSE floor; any real model must beat this score.
# **Skore Hub:** report key `01_dummy` (id not listed in the report).
# **Run:** `python experiments/01_dummy.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/01_dummy.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see the report (report/rapport_SPS.pdf), How to run.

# %%
from pathlib import Path

import pandas as pd
import skore
from skore import login
from sklearn.dummy import DummyRegressor

# Hub credentials from .skore (written by scripts/install_skore.py).
# load_skore_credentials() sets SKORE_HUB_URI and SKORE_HUB_API_KEY so
# login() does not open a browser.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

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
# ## Load data

# %%
X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

X_train.shape, y_train.shape, X_test.shape

# %% [markdown]
# ## Evaluate
#
# DummyRegressor(strategy="mean") ignores every feature and predicts
# the training mean for every visit. Evaluated on a random 20% holdout
# (splitter=0.2) so the score is comparable with 02_ridge.

# %%
dummy = DummyRegressor(strategy="mean")
report = skore.evaluate(dummy, X_train, y_train, splitter=0.2)
report

# %% [markdown]
# ## Persist to Hub

# %%
project.put("01_dummy", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on ALL training rows (no held-out split), predict on test,
# write submissions/01_dummy.csv.  The dummy always predicts the same
# value: the training mean.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

final = DummyRegressor(strategy="mean")
final.fit(X_train, y_train)
predictions = final.predict(X_test)

submission = pd.DataFrame({"Index": X_test.index, "target": predictions})
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
submission.head()
