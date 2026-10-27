# %% [markdown]
# # EDA: Parkinson's OFF motor score (bobathon-esilv)
#
# Exploratory data analysis of the bobathon-esilv dataset, run before
# designing a model.
#
# - **Raw data** is read-only — `X_train.csv`, `y_train.csv`, `X_test.csv`
#   are loaded from `data/` and never modified.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): `eda_visits.html`
#   (interactive skrub report) and `eda.md` (narrative + modelling implications).

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# X_train + y_train are joined on their Index to form a single visits table.
# X_test is loaded separately.

# %%
X_train = pd.read_csv(EDA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(EDA_DIR / "y_train.csv", index_col="Index")
RAW = X_train.join(y_train)
X_test = pd.read_csv(EDA_DIR / "X_test.csv", index_col="Index")
{"train_shape": RAW.shape, "test_shape": X_test.shape}  # noqa: B018

# %% [markdown]
# ## Table overview (training visits)
#
# Per-column summary: dtype, fraction missing, and number of unique values.
# Full interactive report saved to `data/eda_visits.html`.

# %%
report = skrub.TableReport(RAW, title="Parkinson visits (train)", verbose=0)
report.write_html(EDA_DIR / "eda_visits.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of `target` (true OFF MDS-UPDRS motor score, 0–132).
# Shapes whether cross-validation should stratify and which metric to use.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns and high-cardinality id / group-like columns.
# `patient_id` repeats across multiple visits — crucial for choosing a
# GroupKFold splitter so a patient's visits do not leak across folds.

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
n_patients_train = RAW["patient_id"].nunique()
n_patients_test = X_test["patient_id"].nunique()
visits_per_patient = RAW.groupby("patient_id").size()
{
    "datetime_cols": datetime_cols,
    "top_unique_ratio": unique_ratio[:10],
    "n_patients_train": n_patients_train,
    "n_patients_test": n_patients_test,
    "visits_per_patient_median": visits_per_patient.median(),
    "visits_per_patient_max": visits_per_patient.max(),
}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links are
# candidate predictors; an implausibly perfect one is a possible leakage flag.

# %%
assoc = skrub.column_associations(RAW)
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
