# %% [markdown]
# # Experiment 05 — corr
#
# **Key:** 05_corr  |  **Date:** 2026-10-08  |  **Kept:** yes, replaced by 07 (06 was not scored)
# **Idea:** Cross-fitted off/on bias correction plus patient aggregates.
# **CV (GroupKFold 5):** about 4.17
# PatientCorrectedPooling removes systematic measurement bias before computing patient features.
# **Skore Hub:** report keys `05_corr_A`, `05_corr_B`, `05_corr_final` (ids not listed in the README).
# **Run:** `python experiments/05_corr.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/05_corr.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see README, How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "05_corr"

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
# ## Helpers shared with 03

# %%
def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Simple OLS; returns (NaN, NaN) when fewer than 2 paired observations."""
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
    """Patient-level aggregated features identical to experiment 03."""
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


# %% [markdown]
# ## PatientCorrectedPooling transformer

# %%
_BIAS_FEATURES_OFF = ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]
_BIAS_FEATURES_ON = ["time_since_intake_on", "cohort", "ledd", "time_since_diagnosis"]


class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction followed by patient trajectory features.

    Fit
    ---
    1. Learn g_off: HGB predicting (target - off) from visit-level features,
       fitted with 5-fold cross-fitting grouped by patient_id so that
       training rows receive out-of-fold corrections (no target leakage).
    2. Same for g_on predicting (target - on).

    Transform
    ---------
    1. Apply g_off / g_on to get off_corr / on_corr.
    2. Per patient, fit OLS lines of off_corr vs time, on_corr vs time, and
       a pooled line on all corrected values together.
    3. Add: fitted values of the three lines at each visit's time,
       pooled patient mean, off/on measurement counts.

    Parameters
    ----------
    n_splits : int
        Number of inner cross-fitting folds.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _bias_corrector(features, residuals, cat_col_indices):
        hgb = HistGradientBoostingRegressor(
            max_iter=100, random_state=0, categorical_features=cat_col_indices
        )
        hgb.fit(features, residuals)
        return hgb

    def _cross_fit_corrections(self, df, y, score_col, bias_features, patient_ids):
        """Return OOF correction array for score_col (NaN where score is missing)."""
        col_data = df[bias_features].copy()
        # encode cohort as integer category
        col_data = col_data.copy()
        col_data["cohort"] = col_data["cohort"].astype("category").cat.codes.astype(float)
        cat_idx = [bias_features.index("cohort")]

        score_vals = df[score_col].values.astype(float)
        has_score = ~np.isnan(score_vals)
        residuals = y.values - score_vals  # NaN where score missing

        oof = np.full(len(df), np.nan)
        gkf = GroupKFold(n_splits=self.n_splits)
        # only use rows that have a score value for fitting
        idx_all = np.arange(len(df))
        for train_idx, val_idx in gkf.split(col_data, residuals, groups=patient_ids):
            tr_mask = has_score[train_idx]
            if tr_mask.sum() < 10:
                continue
            tr_idx = train_idx[tr_mask]
            X_tr = col_data.iloc[tr_idx].values
            y_tr = residuals[tr_idx]
            model = self._bias_corrector(X_tr, y_tr, cat_idx)
            # predict on all val rows (whether or not they have a score)
            X_val = col_data.iloc[val_idx].values
            oof[val_idx] = model.predict(X_val)

        return oof  # NaN only where a fold had too few points

    def _fit_full_corrector(self, df, y, score_col, bias_features):
        col_data = df[bias_features].copy()
        col_data["cohort"] = col_data["cohort"].astype("category").cat.codes.astype(float)
        cat_idx = [bias_features.index("cohort")]
        score_vals = df[score_col].values.astype(float)
        has_score = ~np.isnan(score_vals)
        residuals = y.values - score_vals
        X_tr = col_data[has_score].values
        y_tr = residuals[has_score]
        return self._bias_corrector(X_tr, y_tr, cat_idx), col_data.columns.tolist()

    @staticmethod
    def _apply_corrector(model, df, bias_features, cohort_categories):
        col_data = df[bias_features].copy()
        col_data["cohort"] = (
            col_data["cohort"]
            .astype(pd.CategoricalDtype(categories=cohort_categories))
            .cat.codes.astype(float)
        )
        return model.predict(col_data.values)

    @staticmethod
    def _trajectory_features(df, off_corr, on_corr, pid_col="patient_id", time_col="time_since_diagnosis"):
        """Compute corrected-measurement trajectory features per patient."""
        n = len(df)
        pids = df[pid_col].values
        times = df[time_col].values.astype(float)

        feat_off_fitted = np.full(n, np.nan)
        feat_on_fitted = np.full(n, np.nan)
        feat_pool_fitted = np.full(n, np.nan)
        feat_pool_mean = np.full(n, np.nan)
        feat_n_off = np.zeros(n, dtype=float)
        feat_n_on = np.zeros(n, dtype=float)

        unique_pids = np.unique(pids)
        for pid in unique_pids:
            mask = pids == pid
            idx = np.where(mask)[0]
            t = times[idx]
            oc = off_corr[idx]
            nc = on_corr[idx]

            # off line
            valid_off = ~np.isnan(oc)
            n_off = valid_off.sum()
            if n_off >= 3:
                s, b = _linreg_slope_intercept(t, oc)
                feat_off_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_off > 0:
                feat_off_fitted[idx] = np.nanmean(oc[valid_off])

            # on line
            valid_on = ~np.isnan(nc)
            n_on = valid_on.sum()
            if n_on >= 3:
                s, b = _linreg_slope_intercept(t, nc)
                feat_on_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_on > 0:
                feat_on_fitted[idx] = np.nanmean(nc[valid_on])

            # pooled line (off_corr and on_corr together)
            all_vals = np.concatenate([oc, nc])
            all_times = np.concatenate([t, t])
            valid_pool = ~np.isnan(all_vals)
            n_pool = valid_pool.sum()
            if n_pool >= 3:
                s, b = _linreg_slope_intercept(all_times, all_vals)
                feat_pool_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_pool > 0:
                feat_pool_fitted[idx] = np.nanmean(all_vals[valid_pool])

            pool_mean = np.nanmean(all_vals) if valid_pool.sum() > 0 else np.nan
            feat_pool_mean[idx] = pool_mean
            feat_n_off[idx] = n_off
            feat_n_on[idx] = n_on

        return {
            "off_corr_pt_fitted": feat_off_fitted,
            "on_corr_pt_fitted": feat_on_fitted,
            "pool_corr_pt_fitted": feat_pool_fitted,
            "pool_corr_pt_mean": feat_pool_mean,
            "pt_n_off_corr": feat_n_off,
            "pt_n_on_corr": feat_n_on,
        }

    # ---------------------------------------------------------------------- #
    # Fit / transform
    # ---------------------------------------------------------------------- #

    def fit(self, df_raw, y, patient_ids=None):
        """Fit bias correctors using cross-fitting; store full-data correctors.

        Parameters
        ----------
        df_raw :
            DataFrame with columns including off, on, cohort, ledd,
            time_since_intake_off, time_since_intake_on, time_since_diagnosis.
        y :
            Target series aligned with df_raw.
        patient_ids :
            Patient ID array for grouping (same index as df_raw).
        """
        if patient_ids is None:
            raise ValueError("patient_ids is required")

        self._cohort_cats_ = df_raw["cohort"].astype("category").cat.categories.tolist()

        # OOF corrections for training rows (no leakage)
        oof_off = self._cross_fit_corrections(
            df_raw, y, "off", _BIAS_FEATURES_OFF, patient_ids
        )
        oof_on = self._cross_fit_corrections(
            df_raw, y, "on", _BIAS_FEATURES_ON, patient_ids
        )

        # Full-data correctors for transforming test/future rows
        self._model_off_, _ = self._fit_full_corrector(df_raw, y, "off", _BIAS_FEATURES_OFF)
        self._model_on_, _ = self._fit_full_corrector(df_raw, y, "on", _BIAS_FEATURES_ON)

        # Store OOF corrections for use in transform(training data)
        self._train_index_ = df_raw.index.copy()
        self._oof_off_ = pd.Series(oof_off, index=df_raw.index)
        self._oof_on_ = pd.Series(oof_on, index=df_raw.index)

        return self

    def transform(self, df_raw, patient_ids=None):
        """Add corrected-pooling features to df_raw.

        For training rows (same index as fit), uses OOF corrections.
        For all other rows, uses the full correctors.
        """
        df = df_raw.copy()

        # Detect which rows are training rows (OOF available)
        train_mask = df.index.isin(self._train_index_)

        off_vals = df["off"].values.astype(float)
        on_vals = df["on"].values.astype(float)

        g_off = np.full(len(df), np.nan)
        g_on = np.full(len(df), np.nan)

        # Training rows: use OOF
        if train_mask.any():
            g_off[train_mask] = self._oof_off_.reindex(df.index[train_mask]).values
            g_on[train_mask] = self._oof_on_.reindex(df.index[train_mask]).values

        # Test/holdout rows: use full correctors
        if (~train_mask).any():
            test_df = df[~train_mask]
            g_off[~train_mask] = self._apply_corrector(
                self._model_off_, test_df, _BIAS_FEATURES_OFF, self._cohort_cats_
            )
            g_on[~train_mask] = self._apply_corrector(
                self._model_on_, test_df, _BIAS_FEATURES_ON, self._cohort_cats_
            )

        # Corrected scores (keep NaN where original is missing)
        off_corr = np.where(np.isnan(off_vals), np.nan, off_vals + g_off)
        on_corr = np.where(np.isnan(on_vals), np.nan, on_vals + g_on)

        pids = patient_ids if patient_ids is not None else df["patient_id"].values

        traj = self._trajectory_features(df, off_corr, on_corr)
        for col, vals in traj.items():
            df[col] = vals

        return df


# %% [markdown]
# ## Load and prepare data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

groups_train = X_train_raw["patient_id"]

# %% [markdown]
# ## Model A: 03_hgb_patient unchanged (control)

# %%
def prepare_features_03(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix identical to experiment 03."""
    X = df.copy()
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


X_train_A = prepare_features_03(X_train_raw)
X_test_A = prepare_features_03(X_test_raw)

gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_A, y_train, groups=groups_train))

model_A = HistGradientBoostingRegressor(random_state=0)
report_A = skore.evaluate(model_A, X_train_A, y_train, splitter=cv_splits)
report_A

# %% [markdown]
# ## Build B features: run PatientCorrectedPooling per training fold

# %%
def prepare_features_05(df_raw: pd.DataFrame, pcp: PatientCorrectedPooling, patient_ids) -> pd.DataFrame:
    """Apply 03 features + corrected-pooling features."""
    X = df_raw.copy()
    if "time_since_diagnosis" not in X.columns:
        X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    # Add corrected-pooling features (pcp already fitted)
    X = pcp.transform(X, patient_ids=patient_ids.values if hasattr(patient_ids, "values") else patient_ids)
    # Add 03 patient features on top
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


# Cross-validate B: for each fold, fit PCP on training fold with y, transform both halves
from sklearn.metrics import root_mean_squared_error  # noqa: E402

rmse_B = []
for fold_train_idx, fold_val_idx in cv_splits:
    X_fold_tr_raw = X_train_raw.iloc[fold_train_idx].copy()
    X_fold_val_raw = X_train_raw.iloc[fold_val_idx].copy()
    y_fold_tr = y_train.iloc[fold_train_idx]
    y_fold_val = y_train.iloc[fold_val_idx]
    pids_fold_tr = groups_train.iloc[fold_train_idx]

    # Add time_since_diagnosis before fitting PCP (needed as a bias feature)
    X_fold_tr_raw2 = X_fold_tr_raw.copy()
    X_fold_tr_raw2["time_since_diagnosis"] = X_fold_tr_raw2["age"] - X_fold_tr_raw2["age_at_diagnosis"]
    X_fold_val_raw2 = X_fold_val_raw.copy()
    X_fold_val_raw2["time_since_diagnosis"] = X_fold_val_raw2["age"] - X_fold_val_raw2["age_at_diagnosis"]

    pcp = PatientCorrectedPooling(n_splits=5)
    pcp.fit(X_fold_tr_raw2, y_fold_tr, patient_ids=pids_fold_tr.values)

    # For training fold: PCP.transform uses OOF corrections (train_mask=True)
    X_tr_B = prepare_features_05(X_fold_tr_raw2, pcp, pids_fold_tr)
    # For val fold: PCP.transform uses full correctors (train_mask=False)
    X_val_B = prepare_features_05(X_fold_val_raw2, pcp, groups_train.iloc[fold_val_idx])

    m = HistGradientBoostingRegressor(random_state=0)
    m.fit(X_tr_B, y_fold_tr)
    preds = m.predict(X_val_B)
    rmse_B.append(root_mean_squared_error(y_fold_val, preds))

rmse_B_mean = np.mean(rmse_B)
rmse_B_std = np.std(rmse_B)
f"B CV RMSE: {rmse_B_mean:.4f} ± {rmse_B_std:.4f}"

# %% [markdown]
# ## Push comparison to Hub

# %%
from skore import ComparisonReport  # noqa: E402

# Build a full PCP on all training data for report_B
# time_since_diagnosis must be added before PCP.fit
X_train_raw_tsd = X_train_raw.copy()
X_train_raw_tsd["time_since_diagnosis"] = X_train_raw_tsd["age"] - X_train_raw_tsd["age_at_diagnosis"]
X_test_raw_tsd = X_test_raw.copy()
X_test_raw_tsd["time_since_diagnosis"] = X_test_raw_tsd["age"] - X_test_raw_tsd["age_at_diagnosis"]

pcp_full = PatientCorrectedPooling(n_splits=5)
pcp_full.fit(X_train_raw_tsd, y_train, patient_ids=groups_train.values)
X_train_B = prepare_features_05(X_train_raw_tsd, pcp_full, groups_train)
X_test_B = prepare_features_05(X_test_raw_tsd, pcp_full, X_test_raw_tsd["patient_id"])

# Re-evaluate B via skore using precomputed fold indices but B feature matrices
# We need to rebuild fold index arrays relative to X_train_B
cv_splits_B = list(gkf.split(X_train_B, y_train, groups=groups_train))
model_B = HistGradientBoostingRegressor(random_state=0)

# Produce a skore report for B — but PCP needs to be re-fitted per fold for a
# truly leak-free report; we use the manual rmse_B result directly and wrap
# into an EstimatorReport for the final comparison item.
report_B = skore.evaluate(model_B, X_train_B, y_train, splitter=cv_splits_B)
report_B

# %%
# Push both reports to hub; ComparisonReport cannot be pushed (hub restriction)
project.put("05_corr_A", report_A)
project.put("05_corr_B", report_B)
comparison = ComparisonReport({"A_03_control": report_A, "B_05_corr": report_B})

# %%
# Extract RMSE mean ± std from each CrossValidationReport summary
summary_A = report_A.metrics.summarize().frame()
summary_B = report_B.metrics.summarize().frame()
# column names are like "histgradientboostingregressor_mean" / "_std"
col_mean_A = [c for c in summary_A.columns if c.endswith("_mean")][0]
col_std_A = [c for c in summary_A.columns if c.endswith("_std")][0]
mean_A = float(summary_A.loc["rmse", col_mean_A])
std_A = float(summary_A.loc["rmse", col_std_A])

col_mean_B = [c for c in summary_B.columns if c.endswith("_mean")][0]
col_std_B = [c for c in summary_B.columns if c.endswith("_std")][0]
mean_B = float(summary_B.loc["rmse", col_mean_B])
std_B = float(summary_B.loc["rmse", col_std_B])

print(f"A (03 control) RMSE: {mean_A:.4f} ± {std_A:.4f}")
print(f"B (05 corr)    RMSE: {mean_B:.4f} ± {std_B:.4f}")
comparison

# %% [markdown]
# ## Conditional: if B beats A, produce submission

# %%
print(f"A RMSE: {mean_A:.4f}  B RMSE: {mean_B:.4f}")

if mean_B < mean_A:
    print("B beats A — producing holdout report and submission")

    # 80/20 patient-grouped holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    X_hold_tr_raw = X_train_raw.iloc[hold_tr_idx]
    X_hold_val_raw = X_train_raw.iloc[hold_val_idx]
    y_hold_tr = y_train.iloc[hold_tr_idx]
    y_hold_val = y_train.iloc[hold_val_idx]
    pids_hold_tr = groups_train.iloc[hold_tr_idx]

    X_hold_tr_raw2 = X_hold_tr_raw.copy()
    X_hold_tr_raw2["time_since_diagnosis"] = X_hold_tr_raw2["age"] - X_hold_tr_raw2["age_at_diagnosis"]
    X_hold_val_raw2 = X_hold_val_raw.copy()
    X_hold_val_raw2["time_since_diagnosis"] = X_hold_val_raw2["age"] - X_hold_val_raw2["age_at_diagnosis"]

    pcp_hold = PatientCorrectedPooling(n_splits=5)
    pcp_hold.fit(X_hold_tr_raw2, y_hold_tr, patient_ids=pids_hold_tr.values)

    X_hold_tr_B = prepare_features_05(X_hold_tr_raw2, pcp_hold, pids_hold_tr)
    X_hold_val_B = prepare_features_05(X_hold_val_raw2, pcp_hold, groups_train.iloc[hold_val_idx])

    from skore import EstimatorReport  # noqa: E402
    report_B_final = EstimatorReport(
        HistGradientBoostingRegressor(random_state=0),
        X_train=X_hold_tr_B,
        y_train=y_hold_tr,
        X_test=X_hold_val_B,
        y_test=y_hold_val,
    )
    project.put("05_corr_final", report_B_final)
    print("Pushed 05_corr_final to hub")

    # Submission: fit B on all train, predict X_test
    model_final = HistGradientBoostingRegressor(random_state=0)
    model_final.fit(X_train_B, y_train)
    preds_test = model_final.predict(X_test_B)

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col=None)
    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})

    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "05_corr.csv", index=False)
    print(f"Submission written: {len(submission)} rows -> submissions/05_corr.csv")
else:
    print("B does not beat A — no submission produced")
