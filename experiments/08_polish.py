# %% [markdown]
# # Experiment 08 — polish
#
# **Key:** 08_polish  |  **Date:** 2026-10-09  |  **Kept:** yes, replaced by 10 (09 was merged into 10)
# **Idea:** Tuned HGB settings on the 07-B feature set plus PatientSmoothedRegressor.
# **CV (GroupKFold 5):** 3.93
# HGB_BEST_08 settings (lr 0.03, 1000 iter, 63 leaves) used in all later experiments.
# **Holdout (20 % of patients):** 3.90 (Skore report `08_polish_final`).
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/57274 (`08_polish_final`).
# **Run:** `python experiments/08_polish.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/08_polish.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see README, How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login, ComparisonReport
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "08_polish"

# %% — Open skore Hub project
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% — Helpers (from 07_mixed)
_BIAS_FEATURES_OFF = ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]
_BIAS_FEATURES_ON  = ["time_since_intake_on",  "cohort", "ledd", "time_since_diagnosis"]


def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """OLS; returns (NaN, NaN) when fewer than 2 paired observations."""
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
    """Patient-level aggregated features (identical to experiments 03/07)."""
    df = df.copy()
    for col in ["off", "on", "ledd"]:
        stats = df.groupby("patient_id")[col].agg(
            **{
                f"{col}_pt_mean":   "mean",
                f"{col}_pt_median": "median",
                f"{col}_pt_std":    "std",
                f"{col}_pt_count":  lambda s: s.notna().sum(),
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
        df[f"{score_col}_pt_slope"]     = df["patient_id"].map(slopes)
        df[f"{score_col}_pt_intercept"] = df["patient_id"].map(intercepts)
        df[f"{score_col}_pt_fitted"]    = pd.Series(fitted)
    df["off_dev"]     = df["off"] - df["off_pt_mean"]
    df["on_dev"]      = df["on"]  - df["on_pt_mean"]
    df["pt_n_visits"] = df.groupby("patient_id")["patient_id"].transform("count")
    df["pt_visit_rank"] = df.groupby("patient_id")["time_since_diagnosis"].rank(
        method="first", na_option="keep"
    )
    return df


# %% — PatientCorrectedPooling (from 07_mixed / 05)
class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction (cross-fitted on training fold)."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    @staticmethod
    def _bias_corrector(features, residuals, cat_col_indices):
        hgb = HistGradientBoostingRegressor(
            max_iter=100, random_state=0, categorical_features=cat_col_indices
        )
        hgb.fit(features, residuals)
        return hgb

    def _cross_fit_corrections(self, df, y, score_col, bias_features, patient_ids):
        col_data = df[bias_features].copy()
        col_data["cohort"] = col_data["cohort"].astype("category").cat.codes.astype(float)
        cat_idx = [bias_features.index("cohort")]
        score_vals = df[score_col].values.astype(float)
        has_score  = ~np.isnan(score_vals)
        residuals  = y.values - score_vals
        oof = np.full(len(df), np.nan)
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, val_idx in gkf.split(col_data, residuals, groups=patient_ids):
            tr_mask = has_score[train_idx]
            if tr_mask.sum() < 10:
                continue
            tr_idx = train_idx[tr_mask]
            model  = self._bias_corrector(col_data.iloc[tr_idx].values, residuals[tr_idx], cat_idx)
            oof[val_idx] = model.predict(col_data.iloc[val_idx].values)
        return oof

    def _fit_full_corrector(self, df, y, score_col, bias_features):
        col_data = df[bias_features].copy()
        col_data["cohort"] = col_data["cohort"].astype("category").cat.codes.astype(float)
        cat_idx  = [bias_features.index("cohort")]
        score_vals = df[score_col].values.astype(float)
        has_score  = ~np.isnan(score_vals)
        residuals  = y.values - score_vals
        return self._bias_corrector(
            col_data[has_score].values, residuals[has_score], cat_idx
        ), col_data.columns.tolist()

    @staticmethod
    def _apply_corrector(model, df, bias_features, cohort_categories):
        col_data = df[bias_features].copy()
        col_data["cohort"] = (
            col_data["cohort"]
            .astype(pd.CategoricalDtype(categories=cohort_categories))
            .cat.codes.astype(float)
        )
        return model.predict(col_data.values)

    def fit(self, df_raw, y, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids is required")
        self._cohort_cats_ = df_raw["cohort"].astype("category").cat.categories.tolist()
        oof_off = self._cross_fit_corrections(df_raw, y, "off", _BIAS_FEATURES_OFF, patient_ids)
        oof_on  = self._cross_fit_corrections(df_raw, y, "on",  _BIAS_FEATURES_ON,  patient_ids)
        self._model_off_, _ = self._fit_full_corrector(df_raw, y, "off", _BIAS_FEATURES_OFF)
        self._model_on_,  _ = self._fit_full_corrector(df_raw, y, "on",  _BIAS_FEATURES_ON)
        self._train_index_ = df_raw.index.copy()
        self._oof_off_ = pd.Series(oof_off, index=df_raw.index)
        self._oof_on_  = pd.Series(oof_on,  index=df_raw.index)
        return self

    def get_corrections(self, df_raw):
        """Return (off_corr, on_corr) arrays aligned with df_raw."""
        train_mask = df_raw.index.isin(self._train_index_)
        off_vals = df_raw["off"].values.astype(float)
        on_vals  = df_raw["on"].values.astype(float)
        g_off = np.full(len(df_raw), np.nan)
        g_on  = np.full(len(df_raw), np.nan)
        if train_mask.any():
            g_off[train_mask] = self._oof_off_.reindex(df_raw.index[train_mask]).values
            g_on[train_mask]  = self._oof_on_.reindex(df_raw.index[train_mask]).values
        if (~train_mask).any():
            test_df = df_raw[~train_mask]
            g_off[~train_mask] = self._apply_corrector(
                self._model_off_, test_df, _BIAS_FEATURES_OFF, self._cohort_cats_
            )
            g_on[~train_mask] = self._apply_corrector(
                self._model_on_, test_df, _BIAS_FEATURES_ON, self._cohort_cats_
            )
        off_corr = np.where(np.isnan(off_vals), np.nan, off_vals + g_off)
        on_corr  = np.where(np.isnan(on_vals),  np.nan, on_vals  + g_on)
        return off_corr, on_corr

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        off_corr, on_corr = self.get_corrections(df_raw)
        pids  = patient_ids if patient_ids is not None else df["patient_id"].values
        times = df["time_since_diagnosis"].values.astype(float)
        n     = len(df)

        feat_off_fitted  = np.full(n, np.nan)
        feat_on_fitted   = np.full(n, np.nan)
        feat_pool_fitted = np.full(n, np.nan)
        feat_pool_mean   = np.full(n, np.nan)
        feat_n_off       = np.zeros(n, dtype=float)
        feat_n_on        = np.zeros(n, dtype=float)

        unique_pids = np.unique(pids)
        for pid in unique_pids:
            mask = pids == pid
            idx  = np.where(mask)[0]
            t    = times[idx]
            oc   = off_corr[idx]
            nc   = on_corr[idx]

            valid_off = ~np.isnan(oc)
            n_off = valid_off.sum()
            if n_off >= 3:
                s, b = _linreg_slope_intercept(t, oc)
                feat_off_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_off > 0:
                feat_off_fitted[idx] = np.nanmean(oc[valid_off])

            valid_on = ~np.isnan(nc)
            n_on = valid_on.sum()
            if n_on >= 3:
                s, b = _linreg_slope_intercept(t, nc)
                feat_on_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_on > 0:
                feat_on_fitted[idx] = np.nanmean(nc[valid_on])

            all_vals  = np.concatenate([oc, nc])
            all_times = np.concatenate([t, t])
            valid_pool = ~np.isnan(all_vals)
            n_pool = valid_pool.sum()
            if n_pool >= 3:
                s, b = _linreg_slope_intercept(all_times, all_vals)
                feat_pool_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_pool > 0:
                feat_pool_fitted[idx] = np.nanmean(all_vals[valid_pool])

            feat_pool_mean[idx] = np.nanmean(all_vals) if valid_pool.sum() > 0 else np.nan
            feat_n_off[idx] = n_off
            feat_n_on[idx]  = n_on

        df["off_corr_pt_fitted"]  = feat_off_fitted
        df["on_corr_pt_fitted"]   = feat_on_fitted
        df["pool_corr_pt_fitted"] = feat_pool_fitted
        df["pool_corr_pt_mean"]   = feat_pool_mean
        df["pt_n_off_corr"]       = feat_n_off
        df["pt_n_on_corr"]        = feat_n_on
        return df


# %% — MixedLine transformer (from 07_mixed)
class MixedLine(BaseEstimator, TransformerMixin):
    """Empirical-Bayes linear mixed model — see 07_mixed for full docstring."""

    def __init__(self, pcp: PatientCorrectedPooling):
        self.pcp = pcp

    @staticmethod
    def _patient_params(pids, times, z_off, z_on, y):
        unique_pids = np.unique(pids)
        params = {}
        for pid in unique_pids:
            mask = pids == pid
            t    = times[mask]
            zo   = z_off[mask]
            zn   = z_on[mask]
            yp   = y[mask]
            s, a = _linreg_slope_intercept(t, yp)
            if np.isnan(s):
                params[pid] = (np.nan, np.nan, np.nan, np.nan)
                continue
            fit_y = a + s * t
            valid_off = ~np.isnan(zo)
            d_off = float(np.mean(zo[valid_off] - fit_y[valid_off])) if valid_off.sum() > 0 else np.nan
            valid_on  = ~np.isnan(zn)
            d_on  = float(np.mean(zn[valid_on]  - fit_y[valid_on]))  if valid_on.sum()  > 0 else np.nan
            params[pid] = (float(a), float(s), d_off, d_on)
        return params

    @staticmethod
    def _empirical_prior(params: dict):
        pids = list(params.keys())
        P = np.array([params[p] for p in pids], dtype=float)
        mu = np.nanmean(P, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        P_filled = P.copy()
        for j in range(4):
            col = P[:, j]
            col_mean = np.nanmean(col)
            fill_val = float(col_mean) if not np.isnan(col_mean) else 0.0
            P_filled[np.isnan(col), j] = fill_val
        S = np.cov(P_filled.T, ddof=1)
        S = np.where(np.isnan(S), 0.0, S)
        return mu, S

    @staticmethod
    def _noise_variances(pids, times, z_off, z_on, params):
        ss_off, n_off = 0.0, 0
        ss_on,  n_on  = 0.0, 0
        for pid in np.unique(pids):
            a, b, d_off, d_on = params[pid]
            if np.isnan(a):
                continue
            mask = pids == pid
            t    = times[mask]
            zo   = z_off[mask]
            zn   = z_on[mask]
            fit  = a + b * t
            valid_off = ~np.isnan(zo)
            if valid_off.sum() > 0 and not np.isnan(d_off):
                res_off = zo[valid_off] - fit[valid_off] - d_off
                ss_off += float((res_off ** 2).sum())
                n_off  += int(valid_off.sum())
            valid_on = ~np.isnan(zn)
            if valid_on.sum() > 0 and not np.isnan(d_on):
                res_on = zn[valid_on] - fit[valid_on] - d_on
                ss_on += float((res_on ** 2).sum())
                n_on  += int(valid_on.sum())
        s2_off = ss_off / max(n_off - 1, 1)
        s2_on  = ss_on  / max(n_on  - 1, 1)
        return s2_off, s2_on

    def _posterior(self, t, z_off_p, z_on_p):
        mu = self.mu_; S = self.S_
        s2_off = self.s2_off_; s2_on = self.s2_on_
        rows_X = []; rows_z = []; rows_r = []
        valid_off = ~np.isnan(z_off_p)
        for i, (ti, zi) in enumerate(zip(t, z_off_p)):
            if valid_off[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 1.0, 0.0]); rows_z.append(zi); rows_r.append(s2_off)
        valid_on = ~np.isnan(z_on_p)
        for i, (ti, zi) in enumerate(zip(t, z_on_p)):
            if valid_on[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 0.0, 1.0]); rows_z.append(zi); rows_r.append(s2_on)
        S_reg = S + np.eye(4) * 1e-6
        S_inv = np.linalg.inv(S_reg)
        if len(rows_X) == 0:
            return mu.copy(), np.linalg.inv(S_inv)
        X  = np.array(rows_X); z = np.array(rows_z); r = np.array(rows_r)
        R_inv = np.diag(1.0 / r)
        precision = S_inv + X.T @ R_inv @ X
        V = np.linalg.inv(precision + np.eye(4) * 1e-8)
        m = V @ (S_inv @ mu + X.T @ R_inv @ z)
        return m, V

    def fit(self, df_raw, y, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids is required")
        z_off, z_on = self.pcp.get_corrections(df_raw)
        pids  = np.asarray(patient_ids)
        times = df_raw["time_since_diagnosis"].values.astype(float)
        y_arr = np.asarray(y, dtype=float)
        params = self._patient_params(pids, times, z_off, z_on, y_arr)
        mu, S  = self._empirical_prior(params)
        s2_off, s2_on = self._noise_variances(pids, times, z_off, z_on, params)
        self.mu_ = mu; self.S_ = S
        self.s2_off_ = max(s2_off, 1e-6); self.s2_on_ = max(s2_on, 1e-6)
        return self

    def transform(self, df_raw, patient_ids=None):
        df   = df_raw.copy()
        pids = np.asarray(patient_ids if patient_ids is not None else df_raw["patient_id"].values)
        z_off, z_on = self.pcp.get_corrections(df_raw)
        times = df_raw["time_since_diagnosis"].values.astype(float)
        n     = len(df_raw)
        ml_pred  = np.full(n, np.nan); ml_std   = np.full(n, np.nan)
        ml_slope = np.full(n, np.nan); ml_d_off = np.full(n, np.nan); ml_d_on = np.full(n, np.nan)
        for pid in np.unique(pids):
            mask = pids == pid; idx = np.where(mask)[0]
            t_p = times[idx]; zo_p = z_off[idx]; zn_p = z_on[idx]
            m, V = self._posterior(t_p, zo_p, zn_p)
            raw_pred_p = m[0] + m[1] * t_p
            pred_p = np.where(np.isnan(raw_pred_p), m[0], raw_pred_p)
            std_p  = np.array([
                np.sqrt(max(np.array([1.0, ti, 0.0, 0.0]) @ V @ np.array([1.0, ti if not np.isnan(ti) else 0.0, 0.0, 0.0]), 0.0))
                for ti in t_p
            ])
            ml_pred[idx] = pred_p; ml_std[idx] = std_p
            ml_slope[idx] = m[1]; ml_d_off[idx] = m[2]; ml_d_on[idx] = m[3]
        df["ml_pred"] = ml_pred; df["ml_std"] = ml_std
        df["ml_slope"] = ml_slope; df["ml_d_off"] = ml_d_off; df["ml_d_on"] = ml_d_on
        return df


# %% — PatientSmoothedRegressor (from 04_hgb_smooth)
class PatientSmoothedRegressor(BaseEstimator, RegressorMixin):
    """Wrap a base regressor with per-patient OLS smoothing on predictions."""

    def __init__(self, base, alpha: float = 0.0):
        self.base = base
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        self.base_ = clone(self.base)
        X_feat = X.drop(columns=["patient_id"])
        self.base_.fit(X_feat, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = X.drop(columns=["patient_id"])
        p = self.base_.predict(X_feat)
        if self.alpha == 0:
            return p
        output = p.copy()
        time = X["time_since_diagnosis"].values.astype(float)
        pid = X["patient_id"].values
        for patient in np.unique(pid):
            mask = pid == patient
            if mask.sum() < 3:
                continue
            t = time[mask]; pv = p[mask]
            s, b = _linreg_slope_intercept(t, pv)
            if np.isnan(s):
                continue
            fitted = s * t + b
            output[mask] = self.alpha * fitted + (1 - self.alpha) * pv
        return output


# %% — Load data
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")

groups_train = X_train_raw["patient_id"]

gkf       = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_raw, y_train, groups=groups_train))


# %% — Feature preparation helpers (07-B style: PCP + MixedLine + patient features)
def _add_tsd(df):
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


def prepare_B_with_pid(df_raw, pcp, ml, pids):
    """07-B features with patient_id kept for PatientSmoothedRegressor."""
    df = _add_tsd(df_raw)
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df)
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    # patient_id kept — PatientSmoothedRegressor drops it before base.predict
    # drop only internal id copy if duplicated
    drop_cols = [c for c in df.columns if c == "patient_id"]
    # keep exactly one patient_id column
    return df


# %% — Build full-data PCP + MixedLine (used for skore.evaluate CV reports)
X_train_tsd = _add_tsd(X_train_raw)
X_test_tsd  = _add_tsd(X_test_raw)

pcp_full = PatientCorrectedPooling(n_splits=5)
pcp_full.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

ml_full = MixedLine(pcp=pcp_full)
ml_full.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

X_train_B = prepare_B_with_pid(X_train_tsd, pcp_full, ml_full, groups_train.values)
X_test_B  = prepare_B_with_pid(X_test_tsd,  pcp_full, ml_full, X_test_tsd["patient_id"].values)

# CV splits on B features (same patient grouping)
cv_splits_B = list(gkf.split(X_train_B, y_train, groups=groups_train))


# %% — HGB hyperparameter grid (6 configurations)
# Strategy: vary learning_rate + max_iter (with early stopping), max_leaf_nodes,
# min_samples_leaf, l2_regularization relative to a sensible default baseline.
HGB_CONFIGS = [
    # label,  kwargs
    ("cfg1_baseline",  dict(learning_rate=0.1,  max_iter=300,  max_leaf_nodes=31,  min_samples_leaf=20, l2_regularization=0.0,  early_stopping=True, random_state=0)),
    ("cfg2_slow_lr",   dict(learning_rate=0.05, max_iter=600,  max_leaf_nodes=31,  min_samples_leaf=20, l2_regularization=0.0,  early_stopping=True, random_state=0)),
    ("cfg3_deep",      dict(learning_rate=0.05, max_iter=600,  max_leaf_nodes=63,  min_samples_leaf=10, l2_regularization=0.0,  early_stopping=True, random_state=0)),
    ("cfg4_shallow",   dict(learning_rate=0.05, max_iter=600,  max_leaf_nodes=15,  min_samples_leaf=30, l2_regularization=0.0,  early_stopping=True, random_state=0)),
    ("cfg5_reg",       dict(learning_rate=0.05, max_iter=600,  max_leaf_nodes=31,  min_samples_leaf=20, l2_regularization=1.0,  early_stopping=True, random_state=0)),
    ("cfg6_slow_deep", dict(learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,  min_samples_leaf=15, l2_regularization=0.5,  early_stopping=True, random_state=0)),
]

# %% — Evaluate 6 HGB configs with skore (PatientSmoothedRegressor alpha=0 wrapper)
print("=== Step 1: HGB configuration sweep ===")
hgb_reports = {}
hgb_rmse    = {}

for label, kwargs in HGB_CONFIGS:
    print(f"  Evaluating {label}...")
    est = PatientSmoothedRegressor(
        base=HistGradientBoostingRegressor(**kwargs),
        alpha=0.0,
    )
    rep = skore.evaluate(est, X_train_B, y_train, splitter=cv_splits_B)
    hgb_reports[label] = rep
    s = rep.metrics.summarize().frame()
    mean_col = [c for c in s.columns if c.endswith("_mean")][0]
    std_col  = [c for c in s.columns if c.endswith("_std")][0]
    hgb_rmse[label] = (float(s.loc["rmse", mean_col]), float(s.loc["rmse", std_col]))
    print(f"    RMSE: {hgb_rmse[label][0]:.4f} ± {hgb_rmse[label][1]:.4f}")

best_hgb_label = min(hgb_rmse, key=lambda k: hgb_rmse[k][0])
best_hgb_kwargs = dict(HGB_CONFIGS)
best_hgb_cfg = best_hgb_kwargs[best_hgb_label]
print(f"\nBest HGB config: {best_hgb_label} -> RMSE {hgb_rmse[best_hgb_label][0]:.4f} ± {hgb_rmse[best_hgb_label][1]:.4f}")
print(f"  Params: {best_hgb_cfg}")

# %% — Step 2: Wrap best HGB in PatientSmoothedRegressor with alpha ∈ {0, 0.3, 0.5}
print("\n=== Step 2: PatientSmoothedRegressor alpha sweep ===")
ALPHAS = [0, 0.3, 0.5]
smooth_reports = {}
smooth_rmse    = {}

for alpha in ALPHAS:
    label = f"smooth_alpha={alpha}"
    print(f"  Evaluating {label}...")
    est = PatientSmoothedRegressor(
        base=HistGradientBoostingRegressor(**best_hgb_cfg),
        alpha=alpha,
    )
    rep = skore.evaluate(est, X_train_B, y_train, splitter=cv_splits_B)
    smooth_reports[label] = rep
    s = rep.metrics.summarize().frame()
    mean_col = [c for c in s.columns if c.endswith("_mean")][0]
    std_col  = [c for c in s.columns if c.endswith("_std")][0]
    smooth_rmse[label] = (float(s.loc["rmse", mean_col]), float(s.loc["rmse", std_col]))
    print(f"    RMSE: {smooth_rmse[label][0]:.4f} ± {smooth_rmse[label][1]:.4f}")

# %% — Combine all reports into one ComparisonReport and push to hub
all_reports = {**hgb_reports, **smooth_reports}
comparison = ComparisonReport(all_reports)
print("\nPushing all reports to hub under key 08_polish...")
for key, rep in all_reports.items():
    project.put(f"08_polish_{key}", rep)

HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Hub URL: {HUB_URL}")

# %% — Determine best overall model
BASELINE_RMSE = 4.05
all_rmse = {**hgb_rmse, **smooth_rmse}
best_overall_label = min(all_rmse, key=lambda k: all_rmse[k][0])
best_overall_rmse  = all_rmse[best_overall_label][0]

print(f"\n=== Summary ===")
print(f"Best HGB config:  {best_hgb_label} -> {hgb_rmse[best_hgb_label][0]:.4f} ± {hgb_rmse[best_hgb_label][1]:.4f}")
for alpha in ALPHAS:
    label = f"smooth_alpha={alpha}"
    print(f"  {label}: {smooth_rmse[label][0]:.4f} ± {smooth_rmse[label][1]:.4f}")
print(f"Best overall: {best_overall_label} -> RMSE {best_overall_rmse:.4f}")

# %% — Holdout report, final fit, and submission if best beats 4.05
if best_overall_rmse < BASELINE_RMSE:
    from skore import EstimatorReport  # noqa: E402
    print(f"\nBeats baseline {BASELINE_RMSE} — running holdout report and submission.")

    # Determine best alpha from smooth sweep
    best_smooth_label = min(smooth_rmse, key=lambda k: smooth_rmse[k][0])
    import re as _re
    _m = _re.search(r"alpha=([0-9.]+)", best_smooth_label)
    best_alpha = float(_m.group(1)) if _m else 0.0

    # Patient-grouped 80/20 holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    X_hold_tr_raw  = _add_tsd(X_train_raw.iloc[hold_tr_idx])
    X_hold_val_raw = _add_tsd(X_train_raw.iloc[hold_val_idx])
    y_hold_tr  = y_train.iloc[hold_tr_idx]
    y_hold_val = y_train.iloc[hold_val_idx]
    pids_hold_tr  = groups_train.iloc[hold_tr_idx].values
    pids_hold_val = groups_train.iloc[hold_val_idx].values

    pcp_hold = PatientCorrectedPooling(n_splits=5)
    pcp_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)
    ml_hold = MixedLine(pcp=pcp_hold)
    ml_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)

    X_hold_tr_B  = prepare_B_with_pid(X_hold_tr_raw,  pcp_hold, ml_hold, pids_hold_tr)
    X_hold_val_B = prepare_B_with_pid(X_hold_val_raw, pcp_hold, ml_hold, pids_hold_val)

    report_final = EstimatorReport(
        PatientSmoothedRegressor(
            base=HistGradientBoostingRegressor(**best_hgb_cfg),
            alpha=best_alpha,
        ),
        X_train=X_hold_tr_B,
        y_train=y_hold_tr,
        X_test=X_hold_val_B,
        y_test=y_hold_val,
    )
    project.put("08_polish_final", report_final)
    print(f"Pushed 08_polish_final to hub: {HUB_URL}")

    # Fit on all train
    model_final = PatientSmoothedRegressor(
        base=HistGradientBoostingRegressor(**best_hgb_cfg),
        alpha=best_alpha,
    )
    model_final.fit(X_train_B, y_train)
    preds_test = model_final.predict(X_test_B)

    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    sub_path = submissions_dir / "08_polish.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission written: {len(submission)} rows -> submissions/08_polish.csv")
else:
    print(f"\nBest RMSE {best_overall_rmse:.4f} does not beat baseline {BASELINE_RMSE} — skipping holdout and submission.")
