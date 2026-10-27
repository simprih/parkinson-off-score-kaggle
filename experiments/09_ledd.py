# %% [markdown]
# # Experiment 09 — ledd
#
# **Key:** 09_ledd  |  **Date:** 2026-10-09  |  **Kept:** merged into 10
# **Idea:** Per-patient linear interpolation of ledd over age (LeddInterp).
# **CV (GroupKFold 5):** not scored
# LeddInterp was merged into experiment 10 and kept in all subsequent pipelines.
# **Skore Hub:** report keys `09_ledd`, `09_ledd_final` (ids not listed in the README).
# **Run:** `python experiments/09_ledd.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/09_ledd.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see README, How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login, ComparisonReport, EstimatorReport
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "09_ledd"

# %% — Open skore Hub project
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% — Shared helpers
_BIAS_FEATURES_OFF = ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]
_BIAS_FEATURES_ON  = ["time_since_intake_on",  "cohort", "ledd", "time_since_diagnosis"]

_BIAS_FEATURES_OFF_I = ["time_since_intake_off", "cohort", "ledd_interp", "time_since_diagnosis"]
_BIAS_FEATURES_ON_I  = ["time_since_intake_on",  "cohort", "ledd_interp", "time_since_diagnosis"]


def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
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


def add_patient_features(df: pd.DataFrame, ledd_col: str = "ledd") -> pd.DataFrame:
    """Patient-level aggregated features; ledd_col picks the ledd column for stats."""
    df = df.copy()
    for col in ["off", "on"]:
        stats = df.groupby("patient_id")[col].agg(
            **{
                f"{col}_pt_mean":   "mean",
                f"{col}_pt_median": "median",
                f"{col}_pt_std":    "std",
                f"{col}_pt_count":  lambda s: s.notna().sum(),
            }
        )
        df = df.join(stats, on="patient_id")
    # ledd stats use the chosen column but store as canonical ledd_pt_*
    ledd_stats = df.groupby("patient_id")[ledd_col].agg(
        ledd_pt_mean=   "mean",
        ledd_pt_median= "median",
        ledd_pt_std=    "std",
        ledd_pt_count=  lambda s: s.notna().sum(),
    )
    df = df.join(ledd_stats, on="patient_id")
    for score_col in ["off", "on"]:
        slopes, intercepts, fitted = {}, {}, {}
        for pid, grp in df.groupby("patient_id"):
            x = grp["time_since_diagnosis"].values.astype(float)
            y = grp[score_col].values.astype(float)
            s, b = _linreg_slope_intercept(x, y)
            slopes[pid] = s
            intercepts[pid] = b
            fv = np.where(np.isnan(s), np.nan, s * x + b)
            for idx, v in zip(grp.index, fv):
                fitted[idx] = v
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


# %% — LeddInterp transformer (variant B+)
class LeddInterp(BaseEstimator, TransformerMixin):
    """Per-patient linear interpolation of ledd over age (computed from X only).

    Adds:
      - ledd_interp : linearly interpolated ledd, constant-extrapolated beyond
                      first/last known value; NaN if patient has no ledd at all.
      - ledd_missing: 1 if original ledd is NaN, else 0.
      - ledd_slope  : per-patient OLS slope of ledd over age (NaN if < 2 known).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        n = len(df)
        ledd_interp = df["ledd"].values.astype(float).copy()
        ledd_missing = np.where(np.isnan(df["ledd"].values), 1.0, 0.0)
        ledd_slope_col = np.full(n, np.nan)

        pids  = df["patient_id"].values
        ages  = df["age"].values.astype(float)
        ledd_vals = df["ledd"].values.astype(float)

        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            a_p  = ages[idx]
            l_p  = ledd_vals[idx]
            known = ~np.isnan(l_p)
            if known.sum() == 0:
                ledd_interp[idx] = np.nan
                continue
            s, _ = _linreg_slope_intercept(a_p[known], l_p[known])
            ledd_slope_col[idx] = s
            if known.sum() == 1:
                ledd_interp[idx] = l_p[known][0]
                continue
            order = np.argsort(a_p)
            a_sorted = a_p[order]
            known_sorted = known[order]
            l_sorted = l_p[order]
            known_ages  = a_sorted[known_sorted]
            known_ledds = l_sorted[known_sorted]
            interp_vals = np.interp(a_sorted, known_ages, known_ledds)
            inv_order = np.argsort(order)
            ledd_interp[idx] = interp_vals[inv_order]

        df["ledd_interp"] = ledd_interp
        df["ledd_missing"] = ledd_missing
        df["ledd_slope"]   = ledd_slope_col
        return df


# %% — PatientCorrectedPooling
class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction (cross-fitted on training fold).

    use_ledd_interp=True swaps ledd for ledd_interp in the bias-correction features.
    """

    def __init__(self, n_splits: int = 5, use_ledd_interp: bool = False):
        self.n_splits = n_splits
        self.use_ledd_interp = use_ledd_interp

    @property
    def _bias_off(self):
        return _BIAS_FEATURES_OFF_I if self.use_ledd_interp else _BIAS_FEATURES_OFF

    @property
    def _bias_on(self):
        return _BIAS_FEATURES_ON_I if self.use_ledd_interp else _BIAS_FEATURES_ON

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
        oof_off = self._cross_fit_corrections(df_raw, y, "off", self._bias_off, patient_ids)
        oof_on  = self._cross_fit_corrections(df_raw, y, "on",  self._bias_on,  patient_ids)
        self._model_off_, _ = self._fit_full_corrector(df_raw, y, "off", self._bias_off)
        self._model_on_,  _ = self._fit_full_corrector(df_raw, y, "on",  self._bias_on)
        self._train_index_ = df_raw.index.copy()
        self._oof_off_ = pd.Series(oof_off, index=df_raw.index)
        self._oof_on_  = pd.Series(oof_on,  index=df_raw.index)
        return self

    def get_corrections(self, df_raw):
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
                self._model_off_, test_df, self._bias_off, self._cohort_cats_
            )
            g_on[~train_mask] = self._apply_corrector(
                self._model_on_, test_df, self._bias_on, self._cohort_cats_
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
        for pid in np.unique(pids):
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


# %% — MixedLine
class MixedLine(BaseEstimator, TransformerMixin):
    def __init__(self, pcp: PatientCorrectedPooling):
        self.pcp = pcp

    @staticmethod
    def _patient_params(pids, times, z_off, z_on, y):
        params = {}
        for pid in np.unique(pids):
            mask = pids == pid
            t = times[mask]; zo = z_off[mask]; zn = z_on[mask]; yp = y[mask]
            s, a = _linreg_slope_intercept(t, yp)
            if np.isnan(s):
                params[pid] = (np.nan, np.nan, np.nan, np.nan); continue
            fit_y = a + s * t
            valid_off = ~np.isnan(zo)
            d_off = float(np.mean(zo[valid_off] - fit_y[valid_off])) if valid_off.sum() > 0 else np.nan
            valid_on = ~np.isnan(zn)
            d_on = float(np.mean(zn[valid_on] - fit_y[valid_on])) if valid_on.sum() > 0 else np.nan
            params[pid] = (float(a), float(s), d_off, d_on)
        return params

    @staticmethod
    def _empirical_prior(params):
        pids = list(params.keys())
        P = np.array([params[p] for p in pids], dtype=float)
        mu = np.nanmean(P, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu)
        P_filled = P.copy()
        for j in range(4):
            col = P[:, j]; cm = np.nanmean(col)
            P_filled[np.isnan(col), j] = float(cm) if not np.isnan(cm) else 0.0
        S = np.cov(P_filled.T, ddof=1)
        S = np.where(np.isnan(S), 0.0, S)
        return mu, S

    @staticmethod
    def _noise_variances(pids, times, z_off, z_on, params):
        ss_off, n_off = 0.0, 0; ss_on, n_on = 0.0, 0
        for pid in np.unique(pids):
            a, b, d_off, d_on = params[pid]
            if np.isnan(a): continue
            mask = pids == pid; t = times[mask]; zo = z_off[mask]; zn = z_on[mask]; fit = a + b * t
            valid_off = ~np.isnan(zo)
            if valid_off.sum() > 0 and not np.isnan(d_off):
                res_off = zo[valid_off] - fit[valid_off] - d_off
                ss_off += float((res_off ** 2).sum()); n_off += int(valid_off.sum())
            valid_on = ~np.isnan(zn)
            if valid_on.sum() > 0 and not np.isnan(d_on):
                res_on = zn[valid_on] - fit[valid_on] - d_on
                ss_on += float((res_on ** 2).sum()); n_on += int(valid_on.sum())
        return ss_off / max(n_off - 1, 1), ss_on / max(n_on - 1, 1)

    def _posterior(self, t, z_off_p, z_on_p):
        mu = self.mu_; S = self.S_; s2_off = self.s2_off_; s2_on = self.s2_on_
        rows_X = []; rows_z = []; rows_r = []
        valid_off = ~np.isnan(z_off_p)
        for i, (ti, zi) in enumerate(zip(t, z_off_p)):
            if valid_off[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 1.0, 0.0]); rows_z.append(zi); rows_r.append(s2_off)
        valid_on = ~np.isnan(z_on_p)
        for i, (ti, zi) in enumerate(zip(t, z_on_p)):
            if valid_on[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 0.0, 1.0]); rows_z.append(zi); rows_r.append(s2_on)
        S_reg = S + np.eye(4) * 1e-6; S_inv = np.linalg.inv(S_reg)
        if len(rows_X) == 0:
            return mu.copy(), np.linalg.inv(S_inv)
        X = np.array(rows_X); z = np.array(rows_z); r = np.array(rows_r)
        R_inv = np.diag(1.0 / r)
        precision = S_inv + X.T @ R_inv @ X
        V = np.linalg.inv(precision + np.eye(4) * 1e-8)
        m = V @ (S_inv @ mu + X.T @ R_inv @ z)
        return m, V

    def fit(self, df_raw, y, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids is required")
        z_off, z_on = self.pcp.get_corrections(df_raw)
        pids = np.asarray(patient_ids); times = df_raw["time_since_diagnosis"].values.astype(float)
        y_arr = np.asarray(y, dtype=float)
        params = self._patient_params(pids, times, z_off, z_on, y_arr)
        mu, S = self._empirical_prior(params)
        s2_off, s2_on = self._noise_variances(pids, times, z_off, z_on, params)
        self.mu_ = mu; self.S_ = S
        self.s2_off_ = max(s2_off, 1e-6); self.s2_on_ = max(s2_on, 1e-6)
        return self

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        pids = np.asarray(patient_ids if patient_ids is not None else df_raw["patient_id"].values)
        z_off, z_on = self.pcp.get_corrections(df_raw)
        times = df_raw["time_since_diagnosis"].values.astype(float)
        n = len(df_raw)
        ml_pred = np.full(n, np.nan); ml_std = np.full(n, np.nan)
        ml_slope = np.full(n, np.nan); ml_d_off = np.full(n, np.nan); ml_d_on = np.full(n, np.nan)
        for pid in np.unique(pids):
            mask = pids == pid; idx = np.where(mask)[0]
            t_p = times[idx]; zo_p = z_off[idx]; zn_p = z_on[idx]
            m, V = self._posterior(t_p, zo_p, zn_p)
            raw_pred_p = m[0] + m[1] * t_p
            pred_p = np.where(np.isnan(raw_pred_p), m[0], raw_pred_p)
            std_p = np.array([
                np.sqrt(max(np.array([1.0, ti, 0.0, 0.0]) @ V @ np.array([1.0, ti if not np.isnan(ti) else 0.0, 0.0, 0.0]), 0.0))
                for ti in t_p
            ])
            ml_pred[idx] = pred_p; ml_std[idx] = std_p
            ml_slope[idx] = m[1]; ml_d_off[idx] = m[2]; ml_d_on[idx] = m[3]
        df["ml_pred"] = ml_pred; df["ml_std"] = ml_std
        df["ml_slope"] = ml_slope; df["ml_d_off"] = ml_d_off; df["ml_d_on"] = ml_d_on
        return df


# %% — PatientSmoothedRegressor
class PatientSmoothedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base, alpha: float = 0.0):
        self.base = base
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        self.base_ = clone(self.base)
        self.base_.fit(X.drop(columns=["patient_id"]), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.base_.predict(X.drop(columns=["patient_id"]))
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
            output[mask] = self.alpha * (s * t + b) + (1 - self.alpha) * pv
        return output


# %% — EnsembleRandomStateRegressor (variant D)
class EnsembleRandomStateRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_kwargs: dict, alpha: float, random_states=None):
        self.base_kwargs = base_kwargs
        self.alpha = alpha
        self.random_states = random_states or [0, 1, 2, 3, 4]

    def fit(self, X: pd.DataFrame, y):
        self.estimators_ = []
        for rs in self.random_states:
            est = PatientSmoothedRegressor(
                base=HistGradientBoostingRegressor(**{**self.base_kwargs, "random_state": rs}),
                alpha=self.alpha,
            )
            est.fit(X, y)
            self.estimators_.append(est)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.stack([e.predict(X) for e in self.estimators_], axis=0).mean(axis=0)


# %% — Load data
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")
groups_train = X_train_raw["patient_id"]
gkf          = GroupKFold(n_splits=5)
cv_splits    = list(gkf.split(X_train_raw, y_train, groups=groups_train))


def _add_tsd(df):
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


# %% — Pre-build feature sets with parquet caching (slow PCP/MixedLine only runs once)
import pickle  # noqa: E402

_CACHE_DIR = PROJECT_ROOT / "scratch" / "09_ledd_feats"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_A = _CACHE_DIR / "feats_A.pkl"
_CACHE_B = _CACHE_DIR / "feats_B.pkl"

X_train_tsd = _add_tsd(X_train_raw)
X_test_tsd  = _add_tsd(X_test_raw)

if _CACHE_A.exists():
    print("Loading variant A features from cache...")
    with open(_CACHE_A, "rb") as f:
        _a = pickle.load(f)
    X_train_A, X_test_A, pcp_A, ml_A = _a["Xtr"], _a["Xte"], _a["pcp"], _a["ml"]
    def _feats_A(df_raw, pids):
        df = _add_tsd(df_raw)
        df = pcp_A.transform(df)
        df = ml_A.transform(df, patient_ids=pids)
        df = add_patient_features(df, ledd_col="ledd")
        df["cohort"] = df["cohort"].astype("category")
        df["gene"]   = df["gene"].astype("category")
        return df
else:
    print("Building variant A features (08-best, raw ledd)...", flush=True)
    pcp_A = PatientCorrectedPooling(n_splits=5, use_ledd_interp=False)
    pcp_A.fit(X_train_tsd, y_train, patient_ids=groups_train.values)
    ml_A = MixedLine(pcp=pcp_A)
    ml_A.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

    def _feats_A(df_raw, pids):
        df = _add_tsd(df_raw)
        df = pcp_A.transform(df)
        df = ml_A.transform(df, patient_ids=pids)
        df = add_patient_features(df, ledd_col="ledd")
        df["cohort"] = df["cohort"].astype("category")
        df["gene"]   = df["gene"].astype("category")
        return df

    X_train_A = _feats_A(X_train_raw, groups_train.values)
    X_test_A  = _feats_A(X_test_raw,  X_test_raw["patient_id"].values)
    with open(_CACHE_A, "wb") as f:
        pickle.dump({"Xtr": X_train_A, "Xte": X_test_A, "pcp": pcp_A, "ml": ml_A}, f)
    print(f"  Cached A features: {X_train_A.shape[1]} cols")

cv_A = list(gkf.split(X_train_A, y_train, groups=groups_train))

ledd_interp_tr = LeddInterp()

if _CACHE_B.exists():
    print("Loading variant B features from cache...")
    with open(_CACHE_B, "rb") as f:
        _b = pickle.load(f)
    X_train_B, X_test_B, pcp_B, ml_B = _b["Xtr"], _b["Xte"], _b["pcp"], _b["ml"]
    def _feats_B(df_raw, pids):
        df = ledd_interp_tr.transform(_add_tsd(df_raw))
        df = pcp_B.transform(df)
        df = ml_B.transform(df, patient_ids=pids)
        df = add_patient_features(df, ledd_col="ledd_interp")
        df["cohort"] = df["cohort"].astype("category")
        df["gene"]   = df["gene"].astype("category")
        return df
else:
    print("Building variant B features (ledd_interp)...", flush=True)
    X_train_li = ledd_interp_tr.transform(X_train_tsd)

    pcp_B = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
    pcp_B.fit(X_train_li, y_train, patient_ids=groups_train.values)
    ml_B = MixedLine(pcp=pcp_B)
    ml_B.fit(X_train_li, y_train, patient_ids=groups_train.values)

    def _feats_B(df_raw, pids):
        df = ledd_interp_tr.transform(_add_tsd(df_raw))
        df = pcp_B.transform(df)
        df = ml_B.transform(df, patient_ids=pids)
        df = add_patient_features(df, ledd_col="ledd_interp")
        df["cohort"] = df["cohort"].astype("category")
        df["gene"]   = df["gene"].astype("category")
        return df

    X_train_B = _feats_B(X_train_raw, groups_train.values)
    X_test_B  = _feats_B(X_test_raw,  X_test_raw["patient_id"].values)
    with open(_CACHE_B, "wb") as f:
        pickle.dump({"Xtr": X_train_B, "Xte": X_test_B, "pcp": pcp_B, "ml": ml_B}, f)
    print(f"  Cached B features: {X_train_B.shape[1]} cols")

cv_B = list(gkf.split(X_train_B, y_train, groups=groups_train))
print(f"  A cols: {X_train_A.shape[1]}, B cols: {X_train_B.shape[1]}", flush=True)


# %% — Manual CV helper
def manual_cv_rmse(est_proto, X_feat, cv_splits_v, y=y_train):
    """5-fold GroupKFold CV; returns (mean_rmse, std_rmse, oof_preds)."""
    oof = np.full(len(y), np.nan)
    for tr_idx, val_idx in cv_splits_v:
        e = clone(est_proto)
        e.fit(X_feat.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = e.predict(X_feat.iloc[val_idx])
    fold_rmses = [root_mean_squared_error(y.iloc[vi], oof[vi]) for _, vi in cv_splits_v]
    return float(np.mean(fold_rmses)), float(np.std(fold_rmses, ddof=1)), oof


# Hub EstimatorReports use a single 80/20 patient-grouped holdout (fast — no re-CV).
# CV RMSE (ground truth) comes from manual_cv_rmse above.
_gss_hub = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_hub_tr_idx, _hub_val_idx = next(_gss_hub.split(X_train_A, y_train, groups=groups_train))
y_hub_tr  = y_train.iloc[_hub_tr_idx]
y_hub_val = y_train.iloc[_hub_val_idx]


def _hub_report(est_proto, X_tr, X_val):
    """Build EstimatorReport (fits on tr, evaluates on val) for hub comparison."""
    return EstimatorReport(clone(est_proto), X_train=X_tr, y_train=y_hub_tr,
                           X_test=X_val, y_test=y_hub_val)


X_hub_tr_A  = X_train_A.iloc[_hub_tr_idx]
X_hub_val_A = X_train_A.iloc[_hub_val_idx]
X_hub_tr_B  = X_train_B.iloc[_hub_tr_idx]
X_hub_val_B = X_train_B.iloc[_hub_val_idx]


# %% — Best 08 config
HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0
)
BEST_ALPHA_08 = 0.3

# %% — Evaluate A
print("=== Variant A (control) ===")
est_A = PatientSmoothedRegressor(base=HistGradientBoostingRegressor(**HGB_BEST_08), alpha=BEST_ALPHA_08)
rmse_A_mean, rmse_A_std, oof_A = manual_cv_rmse(est_A, X_train_A, cv_A)
print(f"  RMSE: {rmse_A_mean:.4f} ± {rmse_A_std:.4f}")
rep_A = _hub_report(est_A, X_hub_tr_A, X_hub_val_A)

# %% — Evaluate B
print("=== Variant B (ledd_interp) ===")
est_B = PatientSmoothedRegressor(base=HistGradientBoostingRegressor(**HGB_BEST_08), alpha=BEST_ALPHA_08)
rmse_B_mean, rmse_B_std, oof_B = manual_cv_rmse(est_B, X_train_B, cv_B)
print(f"  RMSE: {rmse_B_mean:.4f} ± {rmse_B_std:.4f}")
rep_B = _hub_report(est_B, X_hub_tr_B, X_hub_val_B)

# %% — Variant C: deeper HGB on B features
C_CONFIGS = [
    ("C1_mln127_lr002", dict(learning_rate=0.02, max_iter=3000, max_leaf_nodes=127,
                             min_samples_leaf=10, l2_regularization=0.5,
                             early_stopping=True, random_state=0)),
    ("C2_mln255_l2_2",  dict(learning_rate=0.02, max_iter=3000, max_leaf_nodes=255,
                             min_samples_leaf=20, l2_regularization=2.0,
                             early_stopping=True, random_state=0)),
    ("C3_mln127_l2_2",  dict(learning_rate=0.02, max_iter=3000, max_leaf_nodes=127,
                             min_samples_leaf=20, l2_regularization=2.0,
                             early_stopping=True, random_state=0)),
]
print("=== Variant C (deeper HGB on B features) ===")
c_results = {}
for lbl, kwargs in C_CONFIGS:
    est = PatientSmoothedRegressor(base=HistGradientBoostingRegressor(**kwargs), alpha=BEST_ALPHA_08)
    m, s, _ = manual_cv_rmse(est, X_train_B, cv_B)
    print(f"  {lbl}: RMSE {m:.4f} ± {s:.4f}")
    c_results[lbl] = (m, s, kwargs)

best_C_label = min(c_results, key=lambda k: c_results[k][0])
rmse_C_mean, rmse_C_std, best_C_kwargs = c_results[best_C_label]
print(f"Best C: {best_C_label} -> RMSE {rmse_C_mean:.4f} ± {rmse_C_std:.4f}")
est_C = PatientSmoothedRegressor(base=HistGradientBoostingRegressor(**best_C_kwargs), alpha=BEST_ALPHA_08)
rep_C = _hub_report(est_C, X_hub_tr_B, X_hub_val_B)

# %% — Variant D: best C config over 5 random states
print("=== Variant D (5-seed ensemble of best C) ===")
est_D = EnsembleRandomStateRegressor(
    base_kwargs={k: v for k, v in best_C_kwargs.items() if k != "random_state"},
    alpha=BEST_ALPHA_08,
    random_states=[0, 1, 2, 3, 4],
)
rmse_D_mean, rmse_D_std, oof_D = manual_cv_rmse(est_D, X_train_B, cv_B)
print(f"  RMSE: {rmse_D_mean:.4f} ± {rmse_D_std:.4f}")
rep_D = _hub_report(est_D, X_hub_tr_B, X_hub_val_B)

# %% — Push comparison to hub
print("\nPushing comparison to hub under key 09_ledd...")
all_reports = {
    "A_control":          rep_A,
    "B_ledd_interp":      rep_B,
    f"C_{best_C_label}":  rep_C,
    "D_ensemble":         rep_D,
}
comparison = ComparisonReport(all_reports)
project.put(EXPERIMENT_KEY, comparison)
for k, rep in all_reports.items():
    project.put(f"{EXPERIMENT_KEY}_{k}", rep)

HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Hub URL: {HUB_URL}")

# %% — Determine best variant
all_rmse = {
    "A": (rmse_A_mean, rmse_A_std),
    "B": (rmse_B_mean, rmse_B_std),
    "C": (rmse_C_mean, rmse_C_std),
    "D": (rmse_D_mean, rmse_D_std),
}
best_variant = min(all_rmse, key=lambda k: all_rmse[k][0])
best_rmse_val = all_rmse[best_variant][0]
print(f"\nBest variant: {best_variant} -> RMSE {best_rmse_val:.4f}")

# %% — Save OOF predictions of best variant
_oof_map = {"A": (oof_A, X_train_A), "B": (oof_B, X_train_B)}
if best_variant in ("C", "D"):
    X_best = X_train_B
    cv_best = cv_B
    est_best_proto = est_D if best_variant == "D" else est_C
    _, _, oof_best = manual_cv_rmse(est_best_proto, X_best, cv_best)
    X_best_test = X_test_B
else:
    oof_best, X_best = _oof_map[best_variant]
    X_best_test = X_test_A if best_variant == "A" else X_test_B
    cv_best = cv_A if best_variant == "A" else cv_B
    est_best_proto = est_A if best_variant == "A" else est_B

print("Saving OOF predictions...")
oof_df = pd.DataFrame({
    "Index":      X_train_raw.index,
    "patient_id": groups_train.values,
    "oof":        oof_best,
})
audit_dir = PROJECT_ROOT / "audit"
audit_dir.mkdir(exist_ok=True)
oof_path = audit_dir / "oof_09.csv"
oof_df.to_csv(oof_path, index=False)
print(f"OOF RMSE: {root_mean_squared_error(y_train, oof_best):.4f} | saved to audit/oof_09.csv")

# %% — Holdout report, final fit, submission if best < 3.93
BASELINE_RMSE = 3.93
if best_rmse_val < BASELINE_RMSE:
    print(f"\nBeats baseline {BASELINE_RMSE} — running holdout report and submission.")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    pids_hold_tr  = groups_train.iloc[hold_tr_idx].values
    pids_hold_val = groups_train.iloc[hold_val_idx].values
    y_hold_tr     = y_train.iloc[hold_tr_idx]
    y_hold_val    = y_train.iloc[hold_val_idx]

    if best_variant == "A":
        X_htr = _feats_A(X_train_raw.iloc[hold_tr_idx], pids_hold_tr)
        X_hval = _feats_A(X_train_raw.iloc[hold_val_idx], pids_hold_val)
    else:
        X_htr = _feats_B(X_train_raw.iloc[hold_tr_idx], pids_hold_tr)
        X_hval = _feats_B(X_train_raw.iloc[hold_val_idx], pids_hold_val)

    report_final = EstimatorReport(
        clone(est_best_proto),
        X_train=X_htr, y_train=y_hold_tr,
        X_test=X_hval,  y_test=y_hold_val,
    )
    project.put("09_ledd_final", report_final)
    print(f"Pushed 09_ledd_final to hub: {HUB_URL}")

    model_final = clone(est_best_proto)
    model_final.fit(X_best, y_train)
    preds_test = model_final.predict(X_best_test)

    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    sub_path = submissions_dir / "09_ledd.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission: {len(submission)} rows -> submissions/09_ledd.csv")
else:
    print(f"\nBest RMSE {best_rmse_val:.4f} does not beat baseline {BASELINE_RMSE} — skipping submission.")

# %% — Final summary
print("\n=== Final Summary ===")
for v, (m, s) in all_rmse.items():
    print(f"  Variant {v}: RMSE {m:.4f} ± {s:.4f}")
print(f"Best variant: {best_variant} (RMSE {best_rmse_val:.4f})")
print(f"Best C config: {best_C_label}")
print(f"Hub URL: {HUB_URL}")
print(f"OOF predictions: audit/oof_09.csv")
