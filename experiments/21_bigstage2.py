# %% [markdown]
# # Experiment 21 — bigger stage-2 HGB (Change O)
#
# **Date:** 2026-10-13
# **Goal:** Improve the under-fit stage-2 HistGradientBoostingRegressor used
# inside TwoStageRegressor.fit for both e_modes.
#
# Change O (stage 2 only; everything else exactly as in Experiment 20):
#   O1. Module-level STAGE2_HGB_PARAMS = dict(max_iter=900, learning_rate=0.02,
#       max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
#       early_stopping=False).
#   O2. Each of the 5 bagged stage-2 models is built as
#       HistGradientBoostingRegressor(**STAGE2_HGB_PARAMS, random_state=seed).
#   O3. First-stage models, sev_evidence, densities, projections, isotonic
#       step and BlendRegressor are unchanged.
#
# Benchmark: Experiment 20 (20_blend) — recomputed from audit/oof_20.csv.
# Keep criterion: mean RMSE < v20 mean AND >= 4 of 5 folds improve vs v20.
#
# **Key:** 21_bigstage2  |  **Decision:** kept, better than 20 on 5/5 folds; replaced by 22 (21 uses the first version of the files)
# **Results:** CV (GroupKFold 5) 2.7237 ± 0.047  |  holdout (20 % of patients) 2.74  |  public LB 2.820
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/71702
# **Run:** `python experiments/21_bigstage2.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/21_bigstage2.py`).
# Reads data/ (first version of the files, before 9 October), audit/oof_19.csv,
# audit/oof_20.csv, audit/oof_20b.csv and submissions/20_blend.csv; needs the Skore key
# (.skore), see README, How to run.
# `SMOKE_TEST=1`: 10 % of the patients, 2 folds, nothing saved or pushed (no Skore key needed).

# %%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login, EstimatorReport
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import math
from scipy.ndimage import convolve1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "21_bigstage2"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SPLITS_CV = 2 if SMOKE else 5

# Benchmark v20 values are recomputed from audit/oof_20.csv at runtime below.
# These placeholders are overwritten before any comparison is made.
BENCHMARK_RMSE  = None   # set after loading oof_20.csv
BENCHMARK_FOLDS = None   # set after loading oof_20.csv

# =============================================================================
# Change O: stage-2 HGB hyper-parameters
# =============================================================================

STAGE2_HGB_PARAMS = dict(
    max_iter=900,
    learning_rate=0.02,
    max_leaf_nodes=63,
    min_samples_leaf=50,
    l2_regularization=1.0,
    early_stopping=False,
)

# %% -- skore hub (skipped in smoke mode)
if not SMOKE:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.load_skore_credentials import load_skore_credentials
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

# =============================================================================
# Shared primitives (verbatim from 19_likelihood)
# =============================================================================

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


class LeddInterp(BaseEstimator, TransformerMixin):
    """Per-patient linear interpolation of ledd over age (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        n = len(df)
        ledd_interp = df["ledd"].values.astype(float).copy()
        ledd_missing = np.where(np.isnan(df["ledd"].values), 1.0, 0.0)
        ledd_slope_col = np.full(n, np.nan)
        pids      = df["patient_id"].values
        ages      = df["age"].values.astype(float)
        ledd_vals = df["ledd"].values.astype(float)
        for pid in np.unique(pids):
            mask  = pids == pid
            idx   = np.where(mask)[0]
            a_p   = ages[idx]
            l_p   = ledd_vals[idx]
            known = ~np.isnan(l_p)
            if known.sum() == 0:
                ledd_interp[idx] = np.nan
                continue
            s, _ = _linreg_slope_intercept(a_p[known], l_p[known])
            ledd_slope_col[idx] = s
            if known.sum() == 1:
                ledd_interp[idx] = l_p[known][0]
                continue
            order         = np.argsort(a_p)
            a_sorted      = a_p[order]
            known_sorted  = known[order]
            l_sorted      = l_p[order]
            known_ages    = a_sorted[known_sorted]
            known_ledds   = l_sorted[known_sorted]
            interp_vals   = np.interp(a_sorted, known_ages, known_ledds)
            inv_order     = np.argsort(order)
            ledd_interp[idx] = interp_vals[inv_order]
        df["ledd_interp"]  = ledd_interp
        df["ledd_missing"] = ledd_missing
        df["ledd_slope"]   = ledd_slope_col
        return df


class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction (cross-fitted on training fold)."""

    def __init__(self, n_splits: int = 5, use_ledd_interp: bool = False):
        self.n_splits = n_splits
        self.use_ledd_interp = use_ledd_interp

    @property
    def _bias_off(self):
        return _BIAS_FEATURES_OFF_I if self.use_ledd_interp else \
            ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]

    @property
    def _bias_on(self):
        return _BIAS_FEATURES_ON_I if self.use_ledd_interp else \
            ["time_since_intake_on", "cohort", "ledd", "time_since_diagnosis"]

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
        cat_idx    = [bias_features.index("cohort")]
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
        cat_idx    = [bias_features.index("cohort")]
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
        df["off_corr"] = off_corr
        df["on_corr"]  = on_corr
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
        pids  = np.asarray(patient_ids); times = df_raw["time_since_diagnosis"].values.astype(float)
        y_arr = np.asarray(y, dtype=float)
        params = self._patient_params(pids, times, z_off, z_on, y_arr)
        mu, S = self._empirical_prior(params)
        s2_off, s2_on = self._noise_variances(pids, times, z_off, z_on, params)
        self.mu_ = mu; self.S_ = S
        self.s2_off_ = max(s2_off, 1e-6); self.s2_on_ = max(s2_on, 1e-6)
        return self

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        pids  = np.asarray(patient_ids if patient_ids is not None else df_raw["patient_id"].values)
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


def _upper_stats_for_scores(
    raw_vals: np.ndarray,
    times: np.ndarray,
    beta: float,
    t_query: np.ndarray,
    prefix: str,
) -> dict:
    valid = ~np.isnan(raw_vals)
    n_val = valid.sum()
    n_rows = len(t_query)
    top1    = np.full(n_rows, np.nan)
    mean2   = np.full(n_rows, np.nan)
    mean_uh = np.full(n_rows, np.nan)
    mean_nlo= np.full(n_rows, np.nan)
    mean_all= np.full(n_rows, np.nan)
    count   = np.full(n_rows, np.nan)
    spread  = np.full(n_rows, np.nan)
    proj2   = np.full(n_rows, np.nan)
    proj_uh = np.full(n_rows, np.nan)
    if n_val == 0:
        return {
            f"{prefix}_top1": top1, f"{prefix}_mean2": mean2,
            f"{prefix}_mean_uh": mean_uh, f"{prefix}_mean_nlo": mean_nlo,
            f"{prefix}_mean": mean_all, f"{prefix}_count": count,
            f"{prefix}_spread": spread, f"{prefix}_proj2": proj2,
            f"{prefix}_proj_uh": proj_uh,
        }
    v_vals  = raw_vals[valid]
    v_times = times[valid]
    t_mean  = v_times.mean()
    a = v_vals - beta * (v_times - t_mean)
    order = np.argsort(-a)
    a_sorted = a[order]
    n = len(a_sorted)
    n_uh = math.ceil(n / 2)
    _top1_val     = a_sorted[0]
    _mean2_val    = a_sorted[:min(2, n)].mean()
    _mean_uh_val  = a_sorted[:n_uh].mean()
    _mean_nlo_val = a_sorted[:-1].mean() if n > 1 else a_sorted[0]
    _mean_val     = a_sorted.mean()
    _spread_val   = _top1_val - _mean_val
    top1[:]     = _top1_val
    mean2[:]    = _mean2_val
    mean_uh[:]  = _mean_uh_val
    mean_nlo[:] = _mean_nlo_val
    mean_all[:] = _mean_val
    count[:]    = float(n)
    spread[:]   = _spread_val
    proj2[:]   = _mean2_val   + beta * (t_query - t_mean)
    proj_uh[:] = _mean_uh_val + beta * (t_query - t_mean)
    return {
        f"{prefix}_top1": top1, f"{prefix}_mean2": mean2,
        f"{prefix}_mean_uh": mean_uh, f"{prefix}_mean_nlo": mean_nlo,
        f"{prefix}_mean": mean_all, f"{prefix}_count": count,
        f"{prefix}_spread": spread, f"{prefix}_proj2": proj2,
        f"{prefix}_proj_uh": proj_uh,
    }


class UpperStats(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y):
        pids  = X["patient_id"].values
        times = (X["age"] - X["age_at_diagnosis"]).values.astype(float)
        y_arr = np.asarray(y, dtype=float)
        num, den = 0.0, 0.0
        for pid in np.unique(pids):
            mask = pids == pid
            t_p  = times[mask]
            y_p  = y_arr[mask]
            valid = ~(np.isnan(t_p) | np.isnan(y_p))
            if valid.sum() < 2:
                continue
            tc = t_p[valid] - t_p[valid].mean()
            yc = y_p[valid] - y_p[valid].mean()
            num += float((tc * yc).sum())
            den += float((tc ** 2).sum())
        self.beta_ = num / den if den > 0 else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        pids  = df["patient_id"].values
        times = (df["age"] - df["age_at_diagnosis"]).values.astype(float)
        beta  = self.beta_

        off_corr_col = "off_corr" if "off_corr" in df.columns else "off"
        on_corr_col  = "on_corr"  if "on_corr"  in df.columns else "on"

        off_vals      = df["off"].values.astype(float)
        on_vals       = df["on"].values.astype(float)
        off_corr_vals = df[off_corr_col].values.astype(float)
        on_corr_vals  = df[on_corr_col].values.astype(float)

        n = len(df)
        prefixes   = ["off_up", "on_up", "off_corr_up", "on_corr_up"]
        feat_names = ["top1","mean2","mean_uh","mean_nlo","mean","count","spread","proj2","proj_uh"]
        all_arrs = {f"{p}_{f}": np.full(n, np.nan) for p in prefixes for f in feat_names}

        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            t_p  = times[idx]
            for prefix, src_vals in [
                ("off_up",      off_vals[idx]),
                ("on_up",       on_vals[idx]),
                ("off_corr_up", off_corr_vals[idx]),
                ("on_corr_up",  on_corr_vals[idx]),
            ]:
                res = _upper_stats_for_scores(src_vals, t_p, beta, t_p, prefix)
                for col_name, arr in res.items():
                    all_arrs[col_name][idx] = arr

        for col_name, arr in all_arrs.items():
            df[col_name] = arr
        return df


# =============================================================================
# ImpliedOffTransformer (kept from 14 for feature compatibility)
# =============================================================================

_POP_GAP_FEATURES = [
    "time_since_intake_on", "ledd_interp", "time_since_diagnosis", "age_at_diagnosis",
]


class ImpliedOffTransformer(BaseEstimator, TransformerMixin):
    """Add implied_off, gap_weight, n_paired features."""

    def _compute_population_gaps(self, df: pd.DataFrame) -> np.ndarray:
        X_pop = df[_POP_GAP_FEATURES].copy()
        return self._pop_model_.predict(X_pop.values)

    def fit(self, df: pd.DataFrame, y=None):
        off_corr = df["off_corr"].values.astype(float)
        on_corr  = df["on_corr"].values.astype(float)
        pids     = df["patient_id"].values

        pt_gaps   = {}
        pt_n      = {}
        for pid in np.unique(pids):
            mask = pids == pid
            oc = off_corr[mask]; nc = on_corr[mask]
            paired = ~(np.isnan(oc) | np.isnan(nc))
            n_paired = int(paired.sum())
            pt_n[pid] = n_paired
            if n_paired > 0:
                pt_gaps[pid] = float(np.median(oc[paired] - nc[paired]))
            else:
                pt_gaps[pid] = np.nan

        paired_mask = ~(np.isnan(off_corr) | np.isnan(on_corr))
        gap_vals = (off_corr - on_corr)[paired_mask]
        feat_df  = df[_POP_GAP_FEATURES][paired_mask]
        self._pop_model_ = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=0,
        )
        if paired_mask.sum() >= 20:
            self._pop_model_.fit(feat_df.values, gap_vals)
        else:
            from sklearn.dummy import DummyRegressor
            self._pop_model_ = DummyRegressor(strategy="mean")
            self._pop_model_.fit(feat_df.values, gap_vals)

        pids_with_gap = [p for p, g in pt_gaps.items() if not np.isnan(g)]
        if len(pids_with_gap) < 5:
            self._k_ = 1.0
            return self

        pop_preds_all = self._compute_population_gaps(df)
        row_pop = dict(zip(range(len(df)), pop_preds_all))

        pt_pop_gap = {}
        for pid in pids_with_gap:
            mask = pids == pid
            idx  = np.where(mask)[0]
            oc = off_corr[idx]; nc = on_corr[idx]
            paired = ~(np.isnan(oc) | np.isnan(nc))
            if paired.sum() == 0:
                continue
            pp = np.array([row_pop[i] for i in idx[paired]])
            pt_pop_gap[pid] = float(pp.mean())

        k_candidates = np.concatenate([np.arange(0.5, 5, 0.5), np.arange(5, 50, 5)])
        best_k = 1.0
        best_mse = np.inf
        for k in k_candidates:
            mse = 0.0
            n_eval = 0
            for pid in pids_with_gap:
                n_p = pt_n[pid]
                if n_p < 2:
                    continue
                mask = pids == pid
                idx  = np.where(mask)[0]
                oc = off_corr[idx]; nc = on_corr[idx]
                paired_idx_local = np.where(~(np.isnan(oc) | np.isnan(nc)))[0]
                for leave_i in paired_idx_local:
                    rest = [j for j in paired_idx_local if j != leave_i]
                    if len(rest) == 0:
                        continue
                    g_rest = float(np.median(oc[rest] - nc[rest]))
                    pop_val = pt_pop_gap.get(pid, g_rest)
                    w = len(rest) / (len(rest) + k)
                    shrunk = w * g_rest + (1 - w) * pop_val
                    implied = nc[leave_i] + shrunk
                    err = oc[leave_i] - implied
                    mse += err ** 2
                    n_eval += 1
            if n_eval > 0 and mse / n_eval < best_mse:
                best_mse = mse / n_eval
                best_k = k

        self._k_ = float(best_k)
        self._pt_gaps_ = pt_gaps
        self._pt_n_ = pt_n
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        off_corr = df["off_corr"].values.astype(float)
        on_corr  = df["on_corr"].values.astype(float)
        pids     = df["patient_id"].values

        pop_gaps = self._compute_population_gaps(df)
        k = self._k_

        implied_off = np.full(len(df), np.nan)
        gap_weight  = np.zeros(len(df), dtype=float)
        n_paired_arr = np.zeros(len(df), dtype=float)

        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            oc = off_corr[idx]; nc = on_corr[idx]
            paired = ~(np.isnan(oc) | np.isnan(nc))
            n_p = int(paired.sum())
            n_paired_arr[idx] = n_p

            if n_p > 0:
                g_pt = float(np.median(oc[paired] - nc[paired]))
            elif pid in self._pt_gaps_ and not np.isnan(self._pt_gaps_[pid]):
                g_pt = self._pt_gaps_[pid]
                n_p  = self._pt_n_[pid]
            else:
                g_pt = np.nan
                n_p  = 0

            pop_val = float(np.mean(pop_gaps[idx]))
            if np.isnan(g_pt):
                w = 0.0
                shrunk_gap = pop_val
            else:
                w = n_p / (n_p + k)
                shrunk_gap = w * g_pt + (1 - w) * pop_val

            gap_weight[idx] = w
            has_on = ~np.isnan(nc)
            implied_off[idx[has_on]] = nc[has_on] + shrunk_gap

        df["implied_off"] = implied_off
        df["gap_weight"]  = gap_weight
        df["n_paired"]    = n_paired_arr
        return df


# =============================================================================
# DiagnosisImputer (verbatim from 19)
# =============================================================================

class DiagnosisImputer(BaseEstimator, TransformerMixin):
    """Impute missing age_at_diagnosis per cohort before time_since_diagnosis."""

    def fit(self, X: pd.DataFrame, y=None):
        pids    = X["patient_id"].values
        ages    = X["age"].values.astype(float)
        diag    = X["age_at_diagnosis"].values.astype(float)
        cohorts = X["cohort"].values

        pt_min_age = {}
        pt_diag    = {}
        pt_cohort  = {}
        for pid in np.unique(pids):
            mask = pids == pid
            d = diag[mask]
            if np.isnan(d).all():
                continue
            pt_min_age[pid] = float(np.nanmin(ages[mask]))
            pt_diag[pid]    = float(np.nanmean(d[~np.isnan(d)]))
            pt_cohort[pid]  = cohorts[mask][0]

        gaps_by_cohort: dict = {}
        global_gaps = []
        for pid, d in pt_diag.items():
            gap = pt_min_age[pid] - d
            cohort = pt_cohort[pid]
            gaps_by_cohort.setdefault(cohort, []).append(gap)
            global_gaps.append(gap)

        self._cohort_median_gap_: dict = {
            c: float(np.median(v)) for c, v in gaps_by_cohort.items()
        }
        self._global_median_gap_ = float(np.median(global_gaps)) if global_gaps else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        pids    = df["patient_id"].values
        ages    = df["age"].values.astype(float)
        diag    = df["age_at_diagnosis"].values.astype(float)
        cohorts = df["cohort"].values

        diag_imputed = np.zeros(len(df), dtype=float)

        impute_pids = set()
        for pid in np.unique(pids):
            mask = pids == pid
            if np.isnan(diag[mask]).all():
                impute_pids.add(pid)

        if impute_pids:
            new_diag = diag.copy()
            for pid in impute_pids:
                mask = pids == pid
                idx  = np.where(mask)[0]
                min_age = float(np.nanmin(ages[idx]))
                cohort  = cohorts[idx[0]]
                gap = self._cohort_median_gap_.get(cohort, self._global_median_gap_)
                imputed_val = min_age - gap
                new_diag[idx] = imputed_val
                diag_imputed[idx] = 1.0
            df["age_at_diagnosis"] = new_diag

        df["diag_imputed"] = diag_imputed
        return df


# =============================================================================
# quadratic_projection helper (verbatim from 19)
# =============================================================================

def quadratic_projection(pids: np.ndarray, ages: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Replace predictions with per-patient quadratic fit on centered age."""
    output = preds.copy()
    for pid in np.unique(pids):
        mask = pids == pid
        idx  = np.where(mask)[0]
        n    = len(idx)
        if n < 3:
            continue
        a = ages[idx].astype(float)
        p = preds[idx]
        valid = ~(np.isnan(a) | np.isnan(p))
        if valid.sum() < 3:
            continue
        tc = a - a[valid].mean()
        if n == 3:
            A = np.column_stack([np.ones(valid.sum()), tc[valid]])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, p[valid], rcond=None)
                fitted = np.ones(n) * coeffs[0] + tc * coeffs[1]
            except np.linalg.LinAlgError:
                continue
        else:
            A = np.column_stack([np.ones(valid.sum()), tc[valid], tc[valid] ** 2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, p[valid], rcond=None)
                fitted = coeffs[0] + tc * coeffs[1] + tc ** 2 * coeffs[2]
            except np.linalg.LinAlgError:
                continue
        for local_i, global_i in enumerate(idx):
            if not np.isnan(a[local_i]):
                output[global_i] = fitted[local_i]
    return output


# =============================================================================
# Evidence model parameters (shared HGB params)
# =============================================================================

_EV_HGB_PARAMS = dict(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=20, random_state=0,
)

_SIMPLE_FEATS = [
    "cohort", "gene", "sexM", "age_at_diagnosis", "age", "time_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off", "on", "off",
    "off_pt_mean", "off_pt_max", "off_pt_count",
    "on_pt_mean", "on_pt_max", "on_pt_count",
    "ledd_pt_mean", "ledd_pt_max", "ledd_pt_count",
    "age_tc", "pt_n_visits",
]

_VISIT_FEATS = [
    "cohort", "gene", "sexM", "age_at_diagnosis", "age", "time_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off", "on", "off",
]

_OFF_EV_FEATS = [
    "off", "time_since_intake_off", "ledd", "age", "time_since_diagnosis",
    "cohort", "sexM", "gene", "age_at_diagnosis",
]

_ON_EV_FEATS = [
    "on", "time_since_intake_on", "ledd", "age", "time_since_diagnosis",
    "cohort", "sexM", "gene", "age_at_diagnosis",
]

_EV_CAT_COLS = ["cohort", "gene"]

_SEV_EV_VISIT_FEATS = [
    "s_ratio", "off_present", "f_drug", "nE_i",
    "cohort", "sexM", "gene", "age", "age_at_diagnosis", "time_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off", "on", "off",
]
_SEV_EV_PT_FEATS = [
    "s_pt_max", "s_pt_mean2", "s_pt_mean3", "s_pt_mean", "s_pt_median",
    "s_pt_p75", "s_pt_min", "s_pt_count",
    "s_age_slope", "s_age_resid",
    "s_age_resid_pt_max", "s_age_resid_pt_mean2",
    "age_tc_sev", "pt_n_visits_sev", "rel_pos_sev",
]


def _encode_cats(df: pd.DataFrame, cat_cols: list) -> np.ndarray:
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(float)
    return out.values.astype(float)


def _add_simple_aggs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["off", "on", "ledd"]:
        mx = df.groupby("patient_id")[col].transform("max")
        ct = df.groupby("patient_id")[col].transform(lambda s: s.notna().sum())
        df[f"{col}_pt_max"]   = mx
        df[f"{col}_pt_count"] = ct
    pids = df["patient_id"].values
    ages = df["age"].values.astype(float)
    age_tc = np.full(len(df), np.nan)
    for pid in np.unique(pids):
        mask = pids == pid
        idx = np.where(mask)[0]
        a = ages[idx]
        valid = ~np.isnan(a)
        if valid.sum() > 0:
            age_tc[idx] = a - a[valid].mean()
    df["age_tc"] = age_tc
    return df


# =============================================================================
# N: OFF-scale severity s computation — extended with e_mode parameter
# =============================================================================

# Shrinkage constants for e_mode="shrunk_mean"
_E_SD_BETWEEN = 0.094   # between-patient SD of E_i
_E_SD_WITHIN  = 0.079   # within-patient SD of per-visit E_t around patient mean


def _compute_f_drug(time_since_intake_on: np.ndarray) -> np.ndarray:
    """f = min(t / 1.5, 1); 1 when missing."""
    f = np.where(
        np.isnan(time_since_intake_on),
        1.0,
        np.minimum(time_since_intake_on / 1.5, 1.0),
    )
    return f


def compute_s_ratio(
    df: pd.DataFrame,
    E_pop: float = None,
    e_mode: str = "median",
) -> tuple:
    """Compute OFF-scale severity s and associated features.

    Parameters
    ----------
    e_mode : "median" (Experiment 19 behaviour) or "shrunk_mean" (James-Stein
             shrinkage of per-patient E_i toward the population mean).
    """
    n = len(df)
    off_col = df["off"].values.astype(float) if "off" in df.columns else np.full(n, np.nan)
    on_col  = df["on"].values.astype(float)  if "on"  in df.columns else np.full(n, np.nan)
    tsi_on  = df["time_since_intake_on"].values.astype(float) if "time_since_intake_on" in df.columns else np.full(n, np.nan)
    pids    = df["patient_id"].values

    f_arr = _compute_f_drug(tsi_on)

    has_off = ~np.isnan(off_col)
    has_on  = ~np.isnan(on_col)
    paired  = has_off & has_on & (off_col > 0) & (f_arr >= 0.5)

    E_vals_paired = np.where(paired, (1.0 - on_col / np.where(off_col > 0, off_col, 1.0)) / np.where(f_arr > 0, f_arr, 1.0), np.nan)

    E_i_arr  = np.full(n, np.nan)
    nE_i_arr = np.zeros(n, dtype=float)

    # Per-patient raw statistics (same for both modes)
    pt_nE_i  = {}
    pt_mean  = {}
    pt_median = {}
    for pid in np.unique(pids):
        mask  = pids == pid
        e_p   = E_vals_paired[mask]
        valid = ~np.isnan(e_p)
        nv    = int(valid.sum())
        pt_nE_i[pid] = nv
        if nv > 0:
            pt_mean[pid]   = float(np.mean(e_p[valid]))
            pt_median[pid] = float(np.clip(np.median(e_p[valid]), 0.0, 0.85))
        else:
            pt_mean[pid]   = np.nan
            pt_median[pid] = np.nan

    all_e_paired = E_vals_paired[paired]
    if E_pop is None:
        if len(all_e_paired) > 0:
            E_pop_out = float(np.clip(np.median(all_e_paired), 0.0, 0.85))
        else:
            E_pop_out = 0.3
    else:
        E_pop_out = E_pop

    # Compute per-patient E_i depending on mode
    if e_mode == "shrunk_mean":
        sb2 = _E_SD_BETWEEN ** 2   # 0.094^2
        sw2 = _E_SD_WITHIN  ** 2   # 0.079^2
        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            nv   = pt_nE_i[pid]
            if nv == 0:
                E_used = E_pop_out
            else:
                lam = sb2 / (sb2 + sw2 / nv)
                E_used = float(np.clip(E_pop_out + lam * (pt_mean[pid] - E_pop_out), 0.0, 0.85))
            E_i_arr[idx]  = E_used
            nE_i_arr[idx] = float(nv)
    else:
        # "median" mode: verbatim Experiment 19
        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            ei   = pt_median[pid]
            if np.isnan(ei):
                E_used = E_pop_out
            else:
                E_used = ei
            E_i_arr[idx]  = E_used
            nE_i_arr[idx] = float(pt_nE_i[pid])

    s_arr = np.full(n, np.nan)
    for i in range(n):
        if has_off[i]:
            s_arr[i] = off_col[i]
        elif has_on[i]:
            denom = max(1.0 - E_i_arr[i] * f_arr[i], 0.15)
            s_arr[i] = on_col[i] / denom

    return s_arr, f_arr, has_off, E_i_arr, nE_i_arr, E_pop_out


# =============================================================================
# sev_evidence feature builder (verbatim from 19)
# =============================================================================

def _build_sev_ev_features(df: pd.DataFrame) -> np.ndarray:
    all_feats = [c for c in _SEV_EV_VISIT_FEATS + _SEV_EV_PT_FEATS if c in df.columns]
    return _encode_cats(df[all_feats], _EV_CAT_COLS)


def _add_sev_ev_pt_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pids = df["patient_id"].values
    ages = df["age"].values.astype(float)
    s_arr = df["s_ratio"].values.astype(float)

    n = len(df)
    s_pt_max    = np.full(n, np.nan)
    s_pt_mean2  = np.full(n, np.nan)
    s_pt_mean3  = np.full(n, np.nan)
    s_pt_mean   = np.full(n, np.nan)
    s_pt_median = np.full(n, np.nan)
    s_pt_p75    = np.full(n, np.nan)
    s_pt_min    = np.full(n, np.nan)
    s_pt_count  = np.full(n, np.nan)
    s_age_slope = np.full(n, np.nan)
    s_age_resid = np.full(n, np.nan)
    age_tc_sev  = np.full(n, np.nan)
    pt_n_visits_sev = np.full(n, np.nan)
    rel_pos_sev = np.full(n, np.nan)

    for pid in np.unique(pids):
        mask = pids == pid
        idx  = np.where(mask)[0]
        s_p  = s_arr[idx]
        a_p  = ages[idx]
        valid_s = ~np.isnan(s_p)
        nv    = int(valid_s.sum())

        pt_n_visits_sev[idx] = float(len(idx))
        valid_a = ~np.isnan(a_p)
        if valid_a.sum() > 0:
            am = a_p[valid_a].mean()
            age_tc_sev[idx] = a_p - am
        if len(idx) > 1:
            order = np.argsort(a_p)
            rp = np.full(len(idx), np.nan)
            for rank_i, si in enumerate(order):
                rp[si] = rank_i / (len(idx) - 1)
            rel_pos_sev[idx] = rp
        else:
            rel_pos_sev[idx] = 0.0

        if nv == 0:
            continue

        sv = np.sort(s_p[valid_s])[::-1]
        s_pt_max[idx]    = sv[0]
        s_pt_mean2[idx]  = sv[:min(2, nv)].mean()
        s_pt_mean3[idx]  = sv[:min(3, nv)].mean()
        s_pt_mean[idx]   = sv.mean()
        s_pt_median[idx] = float(np.median(sv))
        s_pt_p75[idx]    = float(np.percentile(sv, 75))
        s_pt_min[idx]    = sv[-1]
        s_pt_count[idx]  = float(nv)

        sl, ic = _linreg_slope_intercept(a_p[valid_s], s_p[valid_s])
        s_age_slope[idx] = sl if not np.isnan(sl) else 0.0
        if not np.isnan(sl) and nv >= 3:
            fitted = sl * a_p + ic
            s_age_resid[idx] = s_p - fitted
        else:
            s_age_resid[idx] = 0.0

    s_age_resid_pt_max   = np.full(n, np.nan)
    s_age_resid_pt_mean2 = np.full(n, np.nan)
    for pid in np.unique(pids):
        mask = pids == pid
        idx  = np.where(mask)[0]
        rv   = s_age_resid[idx]
        valid_r = ~np.isnan(rv)
        if valid_r.sum() > 0:
            rsorted = np.sort(rv[valid_r])[::-1]
            s_age_resid_pt_max[idx]   = rsorted[0]
            s_age_resid_pt_mean2[idx] = rsorted[:min(2, len(rsorted))].mean()

    df["s_pt_max"]    = s_pt_max
    df["s_pt_mean2"]  = s_pt_mean2
    df["s_pt_mean3"]  = s_pt_mean3
    df["s_pt_mean"]   = s_pt_mean
    df["s_pt_median"] = s_pt_median
    df["s_pt_p75"]    = s_pt_p75
    df["s_pt_min"]    = s_pt_min
    df["s_pt_count"]  = s_pt_count
    df["s_age_slope"] = s_age_slope
    df["s_age_resid"] = s_age_resid
    df["s_age_resid_pt_max"]   = s_age_resid_pt_max
    df["s_age_resid_pt_mean2"] = s_age_resid_pt_mean2
    df["age_tc_sev"]       = age_tc_sev
    df["pt_n_visits_sev"]  = pt_n_visits_sev
    df["rel_pos_sev"]      = rel_pos_sev
    return df


# =============================================================================
# Noise density fitting and lookup (verbatim from 19)
# =============================================================================

_TARGET_BIN_EDGES = np.array([0.0, 8.0, 15.0, 22.0, 30.0, 38.0, 46.0, 55.0, 65.0, 80.0, np.inf])
_N_BINS = 10

_LV_GRID = np.arange(-5.00, 1.51, 0.01)
_LV_GRID_N = len(_LV_GRID)
_LV_STEP = 0.01


def _make_gauss_kernel(sd: float = 0.03, half: int = 25) -> np.ndarray:
    j = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-0.5 * (j * _LV_STEP / sd) ** 2)
    return w / w.sum()

_GAUSS_KERNEL = _make_gauss_kernel()


def _target_bin(v: float) -> int:
    edges = _TARGET_BIN_EDGES
    idx = int(np.searchsorted(edges, v, side="right")) - 1
    return int(np.clip(idx, 0, _N_BINS - 1))


def _target_bin_array(v_arr: np.ndarray) -> np.ndarray:
    edges = _TARGET_BIN_EDGES
    idx = np.searchsorted(edges, v_arr, side="right") - 1
    return np.clip(idx, 0, _N_BINS - 1).astype(int)


def fit_logdens(
    s_arr: np.ndarray,
    target_arr: np.ndarray,
    cat_arr: np.ndarray,
) -> np.ndarray:
    """Fit log-density array of shape (3, 10, 651)."""
    n_cats = 3
    logdens = np.zeros((n_cats, _N_BINS, _LV_GRID_N), dtype=float)

    valid = (target_arr > 0.5) & (~np.isnan(s_arr)) & (~np.isnan(target_arr))
    s_v = s_arr[valid]
    t_v = target_arr[valid]
    c_v = cat_arr[valid].astype(int)

    lv_all = np.log(np.maximum(s_v, 0.5) / np.maximum(t_v, 0.5))
    bins_all = _target_bin_array(t_v)

    for c in range(n_cats):
        c_mask = c_v == c
        lv_c   = lv_all[c_mask]
        bins_c = bins_all[c_mask]

        for b in range(_N_BINS):
            b_mask = bins_c == b
            lv_cb = lv_c[b_mask]

            if len(lv_cb) < 150:
                b_expand = (bins_c >= max(0, b - 1)) & (bins_c <= min(_N_BINS - 1, b + 1))
                lv_cb = lv_c[b_expand]

            if len(lv_cb) < 150:
                lv_cb = lv_c

            if len(lv_cb) == 0:
                logdens[c, b, :] = np.log(1e-4)
                continue

            counts = np.zeros(_LV_GRID_N, dtype=float)
            lv_clipped = lv_cb[(lv_cb >= _LV_GRID[0] - _LV_STEP / 2) &
                                (lv_cb <  _LV_GRID[-1] + _LV_STEP / 2)]
            if len(lv_clipped) > 0:
                k_idx = np.round((lv_clipped - _LV_GRID[0]) / _LV_STEP).astype(int)
                k_idx = np.clip(k_idx, 0, _LV_GRID_N - 1)
                np.add.at(counts, k_idx, 1.0)

            smoothed = convolve1d(counts, _GAUSS_KERNEL, mode="constant", cval=0.0)

            total = smoothed.sum()
            if total > 0:
                density = smoothed / (total * _LV_STEP)
            else:
                density = np.full(_LV_GRID_N, 1e-4)

            density = np.maximum(density, 1e-4)
            logdens[c, b, :] = np.log(density)

    return logdens


def lookup_logdens(logdens: np.ndarray, c: int, b: int, x: float) -> float:
    x_clip = float(np.clip(x, _LV_GRID[0], _LV_GRID[-1]))
    k_frac = (x_clip - _LV_GRID[0]) / _LV_STEP
    k0 = int(k_frac)
    k1 = min(k0 + 1, _LV_GRID_N - 1)
    frac = k_frac - k0
    return float(logdens[c, b, k0] * (1.0 - frac) + logdens[c, b, k1] * frac)


def lookup_logdens_batch(logdens: np.ndarray, c_arr: np.ndarray, b_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x_arr, _LV_GRID[0], _LV_GRID[-1])
    k_frac = (x_clip - _LV_GRID[0]) / _LV_STEP
    k0 = np.floor(k_frac).astype(int)
    k1 = np.minimum(k0 + 1, _LV_GRID_N - 1)
    frac = k_frac - k0
    v0 = logdens[c_arr, b_arr, k0]
    v1 = logdens[c_arr, b_arr, k1]
    return v0 * (1.0 - frac) + v1 * frac


# =============================================================================
# Per-patient likelihood features (verbatim from 19)
# =============================================================================

_DELTA1 = np.arange(-1.00, 1.01, 0.01)
_N_D1   = len(_DELTA1)

_DELTA2 = np.arange(-0.80, 0.81, 0.02)
_GAMMA2 = np.arange(-0.08, 0.09, 0.01)
_N_D2   = len(_DELTA2)
_N_G2   = len(_GAMMA2)

_GAMMA2_PRIOR = 0.5 * (_GAMMA2 / 0.03) ** 2

_MAX_VISITS = 12
_BATCH_PATIENTS = 300


def _likelihood_features_for_patients(
    pids_all: np.ndarray,
    s_all: np.ndarray,
    q_all: np.ndarray,
    c_all: np.ndarray,
    tc_all: np.ndarray,
    logdens: np.ndarray,
) -> np.ndarray:
    unique_pids = np.unique(pids_all)
    n = len(pids_all)
    feats = np.zeros((n, 6), dtype=float)

    for batch_start in range(0, len(unique_pids), _BATCH_PATIENTS):
        batch_pids = unique_pids[batch_start:batch_start + _BATCH_PATIENTS]
        _bp = len(batch_pids)

        s_pad  = np.full((_bp, _MAX_VISITS), np.nan)
        q_pad  = np.full((_bp, _MAX_VISITS), np.nan)
        c_pad  = np.full((_bp, _MAX_VISITS), -1, dtype=int)
        tc_pad = np.full((_bp, _MAX_VISITS), np.nan)
        n_vis  = np.zeros(_bp, dtype=int)
        pid_indices = []

        for pi, pid in enumerate(batch_pids):
            mask = pids_all == pid
            idx_p = np.where(mask)[0]
            nv = min(len(idx_p), _MAX_VISITS)
            n_vis[pi] = nv
            s_pad[pi, :nv]  = s_all[idx_p[:nv]]
            q_pad[pi, :nv]  = q_all[idx_p[:nv]]
            c_pad[pi, :nv]  = c_all[idx_p[:nv]]
            tc_pad[pi, :nv] = tc_all[idx_p[:nv]]
            pid_indices.append(idx_p)

        log_q = np.log(np.maximum(q_pad, 0.5))
        log_s = np.log(np.maximum(s_pad, 0.5))
        lv_base = log_s - log_q

        exp_d1 = np.exp(_DELTA1)
        y1 = q_pad[:, :, np.newaxis] * exp_d1[np.newaxis, np.newaxis, :]
        b1 = _target_bin_array(y1.ravel()).reshape(_bp, _MAX_VISITS, _N_D1)

        lv1 = lv_base[:, :, np.newaxis] - _DELTA1[np.newaxis, np.newaxis, :]

        valid1_3d = np.broadcast_to(
            (~np.isnan(lv_base))[:, :, np.newaxis], (_bp, _MAX_VISITS, _N_D1)
        ).copy()

        c_flat1  = np.repeat(c_pad, _N_D1, axis=1).ravel()
        b1_flat  = b1.ravel()
        lv1_flat = lv1.ravel()
        vld_flat = valid1_3d.ravel() & (c_flat1 >= 0)

        ld1_flat = np.zeros(len(c_flat1), dtype=float)
        if vld_flat.any():
            ld1_flat[vld_flat] = lookup_logdens_batch(
                logdens,
                c_flat1[vld_flat],
                b1_flat[vld_flat],
                lv1_flat[vld_flat],
            )
        ld1_flat[~vld_flat] = 0.0

        ld1 = ld1_flat.reshape(_bp, _MAX_VISITS, _N_D1)
        ld1[~valid1_3d] = 0.0
        LL1 = ld1.sum(axis=1)

        LL1_max = LL1.max(axis=1, keepdims=True)
        w1 = np.exp(LL1 - LL1_max)
        w1_sum = w1.sum(axis=1, keepdims=True)
        w1_sum = np.where(w1_sum == 0, 1.0, w1_sum)
        w1 /= w1_sum

        pm = (w1 * exp_d1[np.newaxis, :]).sum(axis=1)
        dmle_idx = LL1.argmax(axis=1)
        dmle = _DELTA1[dmle_idx]

        wd  = (w1 * _DELTA1[np.newaxis, :]).sum(axis=1)
        wd2 = (w1 * _DELTA1[np.newaxis, :] ** 2).sum(axis=1)
        psd = np.sqrt(np.maximum(wd2 - wd ** 2, 0.0))

        exp_d2 = np.exp(_DELTA2)
        exp_gtc = np.exp(_GAMMA2[np.newaxis, np.newaxis, :] * tc_pad[:, :, np.newaxis])
        y2 = (q_pad[:, :, np.newaxis, np.newaxis]
              * exp_d2[np.newaxis, np.newaxis, :, np.newaxis]
              * exp_gtc[:, :, np.newaxis, :])

        b2 = _target_bin_array(y2.ravel()).reshape(_bp, _MAX_VISITS, _N_D2, _N_G2)

        lv2 = (lv_base[:, :, np.newaxis, np.newaxis]
               - _DELTA2[np.newaxis, np.newaxis, :, np.newaxis]
               - _GAMMA2[np.newaxis, np.newaxis, np.newaxis, :] * tc_pad[:, :, np.newaxis, np.newaxis])

        valid2_4d = np.broadcast_to(
            (~np.isnan(lv_base))[:, :, np.newaxis, np.newaxis],
            (_bp, _MAX_VISITS, _N_D2, _N_G2)
        ).copy()

        total2 = _bp * _MAX_VISITS * _N_D2 * _N_G2
        c_flat2  = np.repeat(c_pad, _N_D2 * _N_G2, axis=1).ravel()
        b2_flat  = b2.ravel()
        lv2_flat = lv2.ravel()
        vld2_flat = valid2_4d.ravel() & (c_flat2 >= 0)

        ld2_flat = np.zeros(total2, dtype=float)
        if vld2_flat.any():
            ld2_flat[vld2_flat] = lookup_logdens_batch(
                logdens,
                c_flat2[vld2_flat],
                b2_flat[vld2_flat],
                lv2_flat[vld2_flat],
            )
        ld2_flat[~vld2_flat] = 0.0

        ld2 = ld2_flat.reshape(_bp, _MAX_VISITS, _N_D2, _N_G2)
        ld2[~valid2_4d] = 0.0
        LL2 = ld2.sum(axis=1)

        LL2 = LL2 - _GAMMA2_PRIOR[np.newaxis, np.newaxis, :]

        LL2_flat = LL2.reshape(_bp, _N_D2 * _N_G2)
        LL2_max  = LL2_flat.max(axis=1, keepdims=True)
        w2_flat  = np.exp(LL2_flat - LL2_max)
        w2_sum   = w2_flat.sum(axis=1, keepdims=True)
        w2_sum   = np.where(w2_sum == 0, 1.0, w2_sum)
        w2_flat /= w2_sum

        w2 = w2_flat.reshape(_bp, _N_D2, _N_G2)

        dpm1 = (w2 * _DELTA2[np.newaxis, :, np.newaxis]).sum(axis=(1, 2))
        gpm1 = (w2 * _GAMMA2[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))

        wd_g2  = (w2 * _GAMMA2[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
        wd2_g2 = (w2 * _GAMMA2[np.newaxis, np.newaxis, :] ** 2).sum(axis=(1, 2))
        gsd1   = np.sqrt(np.maximum(wd2_g2 - wd_g2 ** 2, 0.0))

        for pi, pid in enumerate(batch_pids):
            idx_p = pid_indices[pi]
            q_p   = q_all[idx_p]
            tc_p  = tc_all[idx_p]

            f1 = q_p * (pm[pi] - 1.0)
            f2 = q_p * (np.exp(dmle[pi]) - 1.0)
            f3 = np.full(len(idx_p), psd[pi])
            f4 = q_p * (np.exp(dpm1[pi] + gpm1[pi] * tc_p) - 1.0)
            f5 = np.full(len(idx_p), gpm1[pi])
            f6 = np.full(len(idx_p), gsd1[pi])

            feats[idx_p, 0] = f1
            feats[idx_p, 1] = f2
            feats[idx_p, 2] = f3
            feats[idx_p, 3] = f4
            feats[idx_p, 4] = f5
            feats[idx_p, 5] = f6

    return feats


def _compute_visit_category(off_present: np.ndarray, nE_i_arr: np.ndarray) -> np.ndarray:
    c = np.where(
        off_present,
        0,
        np.where(nE_i_arr > 0, 1, 2),
    ).astype(int)
    return c


# =============================================================================
# N: Helper _add_s_cols with e_mode parameter
# =============================================================================

def _add_s_cols(X: pd.DataFrame, E_pop: float, e_mode: str = "median") -> tuple:
    """Compute s and attach columns; e_mode passed through to compute_s_ratio."""
    s_arr, f_arr, off_present, E_i_arr, nE_i_arr, _ = compute_s_ratio(X, E_pop=E_pop, e_mode=e_mode)
    df = X.copy()
    df["s_ratio"]     = s_arr
    df["f_drug"]      = f_arr
    df["off_present"] = off_present.astype(float)
    df["nE_i"]        = nE_i_arr
    return df, s_arr, f_arr, off_present, nE_i_arr


# =============================================================================
# N: TwoStageRegressor with e_mode parameter
# =============================================================================

class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Two-stage regressor with likelihood features.

    Parameters
    ----------
    e_mode : "median" (Experiment 19, default) or "shrunk_mean" (shrinkage
             of per-patient E_i toward population mean).
    """

    def __init__(self, base_params: dict, n_splits_inner: int = 5, e_mode: str = "median"):
        self.base_params = base_params
        self.n_splits_inner = n_splits_inner
        self.e_mode = e_mode

    def _make_base(self):
        return HistGradientBoostingRegressor(**self.base_params)

    def _make_ev_model(self):
        return HistGradientBoostingRegressor(**_EV_HGB_PARAMS)

    def _make_sev_model(self):
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=30, random_state=0,
        )

    @staticmethod
    def _stage2_features(
        X: pd.DataFrame,
        r: np.ndarray,
        q: np.ndarray,
        ev_visit: np.ndarray,
        ev_off: np.ndarray,
        ev_on: np.ndarray,
        s_arr: np.ndarray,
        f_arr: np.ndarray,
        off_present: np.ndarray,
        nE_i_arr: np.ndarray,
        logdens: np.ndarray,
    ) -> np.ndarray:
        """Build 59-column stage-2 feature matrix."""
        pids    = X["patient_id"].values
        ages    = X["age"].values.astype(float)
        off_col = X["off"].values.astype(float) if "off" in X.columns else np.full(len(X), np.nan)
        on_col  = X["on"].values.astype(float)  if "on"  in X.columns else np.full(len(X), np.nan)

        n = len(X)
        feat_q          = q.copy()
        feat_q_dev      = np.full(n, np.nan)
        feat_r_minus_q  = r - q

        feat_off_minus_q = off_col - q
        feat_on_minus_q  = np.where(np.isnan(on_col), np.nan, on_col - q)

        feat_max_off_q    = np.full(n, np.nan)
        feat_mean_off_q   = np.full(n, np.nan)
        feat_mean2_off_q  = np.full(n, np.nan)
        feat_mean_on_q    = np.full(n, np.nan)
        feat_max_on_q     = np.full(n, np.nan)
        feat_std_r_q      = np.full(n, np.nan)
        feat_n_off        = np.full(n, np.nan)
        feat_n_on         = np.full(n, np.nan)

        feat_tc           = np.full(n, np.nan)
        feat_rel_pos      = np.full(n, np.nan)
        feat_n_visits     = np.full(n, np.nan)
        feat_off_dev      = X["off_dev"].values.astype(float) if "off_dev" in X.columns else np.full(n, np.nan)
        feat_on_dev       = X["on_dev"].values.astype(float)  if "on_dev"  in X.columns else np.full(n, np.nan)
        feat_ledd_dev     = np.full(n, np.nan)
        feat_tsi_on       = X["time_since_intake_on"].values.astype(float)  if "time_since_intake_on"  in X.columns else np.full(n, np.nan)
        feat_tsi_off      = X["time_since_intake_off"].values.astype(float) if "time_since_intake_off" in X.columns else np.full(n, np.nan)

        ledd_col = X["ledd_interp"].values.astype(float) if "ledd_interp" in X.columns else np.full(n, np.nan)

        ev_sources = [("visit", ev_visit), ("off_ev", ev_off), ("on_ev", ev_on)]
        ev_delta      = {name: e - q for name, e in ev_sources}
        ev_max_eq     = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean_eq    = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean2_eq   = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_min_eq     = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}

        s_minus_q          = s_arr - q
        s_q_max            = np.full(n, np.nan)
        s_q_mean           = np.full(n, np.nan)
        s_q_mean2          = np.full(n, np.nan)
        s_q_mean3          = np.full(n, np.nan)
        s_q_min            = np.full(n, np.nan)
        s_q_median         = np.full(n, np.nan)
        s_q_p75            = np.full(n, np.nan)
        s_count_feat       = np.full(n, np.nan)

        tc_all = np.full(n, np.nan)

        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]

            q_p     = q[idx]
            r_p     = r[idx]
            age_p   = ages[idx]
            off_p   = off_col[idx]
            on_p    = on_col[idx]
            ledd_p  = ledd_col[idx]

            valid_age = ~np.isnan(age_p)
            if valid_age.sum() > 0:
                age_mean = age_p[valid_age].mean()
                tc_p = age_p - age_mean
            else:
                tc_p = np.full(len(idx), np.nan)

            feat_tc[idx] = tc_p
            tc_all[idx]  = tc_p
            feat_q_dev[idx] = q_p - np.nanmean(q_p)

            age_order = np.argsort(age_p)
            rel_pos = np.full(len(idx), np.nan)
            nv = len(idx)
            if nv > 1:
                for rank_i, sorted_i in enumerate(age_order):
                    rel_pos[sorted_i] = rank_i / (nv - 1)
            elif nv == 1:
                rel_pos[0] = 0.0
            feat_rel_pos[idx] = rel_pos
            feat_n_visits[idx] = float(nv)

            lm = np.nanmean(ledd_p) if not np.all(np.isnan(ledd_p)) else np.nan
            feat_ledd_dev[idx] = ledd_p - lm

            off_q_p = off_p - q_p
            valid_off = ~np.isnan(off_q_p)
            n_off = int(valid_off.sum())
            feat_n_off[idx] = float(n_off)
            if n_off > 0:
                off_q_valid = off_q_p[valid_off]
                feat_mean_off_q[idx]  = float(np.mean(off_q_valid))
                feat_max_off_q[idx]   = float(np.max(off_q_valid))
                top2 = np.sort(off_q_valid)[::-1][:2]
                feat_mean2_off_q[idx] = float(top2.mean())

            on_q_p = np.where(np.isnan(on_p), np.nan, on_p - q_p)
            valid_on = ~np.isnan(on_q_p)
            n_on = int(valid_on.sum())
            feat_n_on[idx] = float(n_on)
            if n_on > 0:
                on_q_valid = on_q_p[valid_on]
                feat_mean_on_q[idx] = float(np.mean(on_q_valid))
                feat_max_on_q[idx]  = float(np.max(on_q_valid))

            rq_p = r_p - q_p
            if len(rq_p) > 1:
                feat_std_r_q[idx] = float(np.nanstd(rq_p, ddof=0))
            else:
                feat_std_r_q[idx] = 0.0

            for name, delta_arr in ev_delta.items():
                d_p = delta_arr[idx]
                valid_d = ~np.isnan(d_p)
                nd = int(valid_d.sum())
                if nd > 0:
                    dv = d_p[valid_d]
                    ev_max_eq[name][idx]   = float(np.max(dv))
                    ev_mean_eq[name][idx]  = float(np.mean(dv))
                    top2d = np.sort(dv)[::-1][:2]
                    ev_mean2_eq[name][idx] = float(top2d.mean())
                    ev_min_eq[name][idx]   = float(np.min(dv))

            sq_p = s_minus_q[idx]
            valid_sq = ~np.isnan(sq_p)
            ns = int(valid_sq.sum())
            s_count_feat[idx] = float(ns)
            if ns > 0:
                sqv = np.sort(sq_p[valid_sq])[::-1]
                s_q_max[idx]    = sqv[0]
                s_q_mean[idx]   = sqv.mean()
                s_q_mean2[idx]  = sqv[:min(2, ns)].mean()
                s_q_mean3[idx]  = sqv[:min(3, ns)].mean()
                s_q_min[idx]    = sqv[-1]
                s_q_median[idx] = float(np.median(sqv))
                s_q_p75[idx]    = float(np.percentile(sqv, 75))

        base_feats = np.column_stack([
            feat_q, feat_q_dev, feat_r_minus_q,
            feat_off_minus_q, feat_on_minus_q,
            feat_max_off_q, feat_mean_off_q, feat_mean2_off_q,
            feat_mean_on_q, feat_max_on_q, feat_std_r_q,
            feat_n_off, feat_n_on, feat_tc, feat_rel_pos, feat_n_visits,
            feat_off_dev, feat_on_dev, feat_ledd_dev,
            feat_tsi_on, feat_tsi_off,
        ])

        ev_feats_list = []
        for name in ["visit", "off_ev", "on_ev"]:
            ev_feats_list.append(ev_delta[name])
            ev_feats_list.append(ev_max_eq[name])
            ev_feats_list.append(ev_mean_eq[name])
            ev_feats_list.append(ev_mean2_eq[name])
            ev_feats_list.append(ev_min_eq[name])
        ev_feats = np.column_stack(ev_feats_list)

        s_feats = np.column_stack([
            s_minus_q,
            s_q_max,
            s_q_mean,
            s_q_mean2,
            s_q_mean3,
            s_q_min,
            s_q_median,
            s_q_p75,
            s_count_feat,
            nE_i_arr,
            f_arr,
            off_present.astype(float),
        ])

        cat_arr = _compute_visit_category(off_present, nE_i_arr)
        lik_feats = _likelihood_features_for_patients(
            pids, s_arr, q, cat_arr, tc_all, logdens,
        )

        return np.column_stack([base_feats, ev_feats, s_feats, lik_feats])

    def _build_simple_X(self, X: pd.DataFrame) -> np.ndarray:
        df = _add_simple_aggs(X)
        cols_needed = [c for c in _SIMPLE_FEATS if c in df.columns]
        return _encode_cats(df[cols_needed], _EV_CAT_COLS)

    def _build_visit_X(self, X: pd.DataFrame) -> np.ndarray:
        cols_needed = [c for c in _VISIT_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_off_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        cols_needed = [c for c in _OFF_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_on_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        cols_needed = [c for c in _ON_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_sev_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        df_aug = _add_sev_ev_pt_features(X)
        all_feats = [c for c in _SEV_EV_VISIT_FEATS + _SEV_EV_PT_FEATS if c in df_aug.columns]
        return _encode_cats(df_aug[all_feats], _EV_CAT_COLS)

    def _compute_s_on_df(self, X: pd.DataFrame) -> tuple:
        """Compute s using self.e_mode and attach columns."""
        s_arr, f_arr, off_present, E_i_arr, nE_i_arr, _ = compute_s_ratio(
            X, E_pop=self.E_pop_, e_mode=self.e_mode,
        )
        df = X.copy()
        df["s_ratio"]     = s_arr
        df["f_drug"]      = f_arr
        df["off_present"] = off_present.astype(float)
        df["nE_i"]        = nE_i_arr
        return df, s_arr, f_arr, off_present, nE_i_arr

    def _predict_evidence(self, X: pd.DataFrame, X_with_s: pd.DataFrame) -> tuple:
        ev_visit = self.model_visit_.predict(self._build_visit_X(X))

        off_col = X["off"].values.astype(float) if "off" in X.columns else np.full(len(X), np.nan)
        ev_off = np.full(len(X), np.nan)
        has_off = ~np.isnan(off_col)
        if has_off.any():
            preds_off = self.model_off_ev_.predict(self._build_off_ev_X(X))
            ev_off[has_off] = preds_off[has_off]

        on_col = X["on"].values.astype(float) if "on" in X.columns else np.full(len(X), np.nan)
        ev_on = np.full(len(X), np.nan)
        has_on = ~np.isnan(on_col)
        if has_on.any():
            preds_on = self.model_on_ev_.predict(self._build_on_ev_X(X))
            ev_on[has_on] = preds_on[has_on]

        ev_sev = self.model_sev_ev_.predict(self._build_sev_ev_X(X_with_s))

        return ev_visit, ev_off, ev_on, ev_sev

    def fit(self, X: pd.DataFrame, y):
        y_arr = np.asarray(y, dtype=float)
        pids  = X["patient_id"].values
        ages  = X["age"].values.astype(float)

        # Compute E_pop using e_mode (so shrunk_mean uses the same E_pop as median)
        _, _, _, _, _, E_pop = compute_s_ratio(X, E_pop=None, e_mode=self.e_mode)
        self.E_pop_ = E_pop

        X_s, s_arr, f_arr, off_present, nE_i_arr = self._compute_s_on_df(X)

        cat_arr_full = _compute_visit_category(off_present, nE_i_arr)
        self.logdens_ = fit_logdens(s_arr, y_arr, cat_arr_full)

        gkf = GroupKFold(n_splits=self.n_splits_inner)
        r_oof       = np.full(len(y_arr), np.nan)
        simp_oof    = np.full(len(y_arr), np.nan)
        visit_oof   = np.full(len(y_arr), np.nan)
        off_ev_oof  = np.full(len(y_arr), np.nan)
        on_ev_oof   = np.full(len(y_arr), np.nan)
        sev_ev_oof  = np.full(len(y_arr), np.nan)

        for tr_idx, val_idx in gkf.split(X, y_arr, groups=pids):
            X_tr  = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr  = y_arr[tr_idx]

            m_base = self._make_base()
            m_base.fit(X_tr.drop(columns=["patient_id"]), y_tr)
            r_oof[val_idx] = m_base.predict(X_val.drop(columns=["patient_id"]))

            m_simp = self._make_ev_model()
            m_simp.fit(self._build_simple_X(X_tr), y_tr)
            simp_oof[val_idx] = m_simp.predict(self._build_simple_X(X_val))

            m_visit = self._make_ev_model()
            m_visit.fit(self._build_visit_X(X_tr), y_tr)
            visit_oof[val_idx] = m_visit.predict(self._build_visit_X(X_val))

            off_tr = X_tr["off"].values.astype(float) if "off" in X_tr.columns else np.full(len(X_tr), np.nan)
            has_off_tr = ~np.isnan(off_tr)
            m_off_ev = self._make_ev_model()
            if has_off_tr.sum() >= 10:
                m_off_ev.fit(self._build_off_ev_X(X_tr)[has_off_tr], y_tr[has_off_tr])
                off_val = X_val["off"].values.astype(float) if "off" in X_val.columns else np.full(len(X_val), np.nan)
                has_off_val = ~np.isnan(off_val)
                if has_off_val.any():
                    preds_off = m_off_ev.predict(self._build_off_ev_X(X_val))
                    off_ev_oof[val_idx[has_off_val]] = preds_off[has_off_val]

            on_tr = X_tr["on"].values.astype(float) if "on" in X_tr.columns else np.full(len(X_tr), np.nan)
            has_on_tr = ~np.isnan(on_tr)
            m_on_ev = self._make_ev_model()
            if has_on_tr.sum() >= 10:
                m_on_ev.fit(self._build_on_ev_X(X_tr)[has_on_tr], y_tr[has_on_tr])
                on_val = X_val["on"].values.astype(float) if "on" in X_val.columns else np.full(len(X_val), np.nan)
                has_on_val = ~np.isnan(on_val)
                if has_on_val.any():
                    preds_on = m_on_ev.predict(self._build_on_ev_X(X_val))
                    on_ev_oof[val_idx[has_on_val]] = preds_on[has_on_val]

            # K: sev_evidence — use e_mode for inner fold s computation
            _, _, _, _, _, E_pop_fold = compute_s_ratio(X_tr, E_pop=None, e_mode=self.e_mode)
            X_s_tr, _, _, _, _ = _add_s_cols(X_tr, E_pop_fold, e_mode=self.e_mode)
            X_s_val, _, _, _, _ = _add_s_cols(X_val, E_pop_fold, e_mode=self.e_mode)
            m_sev_ev = self._make_sev_model()
            m_sev_ev.fit(self._build_sev_ev_X(X_s_tr), y_tr)
            sev_ev_oof[val_idx] = m_sev_ev.predict(self._build_sev_ev_X(X_s_val))

        inner_blend = 0.8 * r_oof + 0.2 * simp_oof
        r_blend = 0.7 * inner_blend + 0.3 * sev_ev_oof
        q_oof = quadratic_projection(pids, ages, r_blend)

        s2_feats = self._stage2_features(
            X, r_blend, q_oof, visit_oof, off_ev_oof, on_ev_oof,
            s_arr, f_arr, off_present, nE_i_arr, self.logdens_,
        )
        print(f"  [fit e_mode={self.e_mode}] Stage-2 matrix shape: {s2_feats.shape}", flush=True)
        residuals = y_arr - q_oof

        unique_pids = np.unique(pids)
        self.stage2_models_ = []
        for seed in range(5):
            rng_sub = np.random.default_rng(seed)
            n_sub = max(1, int(0.8 * len(unique_pids)))
            sub_pids = set(rng_sub.choice(unique_pids, size=n_sub, replace=False))
            sub_mask = np.array([pid in sub_pids for pid in pids])
            valid_mask = sub_mask & np.isfinite(q_oof) & np.isfinite(residuals)
            # Change O: use STAGE2_HGB_PARAMS instead of old underfit defaults
            m_s2 = HistGradientBoostingRegressor(**STAGE2_HGB_PARAMS, random_state=seed)
            m_s2.fit(s2_feats[valid_mask], residuals[valid_mask])
            self.stage2_models_.append(m_s2)

        self.base_ = self._make_base()
        self.base_.fit(X.drop(columns=["patient_id"]), y_arr)

        self.model_simple_ = self._make_ev_model()
        self.model_simple_.fit(self._build_simple_X(X), y_arr)

        self.model_visit_ = self._make_ev_model()
        self.model_visit_.fit(self._build_visit_X(X), y_arr)

        off_col_all = X["off"].values.astype(float) if "off" in X.columns else np.full(len(X), np.nan)
        has_off_all = ~np.isnan(off_col_all)
        self.model_off_ev_ = self._make_ev_model()
        if has_off_all.sum() >= 10:
            self.model_off_ev_.fit(self._build_off_ev_X(X)[has_off_all], y_arr[has_off_all])

        on_col_all = X["on"].values.astype(float) if "on" in X.columns else np.full(len(X), np.nan)
        has_on_all = ~np.isnan(on_col_all)
        self.model_on_ev_ = self._make_ev_model()
        if has_on_all.sum() >= 10:
            self.model_on_ev_.fit(self._build_on_ev_X(X)[has_on_all], y_arr[has_on_all])

        self.model_sev_ev_ = self._make_sev_model()
        self.model_sev_ev_.fit(self._build_sev_ev_X(X_s), y_arr)

        return self

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        pids = X["patient_id"].values
        ages = X["age"].values.astype(float)

        X_s, s_arr, f_arr, off_present, nE_i_arr = self._compute_s_on_df(X)

        r_base  = self.base_.predict(X.drop(columns=["patient_id"]))
        r_simp  = self.model_simple_.predict(self._build_simple_X(X))
        ev_sev  = self.model_sev_ev_.predict(self._build_sev_ev_X(X_s))
        inner   = 0.8 * r_base + 0.2 * r_simp
        r = 0.7 * inner + 0.3 * ev_sev

        q = quadratic_projection(pids, ages, r)

        ev_visit, ev_off, ev_on, _ = self._predict_evidence(X, X_s)

        s2_feats = self._stage2_features(
            X, r, q, ev_visit, ev_off, ev_on,
            s_arr, f_arr, off_present, nE_i_arr, self.logdens_,
        )

        delta = np.mean(
            np.column_stack([m.predict(s2_feats) for m in self.stage2_models_]),
            axis=1,
        )
        p = q + delta

        return quadratic_projection(pids, ages, p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_raw = self._predict_raw(X)

        pids = X["patient_id"].values
        ages = X["age"].values.astype(float)

        output = p_raw.copy()
        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            a_p  = ages[idx]
            p_p  = p_raw[idx]
            valid = ~(np.isnan(a_p) | np.isnan(p_p))
            if valid.sum() < 2:
                continue
            order = np.argsort(a_p[valid])
            a_sorted = a_p[valid][order]
            p_sorted = p_p[valid][order]
            iso = IsotonicRegression(increasing=True)
            iso.fit(a_sorted, p_sorted)
            iso_pred = iso.predict(a_sorted)
            valid_idx = idx[valid]
            sorted_idx = valid_idx[order]
            output[sorted_idx] = iso_pred

        output = np.clip(output, 0, None)
        return output


# =============================================================================
# N: BlendRegressor — average predictions of multiple estimators
# =============================================================================

class BlendRegressor(BaseEstimator, RegressorMixin):
    """Fit each estimator independently; predict the mean of their outputs."""

    def __init__(self, estimators: list):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return preds.mean(axis=1)


# =============================================================================
# Data loading
# =============================================================================

print("Loading data...", flush=True)
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")
groups_train = X_train_raw["patient_id"]

if SMOKE:
    rng = np.random.default_rng(0)
    all_pids  = groups_train.unique()
    sub_pids  = rng.choice(all_pids, size=max(4, len(all_pids) // 10), replace=False)
    smoke_mask = groups_train.isin(sub_pids)
    X_train_raw  = X_train_raw[smoke_mask]
    y_train      = y_train[smoke_mask]
    groups_train = groups_train[smoke_mask]
    print(f"  [SMOKE] {len(X_train_raw)} rows, {groups_train.nunique()} patients", flush=True)


def _add_tsd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


# =============================================================================
# Feature pipeline (verbatim from 19)
# =============================================================================

HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)

ledd_interp_tr = LeddInterp()


def _build_features_14(
    df_raw: pd.DataFrame,
    pids: np.ndarray,
    diag_imp: DiagnosisImputer,
    pcp: PatientCorrectedPooling,
    ml: MixedLine,
    upper_stats: UpperStats,
    implied_off_tr: ImpliedOffTransformer,
) -> pd.DataFrame:
    """Full supervised feature stack."""
    df = diag_imp.transform(df_raw)
    df = ledd_interp_tr.transform(_add_tsd(df))
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df, ledd_col="ledd_interp")
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    df = upper_stats.transform(df)
    df = implied_off_tr.transform(df)
    return df


# =============================================================================
# Smoke diagnostic: E_i comparison between modes on full training data
# =============================================================================

def _print_e_diagnostics():
    """Print sanity checks for shrunk_mean vs median E_i on full training data."""
    print("\n--- E_i diagnostics (full training data) ---", flush=True)

    # Compute both modes; E_pop is derived identically in both (median of E_vals_paired)
    _, _, _, _, nE_i_arr_med, E_pop_med = compute_s_ratio(
        X_train_raw, E_pop=None, e_mode="median"
    )
    _, _, _, E_i_sm, nE_i_arr_sm, _ = compute_s_ratio(
        X_train_raw, E_pop=None, e_mode="shrunk_mean"
    )
    _, _, _, E_i_med, _, _ = compute_s_ratio(
        X_train_raw, E_pop=None, e_mode="median"
    )

    pids = X_train_raw["patient_id"].values
    # First row per patient
    pt_first = {}
    for i, pid in enumerate(pids):
        if pid not in pt_first:
            pt_first[pid] = i
    pt_idx = np.array(list(pt_first.values()))
    pt_nE  = nE_i_arr_med[pt_idx]

    print(f"  E_pop (median mode): {E_pop_med:.4f}  (expected ~0.435)", flush=True)

    # Correlation between E_i of both modes for patients with nE_i > 0
    has_paired = pt_nE > 0
    ei_med_has = E_i_med[pt_idx[has_paired]]
    ei_sm_has  = E_i_sm[pt_idx[has_paired]]
    corr = float(np.corrcoef(ei_med_has, ei_sm_has)[0, 1])
    print(f"  Corr(E_i_median, E_i_shrunk) for nE>0: {corr:.4f}  (expected ~0.97)", flush=True)

    # Mean |E_i_shrunk - E_i_median| for patients with nE_i = 1
    nE1 = pt_nE == 1
    if nE1.sum() > 0:
        diff1 = float(np.mean(np.abs(E_i_sm[pt_idx[nE1]] - E_i_med[pt_idx[nE1]])))
        print(f"  Mean |E_i diff| for nE=1 patients: {diff1:.4f}  (expected ~0.036)", flush=True)
    else:
        print("  No patients with nE_i=1 in this subset.", flush=True)


# =============================================================================
# CV loop for one e_mode (reusable for both median and shrunk_mean)
# =============================================================================

import time as _time


def run_cv_one_mode(e_mode: str, tag_raw: str, tag_iso: str, smoke_probe: bool = False):
    """Run GroupKFold(5) CV for TwoStageRegressor with the given e_mode.

    Parameters
    ----------
    e_mode      : "median" or "shrunk_mean"
    tag_raw     : column name for _predict_raw output (e.g. "v21a_raw")
    tag_iso     : column name for predict output     (e.g. "v21a")
    smoke_probe : if True, print n_iter_ and fit time of first stage-2 model
    """
    y = y_train
    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    oof_raw = np.full(len(y), np.nan)
    oof_iso = np.full(len(y), np.nan)
    fold_rmses_raw = []
    fold_rmses_iso = []

    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf.split(X_train_raw, y, groups=groups_train)
    ):
        X_raw_tr  = X_train_raw.iloc[tr_idx]
        X_raw_val = X_train_raw.iloc[val_idx]
        y_tr      = y.iloc[tr_idx]
        pids_tr   = groups_train.iloc[tr_idx].values
        pids_val  = groups_train.iloc[val_idx].values

        diag_imp = DiagnosisImputer()
        diag_imp.fit(X_raw_tr)

        X_li_tr = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_tr)))

        pcp = PatientCorrectedPooling(n_splits=N_SPLITS_CV, use_ledd_interp=True)
        pcp.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        ml = MixedLine(pcp=pcp)
        ml.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        X_tr_base  = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_tr)))
        X_tr_base  = pcp.transform(X_tr_base)
        X_val_base = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_val)))
        X_val_base = pcp.transform(X_val_base)

        X_tr_up_in = ml.transform(X_tr_base, patient_ids=pids_tr)
        X_tr_up_in = add_patient_features(X_tr_up_in, ledd_col="ledd_interp")
        X_tr_up_in["cohort"] = X_tr_up_in["cohort"].astype("category")
        X_tr_up_in["gene"]   = X_tr_up_in["gene"].astype("category")
        us = UpperStats()
        us.fit(X_tr_up_in, y_tr)

        implied_off_tr_fold = ImpliedOffTransformer()
        implied_off_tr_fold.fit(X_tr_base)

        X_tr  = _build_features_14(X_raw_tr,  pids_tr,  diag_imp, pcp, ml, us, implied_off_tr_fold)
        X_val = _build_features_14(X_raw_val, pids_val, diag_imp, pcp, ml, us, implied_off_tr_fold)

        est = TwoStageRegressor(base_params=HGB_BEST_08, n_splits_inner=N_SPLITS_CV, e_mode=e_mode)

        if smoke_probe and fold_i == 0:
            # Time a single stage-2 model fit to verify Change O
            t0 = _time.perf_counter()
            est.fit(X_tr, y_tr)
            t1 = _time.perf_counter()
            n_iter_s2 = est.stage2_models_[0].n_iter_
            print(
                f"  [SMOKE probe] stage2[0].n_iter_={n_iter_s2}  "
                f"(expected 900, early_stopping=False)",
                flush=True,
            )
            print(
                f"  [SMOKE probe] time for one TwoStageRegressor.fit: {t1-t0:.1f}s",
                flush=True,
            )
        else:
            est.fit(X_tr, y_tr)

        val_raw = est._predict_raw(X_val)
        val_iso = est.predict(X_val)

        oof_raw[val_idx] = val_raw
        oof_iso[val_idx] = val_iso

        rmse_raw = root_mean_squared_error(y.iloc[val_idx], val_raw)
        rmse_iso = root_mean_squared_error(y.iloc[val_idx], val_iso)
        fold_rmses_raw.append(rmse_raw)
        fold_rmses_iso.append(rmse_iso)
        print(
            f"  Fold {fold_i+1}/{N_SPLITS_CV}: "
            f"{tag_raw}={rmse_raw:.4f}  {tag_iso}={rmse_iso:.4f}",
            flush=True,
        )

    return {
        tag_raw: {"oof": oof_raw, "folds": fold_rmses_raw,
                  "mean": float(np.mean(fold_rmses_raw)),
                  "std":  float(np.std(fold_rmses_raw, ddof=1))},
        tag_iso: {"oof": oof_iso, "folds": fold_rmses_iso,
                  "mean": float(np.mean(fold_rmses_iso)),
                  "std":  float(np.std(fold_rmses_iso, ddof=1))},
    }


# =============================================================================
# Run
# =============================================================================

audit_dir = PROJECT_ROOT / "audit"
audit_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark: recompute v20 fold RMSEs from audit/oof_20.csv on the SAME splits
# ---------------------------------------------------------------------------
print("Loading v20 benchmark from audit/oof_20.csv...", flush=True)
oof_20_df = pd.read_csv(audit_dir / "oof_20.csv").set_index("Index")
v20_oof = oof_20_df["v20"].reindex(X_train_raw.index).values
y_arr_all = y_train.values

gkf_bench = GroupKFold(n_splits=5)
BENCHMARK_FOLDS = []
for _, val_idx in gkf_bench.split(X_train_raw, y_train, groups=groups_train):
    BENCHMARK_FOLDS.append(
        root_mean_squared_error(y_arr_all[val_idx], v20_oof[val_idx])
    )
BENCHMARK_RMSE = float(np.mean(BENCHMARK_FOLDS))

print(f"\n=== v20 benchmark (from oof_20.csv, GroupKFold(5)) ===", flush=True)
print(f"  mean RMSE: {BENCHMARK_RMSE:.4f}  std: {float(np.std(BENCHMARK_FOLDS, ddof=1)):.4f}", flush=True)
for i, fv in enumerate(BENCHMARK_FOLDS):
    print(f"  Fold {i+1}: v20={fv:.4f}", flush=True)

# ---------------------------------------------------------------------------
# Step 2: Smoke test (SMOKE_TEST=1) — median mode only, 2 folds, no CSV
# ---------------------------------------------------------------------------

if SMOKE:
    print("\n=== [SMOKE] Step 2: smoke test — median mode, 2 outer folds ===", flush=True)
    run_cv_one_mode("median", "v21a_raw", "v21a", smoke_probe=True)
    print("\n[SMOKE] All checks passed.", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 3: Full CV for median model — load from audit/oof_21a.csv if present
# ---------------------------------------------------------------------------

oof_21a_path = audit_dir / "oof_21a.csv"
if oof_21a_path.exists():
    print("\nLoading existing audit/oof_21a.csv (skipping CV for median mode).", flush=True)
    oof_21a_df = pd.read_csv(oof_21a_path)
    results_21a = {
        "v21a_raw": {"oof": oof_21a_df["v21a_raw"].values, "folds": [], "mean": np.nan, "std": np.nan},
        "v21a":     {"oof": oof_21a_df["v21a"].values,     "folds": [], "mean": np.nan, "std": np.nan},
    }
    # Recompute per-fold RMSEs from loaded OOF
    gkf_reload = GroupKFold(n_splits=5)
    fold_21a_raw = []
    fold_21a_iso = []
    for _, val_idx in gkf_reload.split(X_train_raw, y_train, groups=groups_train):
        fold_21a_raw.append(root_mean_squared_error(y_arr_all[val_idx], results_21a["v21a_raw"]["oof"][val_idx]))
        fold_21a_iso.append(root_mean_squared_error(y_arr_all[val_idx], results_21a["v21a"]["oof"][val_idx]))
    results_21a["v21a_raw"]["folds"] = fold_21a_raw
    results_21a["v21a"]["folds"]     = fold_21a_iso
    results_21a["v21a_raw"]["mean"]  = float(np.mean(fold_21a_raw))
    results_21a["v21a"]["mean"]      = float(np.mean(fold_21a_iso))
    results_21a["v21a_raw"]["std"]   = float(np.std(fold_21a_raw, ddof=1))
    results_21a["v21a"]["std"]       = float(np.std(fold_21a_iso, ddof=1))
else:
    print("\n=== Step 3: Full CV — TwoStageRegressor(e_mode='median') ===", flush=True)
    results_21a = run_cv_one_mode("median", "v21a_raw", "v21a")
    oof_21a_save = pd.DataFrame({
        "Index":      X_train_raw.index,
        "patient_id": groups_train.values,
        "v21a_raw":   results_21a["v21a_raw"]["oof"],
        "v21a":       results_21a["v21a"]["oof"],
    })
    oof_21a_save.to_csv(oof_21a_path, index=False)
    print(f"OOF saved -> audit/oof_21a.csv", flush=True)

# ---------------------------------------------------------------------------
# Step 4: Full CV for shrunk_mean model — load from audit/oof_21b.csv if present
# ---------------------------------------------------------------------------

oof_21b_path = audit_dir / "oof_21b.csv"
if oof_21b_path.exists():
    print("\nLoading existing audit/oof_21b.csv (skipping CV for shrunk_mean mode).", flush=True)
    oof_21b_df = pd.read_csv(oof_21b_path)
    results_21b = {
        "v21b_raw": {"oof": oof_21b_df["v21b_raw"].values, "folds": [], "mean": np.nan, "std": np.nan},
        "v21b":     {"oof": oof_21b_df["v21b"].values,     "folds": [], "mean": np.nan, "std": np.nan},
    }
    gkf_reload = GroupKFold(n_splits=5)
    fold_21b_raw = []
    fold_21b_iso = []
    for _, val_idx in gkf_reload.split(X_train_raw, y_train, groups=groups_train):
        fold_21b_raw.append(root_mean_squared_error(y_arr_all[val_idx], results_21b["v21b_raw"]["oof"][val_idx]))
        fold_21b_iso.append(root_mean_squared_error(y_arr_all[val_idx], results_21b["v21b"]["oof"][val_idx]))
    results_21b["v21b_raw"]["folds"] = fold_21b_raw
    results_21b["v21b"]["folds"]     = fold_21b_iso
    results_21b["v21b_raw"]["mean"]  = float(np.mean(fold_21b_raw))
    results_21b["v21b"]["mean"]      = float(np.mean(fold_21b_iso))
    results_21b["v21b_raw"]["std"]   = float(np.std(fold_21b_raw, ddof=1))
    results_21b["v21b"]["std"]       = float(np.std(fold_21b_iso, ddof=1))
else:
    print("\n=== Step 4: Full CV — TwoStageRegressor(e_mode='shrunk_mean') ===", flush=True)
    results_21b = run_cv_one_mode("shrunk_mean", "v21b_raw", "v21b")
    oof_21b_save = pd.DataFrame({
        "Index":      X_train_raw.index,
        "patient_id": groups_train.values,
        "v21b_raw":   results_21b["v21b_raw"]["oof"],
        "v21b":       results_21b["v21b"]["oof"],
    })
    oof_21b_save.to_csv(oof_21b_path, index=False)
    print(f"OOF saved -> audit/oof_21b.csv", flush=True)

# ---------------------------------------------------------------------------
# Step 5: Blend v21 = mean(v21a, v21b); save audit/oof_21.csv
# ---------------------------------------------------------------------------

v21a_raw = results_21a["v21a_raw"]["oof"]
v21a     = results_21a["v21a"]["oof"]
v21b_raw = results_21b["v21b_raw"]["oof"]
v21b     = results_21b["v21b"]["oof"]

v21_raw = (v21a_raw + v21b_raw) / 2.0
v21     = (v21a     + v21b)     / 2.0

oof_21_path = audit_dir / "oof_21.csv"
oof_21_save = pd.DataFrame({
    "Index":      X_train_raw.index,
    "patient_id": groups_train.values,
    "v21_raw":    v21_raw,
    "v21":        v21,
})
oof_21_save.to_csv(oof_21_path, index=False)
print(f"\nOOF saved -> audit/oof_21.csv", flush=True)

# Blend per-fold RMSEs (recompute from OOF arrays on same splits)
fold_21_raw = []
fold_21_iso = []
gkf_blend = GroupKFold(n_splits=5)
for _, val_idx in gkf_blend.split(X_train_raw, y_train, groups=groups_train):
    fold_21_raw.append(root_mean_squared_error(y_arr_all[val_idx], v21_raw[val_idx]))
    fold_21_iso.append(root_mean_squared_error(y_arr_all[val_idx], v21[val_idx]))

# ---------------------------------------------------------------------------
# Step 6: Print diagnostics
# ---------------------------------------------------------------------------

# Load reference OOFs for cross-experiment comparison
oof_19_df  = pd.read_csv(audit_dir / "oof_19.csv").set_index("Index")
oof_20b_df = pd.read_csv(audit_dir / "oof_20b.csv").set_index("Index")
v19    = oof_19_df["v19"].reindex(X_train_raw.index).values
v20b   = oof_20b_df["v20b"].reindex(X_train_raw.index).values

print("\n=== Step 6: CV diagnostics ===", flush=True)

# Helper: print one model's fold table vs benchmark
def _print_folds(tag, fold_rmses_tag, bench_folds, bench_tag):
    mean_r = float(np.mean(fold_rmses_tag))
    std_r  = float(np.std(fold_rmses_tag, ddof=1))
    n_imp  = sum(1 for f, b in zip(fold_rmses_tag, bench_folds) if f < b)
    print(f"\n  {tag}: mean={mean_r:.4f} +/- {std_r:.4f}", flush=True)
    for i, (fv, fb) in enumerate(zip(fold_rmses_tag, bench_folds)):
        delta = fv - fb
        improved = "BETTER" if delta < 0 else "worse"
        print(
            f"    Fold {i+1}: {tag}={fv:.4f}  {bench_tag}={fb:.4f}  "
            f"delta={delta:+.4f}  [{improved}]",
            flush=True,
        )
    print(f"    Folds improved vs {bench_tag}: {n_imp}/5", flush=True)
    return mean_r, std_r, n_imp

# v21a vs v20 (primary comparison) and vs v19 (Change O effect on median)
_print_folds("v21a", results_21a["v21a"]["folds"], BENCHMARK_FOLDS, "v20")
# v21a vs v19 to isolate change O effect on median model
gkf_19 = GroupKFold(n_splits=5)
fold_v19 = [root_mean_squared_error(y_arr_all[vi], v19[vi])
            for _, vi in gkf_19.split(X_train_raw, y_train, groups=groups_train)]
_print_folds("v21a_vs_v19", results_21a["v21a"]["folds"], fold_v19, "v19")

# v21b vs v20 and vs v20b (Change O effect on shrunk_mean)
_print_folds("v21b", results_21b["v21b"]["folds"], BENCHMARK_FOLDS, "v20")
fold_v20b = [root_mean_squared_error(y_arr_all[vi], v20b[vi])
             for _, vi in GroupKFold(n_splits=5).split(X_train_raw, y_train, groups=groups_train)]
_print_folds("v21b_vs_v20b", results_21b["v21b"]["folds"], fold_v20b, "v20b")

# v21 (blend) vs v20
mean_v21, std_v21, n_imp_v21 = _print_folds("v21", fold_21_iso, BENCHMARK_FOLDS, "v20")

# Per-OFF-group RMSE for v21 vs v20
print("\n  RMSE by number of OFF visits (v21 vs v20):", flush=True)
n_off_per_pt = X_train_raw.groupby("patient_id")["off"].apply(lambda s: s.notna().sum())
for label, lo, hi in [("0 OFF", 0, 0), ("1-2 OFF", 1, 2), ("3-5 OFF", 3, 5), ("6+ OFF", 6, 999)]:
    pts = n_off_per_pt[(n_off_per_pt >= lo) & (n_off_per_pt <= hi)].index
    idx_mask = X_train_raw["patient_id"].isin(pts)
    y_grp    = y_arr_all[idx_mask.values]
    v21_grp  = v21[idx_mask.values]
    v20_grp  = v20_oof[idx_mask.values]
    valid    = ~np.isnan(v21_grp) & ~np.isnan(v20_grp)
    if valid.sum() == 0:
        continue
    r21 = root_mean_squared_error(y_grp[valid], v21_grp[valid])
    r20 = root_mean_squared_error(y_grp[valid], v20_grp[valid])
    print(f"    {label}: v21={r21:.4f}  v20={r20:.4f}  delta={r21-r20:+.4f}", flush=True)

# ---------------------------------------------------------------------------
# Step 7: Decision
# ---------------------------------------------------------------------------

print("\n=== Decision ===", flush=True)
qualifies = (mean_v21 < BENCHMARK_RMSE) and (n_imp_v21 >= 4)
print(
    f"  v21: mean={mean_v21:.4f}  benchmark(v20)={BENCHMARK_RMSE:.4f}  "
    f"improved={n_imp_v21}/5  -> {'KEEP' if qualifies else 'discard'}",
    flush=True,
)

if not qualifies:
    print("\nv21 does not meet the keep criterion. Stopping.", flush=True)
    sys.exit(0)

print(f"\nKeeping v21 as {EXPERIMENT_KEY}.", flush=True)

# ---------------------------------------------------------------------------
# Step 8: Keep path — skore report + submission
# ---------------------------------------------------------------------------

print("\nBuilding skore holdout report...", flush=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
hub_tr_idx, hub_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

X_raw_hub_tr  = X_train_raw.iloc[hub_tr_idx]
X_raw_hub_val = X_train_raw.iloc[hub_val_idx]
y_hub_tr      = y_train.iloc[hub_tr_idx]
y_hub_val     = y_train.iloc[hub_val_idx]
pids_hub_tr   = groups_train.iloc[hub_tr_idx].values
pids_hub_val  = groups_train.iloc[hub_val_idx].values

diag_imp_hub = DiagnosisImputer()
diag_imp_hub.fit(X_raw_hub_tr)

X_li_hub = ledd_interp_tr.transform(_add_tsd(diag_imp_hub.transform(X_raw_hub_tr)))
pcp_hub = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)
ml_hub = MixedLine(pcp=pcp_hub)
ml_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)

X_hub_base_tr = ledd_interp_tr.transform(_add_tsd(diag_imp_hub.transform(X_raw_hub_tr)))
X_hub_base_tr = pcp_hub.transform(X_hub_base_tr)
X_hub_ml_tr   = ml_hub.transform(X_hub_base_tr, patient_ids=pids_hub_tr)
X_hub_ml_tr   = add_patient_features(X_hub_ml_tr, ledd_col="ledd_interp")
X_hub_ml_tr["cohort"] = X_hub_ml_tr["cohort"].astype("category")
X_hub_ml_tr["gene"]   = X_hub_ml_tr["gene"].astype("category")
us_hub = UpperStats(); us_hub.fit(X_hub_ml_tr, y_hub_tr)
iot_hub = ImpliedOffTransformer(); iot_hub.fit(X_hub_base_tr)

X_hub_tr  = _build_features_14(X_raw_hub_tr,  pids_hub_tr,  diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)
X_hub_val = _build_features_14(X_raw_hub_val, pids_hub_val, diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)

# BlendRegressor with the new stage-2 (STAGE2_HGB_PARAMS already in TwoStageRegressor.fit)
est_hub = BlendRegressor([
    TwoStageRegressor(base_params=HGB_BEST_08, e_mode="median"),
    TwoStageRegressor(base_params=HGB_BEST_08, e_mode="shrunk_mean"),
])
rep = EstimatorReport(
    est_hub,
    X_train=X_hub_tr, y_train=y_hub_tr,
    X_test=X_hub_val,  y_test=y_hub_val,
)
project.put(EXPERIMENT_KEY, rep)
HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Pushed {EXPERIMENT_KEY}: {HUB_URL}", flush=True)

# Refit both models on all training data
print("\nRefitting both models on all training data for test predictions...", flush=True)
pids_all = groups_train.values

diag_imp_all = DiagnosisImputer()
diag_imp_all.fit(X_train_raw)
X_all_base = ledd_interp_tr.transform(_add_tsd(diag_imp_all.transform(X_train_raw)))

pcp_all = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_all.fit(X_all_base, y_train, patient_ids=pids_all)
ml_all = MixedLine(pcp=pcp_all)
ml_all.fit(X_all_base, y_train, patient_ids=pids_all)

X_all_base_corr = pcp_all.transform(X_all_base)
X_all_ml = ml_all.transform(X_all_base_corr, patient_ids=pids_all)
X_all_ml = add_patient_features(X_all_ml, ledd_col="ledd_interp")
X_all_ml["cohort"] = X_all_ml["cohort"].astype("category")
X_all_ml["gene"]   = X_all_ml["gene"].astype("category")
us_all = UpperStats(); us_all.fit(X_all_ml, y_train)
iot_all = ImpliedOffTransformer(); iot_all.fit(X_all_base_corr)

X_all_final = _build_features_14(X_train_raw, pids_all, diag_imp_all, pcp_all, ml_all, us_all, iot_all)

print("  Fitting median model...", flush=True)
est_med_final = TwoStageRegressor(base_params=HGB_BEST_08, e_mode="median")
est_med_final.fit(X_all_final, y_train)

print("  Fitting shrunk_mean model...", flush=True)
est_sm_final = TwoStageRegressor(base_params=HGB_BEST_08, e_mode="shrunk_mean")
est_sm_final.fit(X_all_final, y_train)

# Predict test set
print("Predicting test set...", flush=True)
pids_test  = X_test_raw["patient_id"].values
X_test_feat = _build_features_14(X_test_raw, pids_test, diag_imp_all, pcp_all, ml_all, us_all, iot_all)

test_preds_med = est_med_final.predict(X_test_feat)
test_preds_sm  = est_sm_final.predict(X_test_feat)
test_preds_21  = (test_preds_med + test_preds_sm) / 2.0

# Load previous blend submission for comparison
sub_dir  = PROJECT_ROOT / "submissions"
sub_20   = pd.read_csv(sub_dir / "20_blend.csv")
assert sub_20["Index"].equals(sample_sub["Index"]), "Index mismatch: 20_blend vs sample_submission"
test_preds_20 = sub_20["target"].values

# Assertions
assert len(test_preds_21) == 11013, f"Expected 11013 predictions, got {len(test_preds_21)}"
n_finite_21 = int(np.isfinite(test_preds_21).sum())
assert n_finite_21 == 11013, f"Expected 11013 finite predictions, got {n_finite_21}"
assert np.all(test_preds_21 >= 0), "Some test predictions are negative"

# Print prediction stats
print("\nPrediction stats (test vs OOF):", flush=True)
print(f"  median    test: mean={test_preds_med.mean():.3f}  std={test_preds_med.std():.3f}", flush=True)
print(f"  shrunk_m  test: mean={test_preds_sm.mean():.3f}  std={test_preds_sm.std():.3f}", flush=True)
print(f"  v21 blend test: mean={test_preds_21.mean():.3f}  std={test_preds_21.std():.3f}", flush=True)
print(f"  v20 blend test: mean={test_preds_20.mean():.3f}  std={test_preds_20.std():.3f}", flush=True)
print(f"  v21a OOF:       mean={np.nanmean(v21a):.3f}  std={np.nanstd(v21a):.3f}", flush=True)
print(f"  v21b OOF:       mean={np.nanmean(v21b):.3f}  std={np.nanstd(v21b):.3f}", flush=True)
print(f"  v21  OOF:       mean={np.nanmean(v21):.3f}  std={np.nanstd(v21):.3f}", flush=True)

# Write submission
sub_path = sub_dir / f"{EXPERIMENT_KEY}.csv"
sub_df = pd.DataFrame({"Index": sample_sub["Index"], "target": test_preds_21})
sub_df.to_csv(sub_path, index=False)
print(f"\nSubmission written -> submissions/{EXPERIMENT_KEY}.csv", flush=True)
print(f"Report URL: {HUB_URL}", flush=True)
