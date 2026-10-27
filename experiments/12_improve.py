# %% [markdown]
# # Experiment 12 — improve
#
# **Date:** 2026-10-11
# **Goal:** Improve visits where OFF is missing by adding reliability-aware
# implied-OFF features estimated from ON scores.
#
# Strategy:
#   For each patient, compute the median gap (off_corr - on_corr) from visits
#   where both are observed.  Shrink toward a population-level gap predicted
#   by a simple HGB fitted on paired training-fold visits (features:
#   time_since_intake_on, ledd_interp, time_since_diagnosis, age_at_diagnosis).
#   Shrinkage weight w = n / (n + k) where k is calibrated on the training fold.
#
# New features added on top of 11_B:
#   implied_off       — on_corr + shrunk patient gap  (NaN if on_corr missing)
#   gap_weight        — empirical-Bayes shrinkage weight [0, 1]
#   n_paired          — number of visits where both off_corr and on_corr valid
#
# Benchmark: 11_B  CV RMSE 3.6472 +/- 0.0789 (GroupKFold 5).
# Keep criterion: mean RMSE < 3.6472 AND >= 4 of 5 folds improve.
#
# **Key:** 12_improve  |  **Decision:** kept, better than 11_B on 5/5 folds; replaced by 13
# **Results:** CV (GroupKFold 5) 3.5987 ± 0.0622  |  holdout (20 % of patients) 3.59
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/60965
# **Run:** `python experiments/12_improve.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/12_improve.py`).
# Reads data/ (first version of the files, before 9 October) and audit/oof_11_B.csv; needs the
# Skore key (.skore), see the report (report/rapport_SPS.pdf), How to run.
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
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "12_improve"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SPLITS_CV = 2 if SMOKE else 5

BENCHMARK_RMSE = 3.6472
BENCHMARK_FOLDS = [3.6987, 3.5603, 3.6219, 3.7137, 3.6414]  # 11_B per-fold

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
# Shared primitives (verbatim from 11_leakage_fix)
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
        pid  = X["patient_id"].values
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


# =============================================================================
# NEW: ImpliedOffTransformer — reliability-aware OFF-from-ON feature
# =============================================================================

_POP_GAP_FEATURES = [
    "time_since_intake_on", "ledd_interp", "time_since_diagnosis", "age_at_diagnosis",
]


class ImpliedOffTransformer(BaseEstimator, TransformerMixin):
    """Add implied_off, gap_weight, n_paired features.

    Fitted on training-fold data only (no target leakage).

    For each patient, the gap is median(off_corr - on_corr) over visits where
    both are observed.  The gap is shrunk toward a population-level model
    prediction using w = n / (n + k).  k is estimated on the training fold
    by minimising the leave-one-patient-out MSE of the shrunk gap vs the
    per-patient sample gap.

    All columns are computed from off_corr / on_corr columns (already present
    in the DataFrame, produced by PatientCorrectedPooling.transform), so no
    target information is used.
    """

    def _compute_population_gaps(self, df: pd.DataFrame) -> np.ndarray:
        """Predict population-level gap for every row in df."""
        X_pop = df[_POP_GAP_FEATURES].copy()
        return self._pop_model_.predict(X_pop.values)

    def fit(self, df: pd.DataFrame, y=None):
        off_corr = df["off_corr"].values.astype(float)
        on_corr  = df["on_corr"].values.astype(float)
        pids     = df["patient_id"].values

        # Per-patient median gap on paired visits
        pt_gaps   = {}   # patient_id -> float (NaN if no paired visits)
        pt_n      = {}   # patient_id -> int
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

        # Fit population gap model on rows that have paired visits
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
            # Fallback: constant model
            from sklearn.dummy import DummyRegressor
            self._pop_model_ = DummyRegressor(strategy="mean")
            self._pop_model_.fit(feat_df.values, gap_vals)

        # Estimate k via leave-one-patient-out on patients with n_paired >= 1
        pids_with_gap = [p for p, g in pt_gaps.items() if not np.isnan(g)]
        if len(pids_with_gap) < 5:
            self._k_ = 1.0
            return self

        pop_preds_all = self._compute_population_gaps(df)
        # Map row index to population prediction
        row_pop = dict(zip(range(len(df)), pop_preds_all))

        # For each patient, population prediction = mean of pop_preds for its paired rows
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

        # Grid search k: minimise sum over patients of (w*g_pt + (1-w)*g_pop - g_pt)^2
        # = sum (1-w)^2 * (g_pop - g_pt)^2  ->  trivially 0 if g_pop = g_pt.
        # Better: LOOCV MSE with respect to the leave-one-patient-out shrunk gap.
        # We want: shrunk_gap_i = w_i * g_i + (1-w_i) * pop_i  where w_i = n_i/(n_i+k)
        # Evaluate against g_i (the best we can estimate on fold train): minimise
        # sum_i (shrunk_gap_i - g_i)^2  which is (1-w_i)^2 * (pop_i - g_i)^2.
        # This is always minimised by k=0.  Instead, use the heldout-visit MSE:
        # For patients with n>=2, hold out one paired visit, compute shrunk gap
        # from the rest, predict that visit's implied OFF, compare to off_corr.
        # Approximate: scan k in log-space, pick minimiser.
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
                # LOOCV over paired visits
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
                # Test patient seen in training — use stored gap (unshrunk)
                g_pt = self._pt_gaps_[pid]
                n_p  = self._pt_n_[pid]
            else:
                # Fully unseen patient with no paired visits: rely on pop only
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
            # implied_off: only meaningful where on_corr is available
            has_on = ~np.isnan(nc)
            implied_off[idx[has_on]] = nc[has_on] + shrunk_gap

        df["implied_off"] = implied_off
        df["gap_weight"]  = gap_weight
        df["n_paired"]    = n_paired_arr
        return df


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
# Feature pipeline (strictly nested)
# =============================================================================

HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)
BEST_ALPHA_08 = 0.3

ledd_interp_tr = LeddInterp()   # stateless


def _build_features_12(
    df_raw: pd.DataFrame,
    pids: np.ndarray,
    pcp: PatientCorrectedPooling,
    ml: MixedLine,
    upper_stats: UpperStats,
    implied_off_tr: ImpliedOffTransformer,
) -> pd.DataFrame:
    """Full supervised feature stack for Experiment 12."""
    df = ledd_interp_tr.transform(_add_tsd(df_raw))
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df, ledd_col="ledd_interp")
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    df = upper_stats.transform(df)
    df = implied_off_tr.transform(df)
    return df


# =============================================================================
# CV loop
# =============================================================================

def run_cv_12(y=None):
    if y is None:
        y = y_train
    gkf  = GroupKFold(n_splits=N_SPLITS_CV)
    oof  = np.full(len(y), np.nan)
    fold_rmses = []

    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf.split(X_train_raw, y, groups=groups_train)
    ):
        X_raw_tr  = X_train_raw.iloc[tr_idx]
        X_raw_val = X_train_raw.iloc[val_idx]
        y_tr      = y.iloc[tr_idx]
        pids_tr   = groups_train.iloc[tr_idx].values
        pids_val  = groups_train.iloc[val_idx].values

        # Fit supervised transformers on train fold
        X_li_tr = ledd_interp_tr.transform(_add_tsd(X_raw_tr))

        pcp = PatientCorrectedPooling(n_splits=N_SPLITS_CV, use_ledd_interp=True)
        pcp.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        ml = MixedLine(pcp=pcp)
        ml.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        # Build intermediate features (off_corr/on_corr) for the fold
        X_tr_base  = ledd_interp_tr.transform(_add_tsd(X_raw_tr))
        X_tr_base  = pcp.transform(X_tr_base)
        X_val_base = ledd_interp_tr.transform(_add_tsd(X_raw_val))
        X_val_base = pcp.transform(X_val_base)

        # Fit UpperStats on train features
        X_tr_up_in = ml.transform(X_tr_base, patient_ids=pids_tr)
        X_tr_up_in = add_patient_features(X_tr_up_in, ledd_col="ledd_interp")
        X_tr_up_in["cohort"] = X_tr_up_in["cohort"].astype("category")
        X_tr_up_in["gene"]   = X_tr_up_in["gene"].astype("category")
        us = UpperStats()
        us.fit(X_tr_up_in, y_tr)

        # Fit ImpliedOffTransformer on train fold (off_corr/on_corr already in X_tr_base)
        implied_off_tr_fold = ImpliedOffTransformer()
        implied_off_tr_fold.fit(X_tr_base)

        # Build final features
        X_tr  = _build_features_12(X_raw_tr,  pids_tr,  pcp, ml, us, implied_off_tr_fold)
        X_val = _build_features_12(X_raw_val, pids_val, pcp, ml, us, implied_off_tr_fold)

        est = PatientSmoothedRegressor(
            base=HistGradientBoostingRegressor(**HGB_BEST_08),
            alpha=BEST_ALPHA_08,
        )
        est.fit(X_tr, y_tr)
        preds_val = est.predict(X_val)
        oof[val_idx] = preds_val

        fold_rmse = root_mean_squared_error(y.iloc[val_idx], preds_val)
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_i+1}/{N_SPLITS_CV}: RMSE = {fold_rmse:.4f}", flush=True)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse  = float(np.std(fold_rmses, ddof=1))
    return mean_rmse, std_rmse, oof, fold_rmses


# =============================================================================
# Run
# =============================================================================

print("\n=== Experiment 12 CV (GroupKFold 5) ===", flush=True)
rmse_mean, rmse_std, oof_12, folds_12 = run_cv_12()
print(f"  12_improve: {rmse_mean:.4f} +/- {rmse_std:.4f}", flush=True)

if SMOKE:
    print("\n[SMOKE] All checks passed.", flush=True)
    import sys; sys.exit(0)

# =============================================================================
# Save OOF
# =============================================================================

audit_dir = PROJECT_ROOT / "audit"
audit_dir.mkdir(exist_ok=True)
oof_df = pd.DataFrame({
    "Index":      X_train_raw.index,
    "patient_id": groups_train.values,
    "oof":        oof_12,
})
oof_path = audit_dir / "oof_12.csv"
oof_df.to_csv(oof_path, index=False)
print(f"OOF saved -> audit/oof_12.csv", flush=True)

# =============================================================================
# Decision analysis
# =============================================================================

oof_11b = pd.read_csv(audit_dir / "oof_11_B.csv")

# Per-fold deltas
gkf5 = GroupKFold(n_splits=5)
fold_rmses_11b = []
for fold_i, (tr_idx, val_idx) in enumerate(
    gkf5.split(X_train_raw, y_train, groups=groups_train)
):
    y_val     = y_train.iloc[val_idx]
    oof_val   = oof_11b.set_index("Index").reindex(X_train_raw.index[val_idx])["oof"].values
    fold_rmses_11b.append(root_mean_squared_error(y_val, oof_val))

print("\n=== Per-fold comparison ===", flush=True)
n_improved = 0
for i, (f12, f11) in enumerate(zip(folds_12, fold_rmses_11b)):
    delta = f12 - f11
    improved = "BETTER" if delta < 0 else "worse"
    if delta < 0:
        n_improved += 1
    print(f"  Fold {i+1}: 12={f12:.4f}  11_B={f11:.4f}  delta={delta:+.4f}  [{improved}]", flush=True)
print(f"  Folds improved: {n_improved}/5", flush=True)

# Missing-OFF breakdown
y_arr = y_train.values
oof_arr_12  = oof_df.set_index("Index")["oof"].reindex(X_train_raw.index).values
oof_arr_11b = oof_11b.set_index("Index")["oof"].reindex(X_train_raw.index).values
off_missing = X_train_raw["off"].isna().values
print(f"\n=== OFF-missing RMSE ===", flush=True)
print(f"  12 (OFF missing): {root_mean_squared_error(y_arr[off_missing], oof_arr_12[off_missing]):.4f}", flush=True)
print(f"  11_B (OFF missing): {root_mean_squared_error(y_arr[off_missing], oof_arr_11b[off_missing]):.4f}", flush=True)
print(f"  12 (OFF present): {root_mean_squared_error(y_arr[~off_missing], oof_arr_12[~off_missing]):.4f}", flush=True)
print(f"  11_B (OFF present): {root_mean_squared_error(y_arr[~off_missing], oof_arr_11b[~off_missing]):.4f}", flush=True)

print(f"\n=== Decision ===", flush=True)
keep = (rmse_mean < BENCHMARK_RMSE) and (n_improved >= 4)
print(f"  12 mean RMSE  : {rmse_mean:.4f}  (benchmark {BENCHMARK_RMSE})", flush=True)
print(f"  Folds improved: {n_improved}/5  (need >= 4)", flush=True)
print(f"  Decision      : {'KEEP' if keep else 'DISCARD'}", flush=True)

if not keep:
    print("\nExperiment 12 does not meet the keep criterion. No submission written.", flush=True)
    import sys; sys.exit(0)

# =============================================================================
# Keep path: skore report + submission
# =============================================================================

print("\nBuilding skore holdout report...", flush=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
hub_tr_idx, hub_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

X_raw_hub_tr  = X_train_raw.iloc[hub_tr_idx]
X_raw_hub_val = X_train_raw.iloc[hub_val_idx]
y_hub_tr      = y_train.iloc[hub_tr_idx]
y_hub_val     = y_train.iloc[hub_val_idx]
pids_hub_tr   = groups_train.iloc[hub_tr_idx].values
pids_hub_val  = groups_train.iloc[hub_val_idx].values

X_li_hub = ledd_interp_tr.transform(_add_tsd(X_raw_hub_tr))
pcp_hub = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)
ml_hub = MixedLine(pcp=pcp_hub)
ml_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)

X_hub_base_tr = ledd_interp_tr.transform(_add_tsd(X_raw_hub_tr))
X_hub_base_tr = pcp_hub.transform(X_hub_base_tr)
X_hub_ml_tr   = ml_hub.transform(X_hub_base_tr, patient_ids=pids_hub_tr)
X_hub_ml_tr   = add_patient_features(X_hub_ml_tr, ledd_col="ledd_interp")
X_hub_ml_tr["cohort"] = X_hub_ml_tr["cohort"].astype("category")
X_hub_ml_tr["gene"]   = X_hub_ml_tr["gene"].astype("category")
us_hub = UpperStats(); us_hub.fit(X_hub_ml_tr, y_hub_tr)

iot_hub = ImpliedOffTransformer(); iot_hub.fit(X_hub_base_tr)

X_hub_tr  = _build_features_12(X_raw_hub_tr,  pids_hub_tr,  pcp_hub, ml_hub, us_hub, iot_hub)
X_hub_val = _build_features_12(X_raw_hub_val, pids_hub_val, pcp_hub, ml_hub, us_hub, iot_hub)

est_hub = PatientSmoothedRegressor(
    base=HistGradientBoostingRegressor(**HGB_BEST_08),
    alpha=BEST_ALPHA_08,
)
rep = EstimatorReport(
    est_hub,
    X_train=X_hub_tr, y_train=y_hub_tr,
    X_test=X_hub_val,  y_test=y_hub_val,
)
project.put(EXPERIMENT_KEY, rep)
HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Pushed {EXPERIMENT_KEY}: {HUB_URL}", flush=True)

# Refit on all training data
print("Refitting on all training data...", flush=True)
X_all_base = ledd_interp_tr.transform(_add_tsd(X_train_raw))
pids_all   = groups_train.values

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

X_all_final = _build_features_12(X_train_raw, pids_all, pcp_all, ml_all, us_all, iot_all)

est_final = PatientSmoothedRegressor(
    base=HistGradientBoostingRegressor(**HGB_BEST_08),
    alpha=BEST_ALPHA_08,
)
est_final.fit(X_all_final, y_train)

# Predict test set
print("Predicting test set...", flush=True)
pids_test = X_test_raw["patient_id"].values
X_test_base = ledd_interp_tr.transform(_add_tsd(X_test_raw))
X_test_base = pcp_all.transform(X_test_base)
X_test_feat = _build_features_12(X_test_raw, pids_test, pcp_all, ml_all, us_all, iot_all)

test_preds = est_final.predict(X_test_feat)

# Validate and write submission
assert X_test_raw.index.equals(pd.Index(sample_sub["Index"])), "Index mismatch vs sample_submission"
assert len(test_preds) == 11013, f"Expected 11013 predictions, got {len(test_preds)}"
assert np.all(np.isfinite(test_preds)), "Non-finite values in predictions"

sub_dir = PROJECT_ROOT / "submissions"
sub_dir.mkdir(exist_ok=True)
sub_path = sub_dir / "12_improve.csv"
sub_df = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
sub_df.to_csv(sub_path, index=False)
print(f"Submission written -> submissions/12_improve.csv", flush=True)
print(f"Report URL: {HUB_URL}", flush=True)
