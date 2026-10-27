# %% [markdown]
# # Experiment 11 — leakage_fix
#
# **Key:** 11_leakage_fix  |  **Date:** 2026-10-09  |  **Kept:** yes (variant B; variant A scored 3.7199), replaced by 12
# **Idea:** Leakage-free pipeline with all supervised steps inside outer CV folds.
# **CV (GroupKFold 5):** 3.6472 ± 0.0789 (variant B)
# Corrects the off_corr/on_corr bug in UpperStats; honest CV confirms the 10_upper gain.
# **Holdout (20 % of patients):** variant A 3.77, variant B 3.65 (Skore reports).
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/60859 and https://skore.probabl.ai/sps/bobathon-esilv/estimators/60860 (variants A and B).
# The Kaggle submission of variant B is written by 11_submission.py.
# **Run:** `python experiments/11_leakage_fix.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/11_leakage_fix.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see README, How to run.
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
EXPERIMENT_KEY = "11_leakage_fix"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SPLITS_CV = 2 if SMOKE else 5

# %% — skore hub (skipped in smoke mode)
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

# ══════════════════════════════════════════════════════════════════════════════
# Shared primitives (copied verbatim from 10_upper — no changes)
# ══════════════════════════════════════════════════════════════════════════════

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
    """Visit-level bias correction (cross-fitted on training fold).

    NEW in exp 11: transform() writes 'off_corr' and 'on_corr' columns
    containing the per-visit corrected scores so UpperStats can use them.
    """

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
        # ── FIX (exp 11): expose per-visit corrected scores for UpperStats ──
        df["off_corr"] = off_corr
        df["on_corr"]  = on_corr
        # ────────────────────────────────────────────────────────────────────
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
    """Upper-half detrended statistics for OFF, ON, and off_corr/on_corr scores.

    In exp 11 the 'off_corr' and 'on_corr' columns are correctly written by
    PatientCorrectedPooling.transform(), so this class no longer falls back to
    raw 'off'.  The fallback is retained for robustness but should never fire.
    """

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


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Strictly nested feature pipeline (fitted per fold, no pre-materialisation)
# ══════════════════════════════════════════════════════════════════════════════

HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)
BEST_ALPHA_08 = 0.3

ledd_interp_tr = LeddInterp()   # stateless — shared safely


def _build_supervised_features(
    df_raw: pd.DataFrame,
    y_fold,
    pids: np.ndarray,
    pcp: PatientCorrectedPooling,
    ml: MixedLine,
    upper_stats: UpperStats | None = None,
) -> pd.DataFrame:
    """Apply the full supervised feature stack to df_raw given fitted pcp/ml.

    upper_stats: if provided, apply UpperStats transform (must already be fitted).
    """
    df = ledd_interp_tr.transform(_add_tsd(df_raw))
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df, ledd_col="ledd_interp")
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    if upper_stats is not None:
        df = upper_stats.transform(df)
    return df


def _verify_off_corr_distinct(X_tr: pd.DataFrame, X_val: pd.DataFrame, fold_idx: int):
    """Assert off_corr_up_* features differ from off_up_* (not duplicates)."""
    for df, label in [(X_tr, "train"), (X_val, "val")]:
        if "off_corr_up_top1" not in df.columns or "off_up_top1" not in df.columns:
            continue
        off_up   = df["off_up_top1"].dropna().values
        off_c_up = df["off_corr_up_top1"].dropna().values
        n_common = min(len(off_up), len(off_c_up))
        if n_common == 0:
            continue
        max_diff = np.abs(off_up[:n_common] - off_c_up[:n_common]).max()
        assert max_diff > 1e-9, (
            f"Fold {fold_idx} {label}: off_corr_up_top1 == off_up_top1 — "
            "off_corr column is still falling back to raw off"
        )


def manual_cv_rmse_nested(variant: str, y=None):
    """Strictly nested GroupKFold CV.

    variant 'A': leakage-free but no off_corr column (matches 10_upper intent)
    variant 'B': leakage-free + off_corr/on_corr propagated into UpperStats
    """
    if y is None:
        y = y_train
    use_upper = True   # both variants use UpperStats; distinction is off_corr column

    gkf = GroupKFold(n_splits=N_SPLITS_CV)
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

        # ── fit supervised transformers on train fold only ──────────────────
        X_li_tr = ledd_interp_tr.transform(_add_tsd(X_raw_tr))

        pcp = PatientCorrectedPooling(n_splits=N_SPLITS_CV, use_ledd_interp=True)
        pcp.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        ml = MixedLine(pcp=pcp)
        ml.fit(X_li_tr, y_tr, patient_ids=pids_tr)

        # ── build features ──────────────────────────────────────────────────
        # For variant A: suppress off_corr column by using a wrapper that strips it
        if variant == "A":
            # Patch PCP so transform does not write off_corr/on_corr (mimics 10_upper)
            _orig_transform = pcp.transform

            def _transform_no_corr(df_raw, patient_ids=None, _orig=_orig_transform):
                df = _orig(df_raw, patient_ids=patient_ids)
                df = df.drop(columns=["off_corr", "on_corr"], errors="ignore")
                return df

            pcp.transform = _transform_no_corr

        X_tr_b  = _build_supervised_features(X_raw_tr,  y_tr, pids_tr,  pcp, ml)
        X_val_b = _build_supervised_features(X_raw_val, None, pids_val, pcp, ml)

        # UpperStats fitted on train features only
        us = UpperStats()
        us.fit(X_tr_b, y_tr)
        X_tr  = us.transform(X_tr_b)
        X_val = us.transform(X_val_b)

        # ── verify off_corr columns are genuinely distinct (variant B only) ─
        if variant == "B":
            _verify_off_corr_distinct(X_tr, X_val, fold_i)

        # ── fit estimator ────────────────────────────────────────────────────
        est_proto = PatientSmoothedRegressor(
            base=HistGradientBoostingRegressor(**HGB_BEST_08),
            alpha=BEST_ALPHA_08,
        )
        est = clone(est_proto)
        est.fit(X_tr, y_tr)
        preds_val = est.predict(X_val)
        oof[val_idx] = preds_val

        fold_rmse = root_mean_squared_error(y.iloc[val_idx], preds_val)
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_i+1}/{N_SPLITS_CV}: RMSE = {fold_rmse:.4f}", flush=True)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse  = float(np.std(fold_rmses, ddof=1))
    return mean_rmse, std_rmse, oof, fold_rmses


# ══════════════════════════════════════════════════════════════════════════════
# Run CV
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n=== Variant A (leakage-free, no off_corr propagation) ===", flush=True)
rmse_A_mean, rmse_A_std, oof_A, folds_A = manual_cv_rmse_nested("A")
print(f"  RMSE: {rmse_A_mean:.4f} ± {rmse_A_std:.4f}", flush=True)

print(f"\n=== Variant B (leakage-free + off_corr/on_corr in UpperStats) ===", flush=True)
rmse_B_mean, rmse_B_std, oof_B, folds_B = manual_cv_rmse_nested("B")
print(f"  RMSE: {rmse_B_mean:.4f} ± {rmse_B_std:.4f}", flush=True)

if SMOKE:
    print("\n[SMOKE] All checks passed — exiting.", flush=True)
    import sys; sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
# Summary and leakage comparison
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_10 = 3.7408
print("\n=== Summary ===", flush=True)
print(f"  10_upper C (snooped)  : 3.7408 ± 0.0519  [reference]", flush=True)
print(f"  11_A (valid, no fix)  : {rmse_A_mean:.4f} ± {rmse_A_std:.4f}", flush=True)
print(f"  11_B (valid + off_corr): {rmse_B_mean:.4f} ± {rmse_B_std:.4f}", flush=True)
print(f"  A vs B delta          : {rmse_A_mean - rmse_B_mean:+.4f}", flush=True)
print(f"  10 vs A delta (leakage): {BASELINE_10 - rmse_A_mean:+.4f}", flush=True)
print(f"\n  Per-fold RMSEs:", flush=True)
for i, (fa, fb) in enumerate(zip(folds_A, folds_B)):
    print(f"    Fold {i+1}: A={fa:.4f}  B={fb:.4f}  delta={fa-fb:+.4f}", flush=True)

best_variant = "A" if rmse_A_mean <= rmse_B_mean else "B"
best_mean    = rmse_A_mean if best_variant == "A" else rmse_B_mean
best_std     = rmse_A_std  if best_variant == "A" else rmse_B_std
oof_best     = oof_A       if best_variant == "A" else oof_B
print(f"\nBest variant: {best_variant} → RMSE {best_mean:.4f} ± {best_std:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Push skore reports (GroupShuffleSplit holdout — strictly valid)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding skore holdout reports...", flush=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
hub_tr_idx, hub_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

X_raw_hub_tr  = X_train_raw.iloc[hub_tr_idx]
X_raw_hub_val = X_train_raw.iloc[hub_val_idx]
y_hub_tr      = y_train.iloc[hub_tr_idx]
y_hub_val     = y_train.iloc[hub_val_idx]
pids_hub_tr   = groups_train.iloc[hub_tr_idx].values
pids_hub_val  = groups_train.iloc[hub_val_idx].values

# Fit supervised transformers on hub train split
X_li_hub_tr = ledd_interp_tr.transform(_add_tsd(X_raw_hub_tr))
pcp_hub = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_hub.fit(X_li_hub_tr, y_hub_tr, patient_ids=pids_hub_tr)
ml_hub = MixedLine(pcp=pcp_hub)
ml_hub.fit(X_li_hub_tr, y_hub_tr, patient_ids=pids_hub_tr)

# Variant A hub features (no off_corr)
pcp_hub_a = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_hub_a.fit(X_li_hub_tr, y_hub_tr, patient_ids=pids_hub_tr)
ml_hub_a  = MixedLine(pcp=pcp_hub_a)
ml_hub_a.fit(X_li_hub_tr, y_hub_tr, patient_ids=pids_hub_tr)
_orig_a = pcp_hub_a.transform
def _transform_hub_a(df_raw, patient_ids=None, _orig=_orig_a):
    df = _orig(df_raw, patient_ids=patient_ids)
    return df.drop(columns=["off_corr", "on_corr"], errors="ignore")
pcp_hub_a.transform = _transform_hub_a

X_hub_tr_b_a  = _build_supervised_features(X_raw_hub_tr,  y_hub_tr, pids_hub_tr,  pcp_hub_a, ml_hub_a)
X_hub_val_b_a = _build_supervised_features(X_raw_hub_val, None,     pids_hub_val, pcp_hub_a, ml_hub_a)
us_hub_a = UpperStats(); us_hub_a.fit(X_hub_tr_b_a, y_hub_tr)
X_hub_tr_A  = us_hub_a.transform(X_hub_tr_b_a)
X_hub_val_A = us_hub_a.transform(X_hub_val_b_a)

# Variant B hub features (with off_corr)
X_hub_tr_b_b  = _build_supervised_features(X_raw_hub_tr,  y_hub_tr, pids_hub_tr,  pcp_hub, ml_hub)
X_hub_val_b_b = _build_supervised_features(X_raw_hub_val, None,     pids_hub_val, pcp_hub, ml_hub)
us_hub_b = UpperStats(); us_hub_b.fit(X_hub_tr_b_b, y_hub_tr)
X_hub_tr_B  = us_hub_b.transform(X_hub_tr_b_b)
X_hub_val_B = us_hub_b.transform(X_hub_val_b_b)

est_proto_final = PatientSmoothedRegressor(
    base=HistGradientBoostingRegressor(**HGB_BEST_08),
    alpha=BEST_ALPHA_08,
)

rep_A = EstimatorReport(
    clone(est_proto_final),
    X_train=X_hub_tr_A, y_train=y_hub_tr,
    X_test=X_hub_val_A,  y_test=y_hub_val,
)
rep_B = EstimatorReport(
    clone(est_proto_final),
    X_train=X_hub_tr_B, y_train=y_hub_tr,
    X_test=X_hub_val_B,  y_test=y_hub_val,
)

project.put(f"{EXPERIMENT_KEY}_A", rep_A)
project.put(f"{EXPERIMENT_KEY}_B", rep_B)
HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Pushed 11_leakage_fix_A and 11_leakage_fix_B: {HUB_URL}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Save OOF predictions
# ══════════════════════════════════════════════════════════════════════════════

audit_dir = PROJECT_ROOT / "audit"
audit_dir.mkdir(exist_ok=True)

for variant_tag, oof_arr in [("A", oof_A), ("B", oof_B), ("best", oof_best)]:
    oof_df = pd.DataFrame({
        "Index":      X_train_raw.index,
        "patient_id": groups_train.values,
        "oof":        oof_arr,
    })
    oof_path = audit_dir / f"oof_11_{variant_tag}.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF saved → audit/oof_11_{variant_tag}.csv", flush=True)

print(f"\nDone. Best variant: {best_variant} ({best_mean:.4f} ± {best_std:.4f})", flush=True)
