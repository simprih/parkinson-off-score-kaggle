# %% [markdown]
# # Experiment 14 — evidence
#
# **Date:** 2026-10-11
# **Goal:** Four targeted improvements on top of 13_twostage:
#   D. Four extra first-stage models (simple, visit, off_evidence, on_evidence).
#   E. Stage-1 blend: r = 0.8 * base + 0.2 * simple; then quadratic projection q.
#   F. Stage-2 features: 21 from 13 + 15 evidence features (visit, off_ev, on_ev).
#   G. Stage-2 ensemble: 5 HGB models each on 80% random patient subsample.
#   H. Post-processing: second quadratic projection, isotonic non-decreasing fit, clip>=0.
#
# Benchmark: Experiment 13 CV RMSE 3.2370 +/- 0.0792 (GroupKFold 5).
# Keep criterion: mean RMSE < 3.2370 AND >= 4 of 5 folds improve vs 13.
#
# **Key:** 14_evidence  |  **Decision:** kept, better than 13 on 5/5 folds; replaced by 18 (15 to 17 were rejected)
# **Results:** CV (GroupKFold 5) 3.1453 ± 0.0784  |  holdout (20 % of patients) 3.17
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/61566
# **Run:** `python experiments/14_evidence.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/14_evidence.py`).
# Reads data/ (first version of the files, before 9 October) and audit/oof_13.csv; needs the
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "14_evidence"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SPLITS_CV = 2 if SMOKE else 5

BENCHMARK_RMSE  = 3.2370
BENCHMARK_FOLDS = [3.2171, 3.1405, 3.3384, 3.1945, 3.2946]  # 13_twostage per-fold

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
# Shared primitives (verbatim from 13_twostage)
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
# ImpliedOffTransformer (unchanged from 12)
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
# A: DiagnosisImputer (verbatim from 13)
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
# B: quadratic_projection helper (verbatim from 13)
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
# D: Evidence model parameters (shared HGB params for simple/visit/evidence)
# =============================================================================

_EV_HGB_PARAMS = dict(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=20, random_state=0,
)

# Features for each evidence model (raw ledd, cohort+gene as categorical)
_SIMPLE_FEATS = [
    "cohort", "gene", "sexM", "age_at_diagnosis", "age", "time_since_diagnosis",
    "ledd", "time_since_intake_on", "time_since_intake_off", "on", "off",
    # per-patient aggregates
    "off_pt_mean", "off_pt_max", "off_pt_count",
    "on_pt_mean", "on_pt_max", "on_pt_count",
    "ledd_pt_mean", "ledd_pt_max", "ledd_pt_count",
    # centered age and n_visits
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


def _encode_cats(df: pd.DataFrame, cat_cols: list) -> np.ndarray:
    """Return numpy array with categorical columns label-encoded as float."""
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(float)
    return out.values.astype(float)


def _add_simple_aggs(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient max and count of off/on/ledd, and age_tc, for simple model."""
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
# NEW C/D: TwoStageRegressor with evidence models
# =============================================================================

class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Two-stage regressor with evidence models (Experiment 14).

    Stage 1: HGB base + simple blend (0.8/0.2); plus visit, off_evidence, on_evidence.
    Stage 2: 5 HGB ensemble on 80% patient subsamples, evidence features added.
    Post-processing: second quadratic projection, isotonic non-decreasing, clip>=0.
    """

    def __init__(self, base_params: dict, n_splits_inner: int = 5):
        self.base_params = base_params
        self.n_splits_inner = n_splits_inner

    def _make_base(self):
        return HistGradientBoostingRegressor(**self.base_params)

    def _make_ev_model(self):
        return HistGradientBoostingRegressor(**_EV_HGB_PARAMS)

    @staticmethod
    def _stage2_features(
        X: pd.DataFrame,
        r: np.ndarray,
        q: np.ndarray,
        ev_visit: np.ndarray,
        ev_off: np.ndarray,
        ev_on: np.ndarray,
    ) -> np.ndarray:
        """Build 36-column stage-2 feature matrix (21 from 13 + 15 evidence)."""
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

        # Evidence delta features (F): e - q and per-patient stats
        ev_sources = [("visit", ev_visit), ("off_ev", ev_off), ("on_ev", ev_on)]
        ev_delta      = {name: e - q for name, e in ev_sources}
        ev_max_eq     = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean_eq    = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean2_eq   = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_min_eq     = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}

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

            # Per-patient evidence delta stats
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

        # Base 21 features (same as 13)
        base_feats = np.column_stack([
            feat_q, feat_q_dev, feat_r_minus_q,
            feat_off_minus_q, feat_on_minus_q,
            feat_max_off_q, feat_mean_off_q, feat_mean2_off_q,
            feat_mean_on_q, feat_max_on_q, feat_std_r_q,
            feat_n_off, feat_n_on, feat_tc, feat_rel_pos, feat_n_visits,
            feat_off_dev, feat_on_dev, feat_ledd_dev,
            feat_tsi_on, feat_tsi_off,
        ])

        # 15 evidence features: for each of visit, off_ev, on_ev: (e-q), max, mean, mean2, min
        ev_feats_list = []
        for name in ["visit", "off_ev", "on_ev"]:
            ev_feats_list.append(ev_delta[name])          # e - q
            ev_feats_list.append(ev_max_eq[name])         # per-patient max(e-q)
            ev_feats_list.append(ev_mean_eq[name])        # per-patient mean(e-q)
            ev_feats_list.append(ev_mean2_eq[name])       # per-patient mean top-2(e-q)
            ev_feats_list.append(ev_min_eq[name])         # per-patient min(e-q)

        ev_feats = np.column_stack(ev_feats_list)
        return np.column_stack([base_feats, ev_feats])

    def _build_simple_X(self, X: pd.DataFrame) -> np.ndarray:
        """Build feature matrix for the simple model."""
        df = _add_simple_aggs(X)
        # off_pt_mean already added by add_patient_features; pick only what's needed
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

    def _predict_evidence(self, X: pd.DataFrame) -> tuple:
        """Return (ev_visit, ev_off, ev_on) predictions; NaN where evidence missing."""
        ev_visit = self.model_visit_.predict(self._build_visit_X(X))

        off_col = X["off"].values.astype(float) if "off" in X.columns else np.full(len(X), np.nan)
        ev_off = np.full(len(X), np.nan)
        has_off = ~np.isnan(off_col)
        if has_off.any():
            X_off = self._build_off_ev_X(X)
            preds_off = self.model_off_ev_.predict(X_off)
            ev_off[has_off] = preds_off[has_off]

        on_col = X["on"].values.astype(float) if "on" in X.columns else np.full(len(X), np.nan)
        ev_on = np.full(len(X), np.nan)
        has_on = ~np.isnan(on_col)
        if has_on.any():
            X_on = self._build_on_ev_X(X)
            preds_on = self.model_on_ev_.predict(X_on)
            ev_on[has_on] = preds_on[has_on]

        return ev_visit, ev_off, ev_on

    def fit(self, X: pd.DataFrame, y):
        y_arr = np.asarray(y, dtype=float)
        pids  = X["patient_id"].values
        ages  = X["age"].values.astype(float)

        # Inner GroupKFold for OOF raw predictions of all 5 first-stage models
        gkf = GroupKFold(n_splits=self.n_splits_inner)
        r_oof       = np.full(len(y_arr), np.nan)  # base
        simp_oof    = np.full(len(y_arr), np.nan)  # simple
        visit_oof   = np.full(len(y_arr), np.nan)  # visit
        off_ev_oof  = np.full(len(y_arr), np.nan)  # off_evidence (NaN where off missing)
        on_ev_oof   = np.full(len(y_arr), np.nan)  # on_evidence  (NaN where on missing)

        for tr_idx, val_idx in gkf.split(X, y_arr, groups=pids):
            X_tr  = X.iloc[tr_idx]
            X_val = X.iloc[val_idx]
            y_tr  = y_arr[tr_idx]

            # Base model
            m_base = self._make_base()
            m_base.fit(X_tr.drop(columns=["patient_id"]), y_tr)
            r_oof[val_idx] = m_base.predict(X_val.drop(columns=["patient_id"]))

            # Simple model
            m_simp = self._make_ev_model()
            X_simp_tr  = self._build_simple_X(X_tr)
            X_simp_val = self._build_simple_X(X_val)
            m_simp.fit(X_simp_tr, y_tr)
            simp_oof[val_idx] = m_simp.predict(X_simp_val)

            # Visit model
            m_visit = self._make_ev_model()
            X_visit_tr  = self._build_visit_X(X_tr)
            X_visit_val = self._build_visit_X(X_val)
            m_visit.fit(X_visit_tr, y_tr)
            visit_oof[val_idx] = m_visit.predict(X_visit_val)

            # off_evidence model (train on rows with off present)
            off_tr = X_tr["off"].values.astype(float) if "off" in X_tr.columns else np.full(len(X_tr), np.nan)
            has_off_tr = ~np.isnan(off_tr)
            m_off_ev = self._make_ev_model()
            if has_off_tr.sum() >= 10:
                X_off_tr = self._build_off_ev_X(X_tr)
                m_off_ev.fit(X_off_tr[has_off_tr], y_tr[has_off_tr])
                off_val = X_val["off"].values.astype(float) if "off" in X_val.columns else np.full(len(X_val), np.nan)
                has_off_val = ~np.isnan(off_val)
                if has_off_val.any():
                    X_off_val = self._build_off_ev_X(X_val)
                    preds_off = m_off_ev.predict(X_off_val)
                    off_ev_oof[val_idx[has_off_val]] = preds_off[has_off_val]

            # on_evidence model (train on rows with on present)
            on_tr = X_tr["on"].values.astype(float) if "on" in X_tr.columns else np.full(len(X_tr), np.nan)
            has_on_tr = ~np.isnan(on_tr)
            m_on_ev = self._make_ev_model()
            if has_on_tr.sum() >= 10:
                X_on_tr = self._build_on_ev_X(X_tr)
                m_on_ev.fit(X_on_tr[has_on_tr], y_tr[has_on_tr])
                on_val = X_val["on"].values.astype(float) if "on" in X_val.columns else np.full(len(X_val), np.nan)
                has_on_val = ~np.isnan(on_val)
                if has_on_val.any():
                    X_on_val = self._build_on_ev_X(X_val)
                    preds_on = m_on_ev.predict(X_on_val)
                    on_ev_oof[val_idx[has_on_val]] = preds_on[has_on_val]

        # E: blend r = 0.8 * base + 0.2 * simple, then quadratic projection q
        r_blend = 0.8 * r_oof + 0.2 * simp_oof
        q_oof = quadratic_projection(pids, ages, r_blend)

        # Stage-2 features and residuals
        s2_feats = self._stage2_features(X, r_blend, q_oof, visit_oof, off_ev_oof, on_ev_oof)
        residuals = y_arr - q_oof

        # G: 5 stage-2 models, each on 80% random patient subsample
        unique_pids = np.unique(pids)
        self.stage2_models_ = []
        for seed in range(5):
            rng_sub = np.random.default_rng(seed)
            n_sub = max(1, int(0.8 * len(unique_pids)))
            sub_pids = set(rng_sub.choice(unique_pids, size=n_sub, replace=False))
            sub_mask = np.array([pid in sub_pids for pid in pids])
            valid_mask = sub_mask & np.isfinite(q_oof) & np.isfinite(residuals)
            m_s2 = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                min_samples_leaf=50, random_state=seed,
            )
            m_s2.fit(s2_feats[valid_mask], residuals[valid_mask])
            self.stage2_models_.append(m_s2)

        # Refit all five first-stage models on ALL training data
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

        return self

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict without isotonic fit or clipping (for v4_raw)."""
        pids = X["patient_id"].values
        ages = X["age"].values.astype(float)

        # E: blend r = 0.8 * base + 0.2 * simple
        r_base  = self.base_.predict(X.drop(columns=["patient_id"]))
        r_simp  = self.model_simple_.predict(self._build_simple_X(X))
        r = 0.8 * r_base + 0.2 * r_simp

        # Quadratic projection
        q = quadratic_projection(pids, ages, r)

        # Evidence predictions
        ev_visit, ev_off, ev_on = self._predict_evidence(X)

        # Stage-2 features
        s2_feats = self._stage2_features(X, r, q, ev_visit, ev_off, ev_on)

        # G: average 5 stage-2 corrections
        delta = np.mean(
            np.column_stack([m.predict(s2_feats) for m in self.stage2_models_]),
            axis=1,
        )
        p = q + delta

        # Final quadratic projection (H step 1)
        return quadratic_projection(pids, ages, p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Full predict: raw + isotonic non-decreasing per patient + clip>=0 (H)."""
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
            # Write back into original positions
            valid_idx = idx[valid]
            sorted_idx = valid_idx[order]
            output[sorted_idx] = iso_pred

        output = np.clip(output, 0, None)
        return output


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
# Feature pipeline (same as 13)
# =============================================================================

HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)

ledd_interp_tr = LeddInterp()   # stateless


def _build_features_13(
    df_raw: pd.DataFrame,
    pids: np.ndarray,
    diag_imp: DiagnosisImputer,
    pcp: PatientCorrectedPooling,
    ml: MixedLine,
    upper_stats: UpperStats,
    implied_off_tr: ImpliedOffTransformer,
) -> pd.DataFrame:
    """Full supervised feature stack (same as Experiment 13)."""
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
# CV loop
# =============================================================================

def run_cv_14(y=None):
    if y is None:
        y = y_train
    gkf  = GroupKFold(n_splits=N_SPLITS_CV)
    oof_v4_raw = np.full(len(y), np.nan)
    oof_v4     = np.full(len(y), np.nan)
    fold_rmses_raw = []
    fold_rmses_v4  = []

    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf.split(X_train_raw, y, groups=groups_train)
    ):
        X_raw_tr  = X_train_raw.iloc[tr_idx]
        X_raw_val = X_train_raw.iloc[val_idx]
        y_tr      = y.iloc[tr_idx]
        pids_tr   = groups_train.iloc[tr_idx].values
        pids_val  = groups_train.iloc[val_idx].values

        # A: fit DiagnosisImputer on train fold
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

        X_tr  = _build_features_13(X_raw_tr,  pids_tr,  diag_imp, pcp, ml, us, implied_off_tr_fold)
        X_val = _build_features_13(X_raw_val, pids_val, diag_imp, pcp, ml, us, implied_off_tr_fold)

        est = TwoStageRegressor(base_params=HGB_BEST_08, n_splits_inner=N_SPLITS_CV)
        est.fit(X_tr, y_tr)

        # Two validation predictions from the same fitted model
        v4_raw_val = est._predict_raw(X_val)
        v4_val     = est.predict(X_val)

        oof_v4_raw[val_idx] = v4_raw_val
        oof_v4[val_idx]     = v4_val

        rmse_raw = root_mean_squared_error(y.iloc[val_idx], v4_raw_val)
        rmse_v4  = root_mean_squared_error(y.iloc[val_idx], v4_val)
        fold_rmses_raw.append(rmse_raw)
        fold_rmses_v4.append(rmse_v4)
        print(
            f"  Fold {fold_i+1}/{N_SPLITS_CV}: "
            f"v4_raw={rmse_raw:.4f}  v4={rmse_v4:.4f}",
            flush=True,
        )

    results = {
        "v4_raw": {"oof": oof_v4_raw, "folds": fold_rmses_raw,
                   "mean": float(np.mean(fold_rmses_raw)), "std": float(np.std(fold_rmses_raw, ddof=1))},
        "v4":     {"oof": oof_v4,     "folds": fold_rmses_v4,
                   "mean": float(np.mean(fold_rmses_v4)),  "std": float(np.std(fold_rmses_v4, ddof=1))},
    }
    return results


# =============================================================================
# Run
# =============================================================================

print("\n=== Experiment 14 CV (GroupKFold 5) ===", flush=True)
results = run_cv_14()

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
    "v4_raw":     results["v4_raw"]["oof"],
    "v4":         results["v4"]["oof"],
})
oof_path = audit_dir / "oof_14.csv"
oof_df.to_csv(oof_path, index=False)
print(f"OOF saved -> audit/oof_14.csv", flush=True)

# =============================================================================
# Print results vs Experiment 13
# =============================================================================

oof_13 = pd.read_csv(audit_dir / "oof_13.csv").set_index("Index")

print("\n=== Results vs Experiment 13 ===", flush=True)
gkf5 = GroupKFold(n_splits=5)
fold_v3 = []
for fold_i, (tr_idx, val_idx) in enumerate(
    gkf5.split(X_train_raw, y_train, groups=groups_train)
):
    y_val   = y_train.iloc[val_idx]
    v3_val  = oof_13.reindex(X_train_raw.index[val_idx])["v3"].values
    fold_v3.append(root_mean_squared_error(y_val, v3_val))

for tag in ["v4_raw", "v4"]:
    r = results[tag]
    print(f"\n  {tag}: mean={r['mean']:.4f} +/- {r['std']:.4f}", flush=True)
    n_improved = 0
    for i, (fv, f13) in enumerate(zip(r["folds"], fold_v3)):
        delta = fv - f13
        improved = "BETTER" if delta < 0 else "worse"
        if delta < 0:
            n_improved += 1
        print(f"    Fold {i+1}: {tag}={fv:.4f}  13={f13:.4f}  delta={delta:+.4f}  [{improved}]", flush=True)
    print(f"    Folds improved vs 13: {n_improved}/5", flush=True)
    r["n_improved"] = n_improved

# =============================================================================
# Decision
# =============================================================================

print("\n=== Decision ===", flush=True)
r = results["v4"]
qualifies = (r["mean"] < BENCHMARK_RMSE) and (r["n_improved"] >= 4)
print(
    f"  v4: mean={r['mean']:.4f}  improved={r['n_improved']}/5  "
    f"-> {'KEEP' if qualifies else 'discard'}",
    flush=True,
)

if not qualifies:
    print("\nv4 does not meet the keep criterion. Stopping.", flush=True)
    import sys; sys.exit(0)

print(f"\nKeeping v4 as {EXPERIMENT_KEY}.", flush=True)

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

X_hub_tr  = _build_features_13(X_raw_hub_tr,  pids_hub_tr,  diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)
X_hub_val = _build_features_13(X_raw_hub_val, pids_hub_val, diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)

est_hub = TwoStageRegressor(base_params=HGB_BEST_08)
rep = EstimatorReport(
    est_hub,
    X_train=X_hub_tr, y_train=y_hub_tr,
    X_test=X_hub_val, y_test=y_hub_val,
)
project.put(EXPERIMENT_KEY, rep)
HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"
print(f"Pushed {EXPERIMENT_KEY}: {HUB_URL}", flush=True)

# Refit on all training data
print("Refitting on all training data...", flush=True)
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

X_all_final = _build_features_13(X_train_raw, pids_all, diag_imp_all, pcp_all, ml_all, us_all, iot_all)

est_final = TwoStageRegressor(base_params=HGB_BEST_08)
est_final.fit(X_all_final, y_train)

# Predict test set
print("Predicting test set...", flush=True)
pids_test = X_test_raw["patient_id"].values
X_test_feat = _build_features_13(X_test_raw, pids_test, diag_imp_all, pcp_all, ml_all, us_all, iot_all)

test_preds = est_final.predict(X_test_feat)

# Validate and write submission
assert X_test_raw.index.equals(pd.Index(sample_sub["Index"])), "Index mismatch vs sample_submission"
assert len(test_preds) == 11013, f"Expected 11013 predictions, got {len(test_preds)}"
n_finite = int(np.isfinite(test_preds).sum())
assert n_finite == 11013, f"Expected 11013 finite predictions, got {n_finite}"
assert np.all(test_preds >= 0), f"Some predictions are negative"

oof_kept = results["v4"]["oof"]
print(f"\nOOF (v4)  stats: mean={np.nanmean(oof_kept):.3f}  std={np.nanstd(oof_kept):.3f}", flush=True)
print(f"Test pred stats : mean={test_preds.mean():.3f}  std={test_preds.std():.3f}", flush=True)

sub_dir = PROJECT_ROOT / "submissions"
sub_dir.mkdir(exist_ok=True)
sub_path = sub_dir / f"{EXPERIMENT_KEY}.csv"
sub_df = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
sub_df.to_csv(sub_path, index=False)
print(f"Submission written -> submissions/{EXPERIMENT_KEY}.csv", flush=True)
print(f"Report URL: {HUB_URL}", flush=True)
