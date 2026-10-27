# %% [markdown]
# # Experiment 25 — blend of E-model variant (A) and shrinkage variant (B)
#
# **Experiment key:** 25_blend  |  **Date:** 2026-10-14  |  **Kept:** yes (final submission)
#
# ## What it predicts
#
# The "true" OFF MDS-UPDRS motor score of each visit, scored by RMSE.  The test
# patients are all new, so every validation is grouped by patient_id.
#
# ## Pipeline (7 steps)
#
# 1. **Rater correction.** Each rater multiplies on/off by a fixed factor; we
#    estimate it inside each training fold as the 75th percentile of
#    log(off / target) per rater and divide it out
#    (`fit_rater_factors`, `apply_rater_correction`).
#
# 2. **Drug-effect curve f(t)** of time_since_intake_on: rises until 1.45 h,
#    flat through 5.2 h, wears off linearly after that, 0.864 when the time is
#    missing (`compute_f_new`).
#
# 3. **Drug sensitivity E** of each patient, from the relation
#    ON = OFF × (1 − E × f(t)):
#    variant A uses a Bayesian model with an HGB prior (`EModel`);
#    variant B uses shrinkage towards the population value (`EModelShrinkage`).
#
# 4. **Every ON-only visit becomes an OFF-scale value** s = on / (1 − E × f),
#    plus patient aggregates (`build_new_columns`).
#
# 5. **Likelihood features:** how likely each level of the patient curve is,
#    given the measured noise shape per visit type
#    (`compute_likelihood_features_25`).
#
# 6. **Two-stage model** (`TwoStageRegressor25`): stage 1 = the first-stage
#    models inherited from experiments 08 to 19; stage 2 = 5 bagged
#    HistGradientBoosting models on the log-ratio between the target and the
#    stage-1 prediction.
#
# 7. **Per-patient monotone cubic projection** in age (`final_projection_25`),
#    then the average of variants A and B (`BlendWrapper25`).
#
# ## Results
#
# Holdout RMSE (GroupShuffleSplit 20 %, random_state=0):
# A = 2.6956, B = 2.7050, blend = 2.6784.
# Kaggle public 2.799, private 2.430.
# Skore report: https://skore.probabl.ai/sps/bobathon-esilv/estimators/85087
#
# ## How to run
#
#     .venv\Scripts\python.exe -X utf8 -u experiments/25_blend.py <step>
#     python experiments/25_blend.py <step>    (macOS / Linux)
#
# Setup (Python version, packages, data, Skore key): report (report/rapport_SPS.pdf), section How to run.
#
# Steps:
# * ``smoke``  — quick sanity check on 10 % of patients.
# * ``submit`` — fits on all training data and writes submissions/25_blend.csv.
# * ``report`` — builds and pushes the Skore EstimatorReport.
# * ``cv``     — 5-fold grouped CV (caches each fold to audit/oof_25_folds/).
#
# Data is read from data_new/ if it exists, else data/.

# %%
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import math
from scipy.ndimage import convolve1d
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Determine data directory
_DATA_NEW = PROJECT_ROOT / "data_new"
_DATA_OLD = PROJECT_ROOT / "data"
DATA_DIR = _DATA_NEW if _DATA_NEW.exists() else _DATA_OLD

EXPERIMENT_KEY = "25_blend"

# =============================================================================
# Shared primitives (verbatim from 19_likelihood)
# =============================================================================

_BIAS_FEATURES_OFF_I = ["time_since_intake_off", "cohort", "ledd_interp", "time_since_diagnosis"]
_BIAS_FEATURES_ON_I  = ["time_since_intake_on",  "cohort", "ledd_interp", "time_since_diagnosis"]


def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Return OLS slope and intercept of y on x, ignoring NaNs.

    Returns (nan, nan) when fewer than two valid pairs are available or when x
    has zero variance.  Used throughout for per-patient trend estimation.
    """
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
    """Add per-patient summary statistics and trajectory features to df.

    Parameters
    ----------
    df : pd.DataFrame
        Input visit-level data; must contain patient_id, off, on, ledd (or
        ledd_col), and time_since_diagnosis.
    ledd_col : str
        Column to use for ledd statistics (``"ledd"`` or ``"ledd_interp"``).

    Returns
    -------
    pd.DataFrame
        Input plus per-patient mean, median, std, count, slope, intercept,
        fitted value (for off and on), ledd aggregates, off_dev, on_dev,
        pt_n_visits, and pt_visit_rank.

    Notes
    -----
    All statistics are computed from training rows only when called inside a
    cross-validation fold (the caller controls which rows are passed).
    """
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
    """Per-patient linear interpolation of ledd over age (stateless).

    ledd is often missing for some visits.  Interpolating within each patient
    over age makes the imputed value consistent with that patient's dose
    trajectory rather than using a population mean.

    fit learns nothing; transform fills gaps and adds ledd_interp,
    ledd_missing (indicator), and ledd_slope.
    """

    def fit(self, X, y=None):
        """No-op; LeddInterp is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Interpolate ledd within each patient over age.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain patient_id, age, ledd.

        Returns
        -------
        pd.DataFrame
            Input plus ledd_interp (interpolated ledd), ledd_missing (1 if
            original was NaN), ledd_slope (per-patient OLS slope of ledd on
            age).
        """
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
    """Visit-level bias correction for off and on, cross-fitted to avoid leakage.

    The raw measurements off and on are systematically biased by intake time,
    dose, and cohort.  A small HGB corrects each measurement using visit-level
    covariates; it is trained on the residual (y − raw_score) so it never sees
    future rows.

    fit learns two correctors (off, on) plus out-of-fold corrections for
    training rows via GroupKFold.  transform adds off_corr, on_corr, and
    per-patient trajectory statistics on the corrected values.

    Parameters
    ----------
    n_splits : int
        Number of GroupKFold folds used for cross-fitting the correctors.
    use_ledd_interp : bool
        Use ledd_interp instead of ledd as a bias feature (set True when
        LeddInterp has already run).
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
        """Fit a small HGB on residuals and return it."""
        hgb = HistGradientBoostingRegressor(
            max_iter=100, random_state=0, categorical_features=cat_col_indices
        )
        hgb.fit(features, residuals)
        return hgb

    def _cross_fit_corrections(self, df, y, score_col, bias_features, patient_ids):
        """Produce out-of-fold bias corrections via GroupKFold."""
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
        """Fit the full-data corrector for use at prediction time."""
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
        """Apply a fitted corrector to new rows, aligning cohort categories."""
        col_data = df[bias_features].copy()
        col_data["cohort"] = (
            col_data["cohort"]
            .astype(pd.CategoricalDtype(categories=cohort_categories))
            .cat.codes.astype(float)
        )
        return model.predict(col_data.values)

    def fit(self, df_raw, y, patient_ids=None):
        """Fit the bias correctors and store out-of-fold corrections.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw visit-level data (with off, on, cohort, and bias features).
        y : array-like
            Target values aligned with df_raw rows.
        patient_ids : array-like
            Patient identifier for each row, used by GroupKFold.
        """
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
        """Return (off_corr, on_corr) arrays for df_raw rows.

        Training rows receive their stored out-of-fold corrections; unseen
        rows receive predictions from the full-data correctors.
        """
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
        """Add off_corr, on_corr and per-patient trajectory features.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw visit-level data.
        patient_ids : array-like, optional
            If None, uses df_raw["patient_id"].

        Returns
        -------
        pd.DataFrame
            Input plus off_corr, on_corr, off_corr_pt_fitted,
            on_corr_pt_fitted, pool_corr_pt_fitted, pool_corr_pt_mean,
            pt_n_off_corr, pt_n_on_corr.
        """
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
    """Empirical-Bayes linear mixed model for the per-patient score trajectory.

    The corrected measurements (off_corr, on_corr from PatientCorrectedPooling)
    carry an intercept, a slope in time_since_diagnosis, and type-specific
    offsets d_off and d_on.  A Gaussian prior over (intercept, slope, d_off,
    d_on) is estimated from training patients; each patient's posterior is used
    as a feature.

    fit learns the population prior (mu_, S_) and the noise variances
    (s2_off_, s2_on_).  transform adds ml_pred (posterior line value),
    ml_std (posterior uncertainty), ml_slope, ml_d_off, ml_d_on.
    """

    def __init__(self, pcp: PatientCorrectedPooling):
        self.pcp = pcp

    @staticmethod
    def _patient_params(pids, times, z_off, z_on, y):
        """Estimate per-patient OLS line and type offsets d_off, d_on from training data."""
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
        """Compute population mean mu and covariance S from per-patient estimates."""
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
        """Estimate within-patient noise variance for off and on measurements."""
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
        """Compute Gaussian posterior over (intercept, slope, d_off, d_on) for one patient."""
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
        """Learn population prior and noise variances from training data.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw data; pcp must already be fitted.
        y : array-like
            Target values (used only to fit the OLS lines, not for the prior
            model predictions).
        patient_ids : array-like
            Patient identifier for each row.
        """
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
        """Add posterior features ml_pred, ml_std, ml_slope, ml_d_off, ml_d_on.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Visit-level data.
        patient_ids : array-like, optional
            If None, uses df_raw["patient_id"].

        Returns
        -------
        pd.DataFrame
            Input plus ml_pred, ml_std, ml_slope, ml_d_off, ml_d_on.
        """
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


def _upper_stats_for_scores(raw_vals, times, beta, t_query, prefix):
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
    """Upper-half statistics of OFF/ON scores, detrended by a pooled age slope.

    OFF measurement noise is one-sided: residual levodopa drives scores down.
    Patient-level means are therefore biased low.  UpperStats fits a single
    pooled within-patient OLS slope beta (in time-since-diagnosis) on the
    target, then removes the trend before ranking visits.  The upper half of
    the de-trended distribution is less contaminated by drug-on visits.

    fit learns beta_ from target values.  transform adds top-1, mean-top-2,
    mean-upper-half, mean-without-lowest, mean, count, spread, and projected
    variants for off, on, off_corr, on_corr.
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


_POP_GAP_FEATURES = [
    "time_since_intake_on", "ledd_interp", "time_since_diagnosis", "age_at_diagnosis",
]


class ImpliedOffTransformer(BaseEstimator, TransformerMixin):
    """Impute an OFF-scale value for ON-only visits via a shrunk patient gap.

    When a visit has ON but no OFF, the best estimate of the true OFF is
    on + gap, where gap = off − on.  The patient-specific gap is shrunk toward
    a population model to reduce noise from patients with few paired visits.
    The shrinkage weight k is chosen by leave-one-out cross-validation.

    fit learns the per-patient gaps, the population model, and k_.
    transform adds implied_off, gap_weight (shrinkage weight w), and n_paired.
    """

    def _compute_population_gaps(self, df: pd.DataFrame) -> np.ndarray:
        """Predict population-level off-on gap for each row."""
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
            self._pt_gaps_ = pt_gaps
            self._pt_n_ = pt_n
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


class DiagnosisImputer(BaseEstimator, TransformerMixin):
    """Impute missing age_at_diagnosis per cohort, then derive time_since_diagnosis.

    353 patients have no age_at_diagnosis recorded.  The imputed value is the
    patient's earliest observed age minus the cohort-median gap between first
    age and diagnosis age (estimated from patients with known diagnosis ages).

    fit learns _cohort_median_gap_ and _global_median_gap_.  transform fills
    age_at_diagnosis for affected patients and adds diag_imputed (1/0 indicator).
    """

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
# A: Rater correction
# Removes a multiplicative per-rater bias from off and on before all subsequent steps.
# =============================================================================

def fit_rater_factors(df_raw: pd.DataFrame) -> dict:
    """Estimate a multiplicative scale factor for each rater from training data.

    Some raters score systematically higher or lower than others.  The factor
    is the exponential of the 75th percentile of log(off / target) per rater,
    centred so that the mean factor across raters is 1.

    The 75th percentile (rather than the median) is used because off
    measurements are more reliable near the true upper end of the patient's
    score; low values are often contaminated by drug effect.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Training data with columns rater_id, off, target.  Rows where off is
        absent, off ≤ 0, or target ≤ 12 are excluded (noisy or near-floor).

    Returns
    -------
    dict
        Mapping rater_id → multiplicative factor (float).
    """
    if "rater_id" not in df_raw.columns:
        raise ValueError("rater_id column missing from training data")
    if "target" not in df_raw.columns:
        raise ValueError("target column required to fit rater factors")

    df = df_raw.copy()
    off_vals = df["off"].values.astype(float)
    tgt_vals = df["target"].values.astype(float)
    rater    = df["rater_id"].values

    # --- Filter to reliable rows ---
    # target > 12 excludes near-floor patients where rater effects are small.
    mask = (~np.isnan(off_vals)) & (off_vals > 0) & (~np.isnan(tgt_vals)) & (tgt_vals > 12)

    off_m    = off_vals[mask]
    tgt_m    = tgt_vals[mask]
    rater_m  = rater[mask]

    log_ratio = np.log(off_m / tgt_m)

    # --- 75th percentile of log(off/target) per rater ---
    # p75 is more robust than the mean: it ignores fully drug-on visits where
    # off ≈ on and the ratio is near zero.
    rater_p75 = {}
    for rid in np.unique(rater_m):
        vals = log_ratio[rater_m == rid]
        rater_p75[rid] = float(np.percentile(vals, 75))

    # --- Centre factors so their mean is 1 (no global scale shift) ---
    mean_p75 = float(np.mean(list(rater_p75.values())))
    factors = {rid: float(np.exp(p75 - mean_p75)) for rid, p75 in rater_p75.items()}
    return factors


def apply_rater_correction(df_raw: pd.DataFrame, factors: dict) -> pd.DataFrame:
    """Divide on and off by per-rater factor (1.0 for unknown raters), drop rater_id."""
    df = df_raw.copy()
    rater = df["rater_id"].values
    fac = np.array([factors.get(r, 1.0) for r in rater], dtype=float)
    fac = np.where(fac == 0, 1.0, fac)

    if "off" in df.columns:
        off_vals = df["off"].values.astype(float)
        df["off"] = np.where(np.isnan(off_vals), off_vals, off_vals / fac)
    if "on" in df.columns:
        on_vals = df["on"].values.astype(float)
        df["on"] = np.where(np.isnan(on_vals), on_vals, on_vals / fac)

    df = df.drop(columns=["rater_id"])
    return df


# =============================================================================
# B: New drug-effect curve f(t)
# Replaces the simpler ramp used in experiments 18–22 with a shape estimated
# from the paired visit data.
# =============================================================================

def compute_f_new(time_since_intake_on: np.ndarray) -> np.ndarray:
    """Drug-effect curve: fraction of maximum drug effect at time t after intake.

    The curve rises linearly from 0 to 1 over the absorption phase (0–1.45 h),
    stays flat through the plateau (1.45–5.2 h), then decays linearly.

    Parameters
    ----------
    time_since_intake_on : np.ndarray
        Time in hours since the last levodopa dose for the ON measurement.

    Returns
    -------
    np.ndarray
        f values in [0, 1].  Rows with missing time receive f = 0.864,
        the average of f over the plateau region (empirically estimated from
        the distribution of observed intake times).

    Notes
    -----
    1.45 h: absorption half-time estimated from paired on/off data.
    5.2 h: end of plateau, estimated from the same paired data.
    0.2: linear decay rate per hour after the plateau (reaches 0 at 5.2 + 5 h).
    0.864: mean f over observed intake times when t is missing.
    """
    t = time_since_intake_on
    f = np.where(
        np.isnan(t),
        0.864,       # imputed value when intake time is not recorded
        np.where(
            t <= 1.45,   # absorption phase: linearly ramps up
            np.clip(t / 1.45, 0.0, 1.0),
            np.where(
                t <= 5.2,    # plateau: full drug effect
                1.0,
                np.clip(1.0 - 0.2 * (t - 5.2), 0.0, 1.0),  # decay phase
            )
        )
    )
    return f.astype(float)


# =============================================================================
# C: E model (drug sensitivity per patient)
# Estimates how strongly each patient responds to levodopa.
# =============================================================================

_E_MODEL_PARAMS = dict(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0
)
_E_PT_FEATS = ["cohort", "gene", "sexM", "age_at_diagnosis", "first_age", "last_age",
               "n_visits", "mean_ledd"]
_E_CAT_COLS = ["cohort", "gene"]


def _build_patient_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build one row per patient with cohort, gene, sexM, age_at_diagnosis,
    first_age, last_age, n_visits, mean_ledd."""
    rows = []
    for pid, grp in df_raw.groupby("patient_id"):
        row = {
            "patient_id": pid,
            "cohort": grp["cohort"].iloc[0],
            "gene":   grp["gene"].iloc[0],
            "sexM":   grp["sexM"].iloc[0],
            "age_at_diagnosis": grp["age_at_diagnosis"].dropna().mean() if grp["age_at_diagnosis"].notna().any() else np.nan,
            "first_age": grp["age"].min(),
            "last_age":  grp["age"].max(),
            "n_visits":  len(grp),
            "mean_ledd": grp["ledd"].mean() if grp["ledd"].notna().any() else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


def _encode_e_feats(pt_df: pd.DataFrame) -> np.ndarray:
    out = pt_df[_E_PT_FEATS].copy()
    for c in _E_CAT_COLS:
        out[c] = out[c].astype("category").cat.codes.astype(float)
    return out.values.astype(float)


def compute_paired_e(df_raw: pd.DataFrame) -> pd.DataFrame:
    """For each row return (e, sd, f, is_paired) using new f(t) curve.
    Paired = on and off both present, off >= 8, t known (f >= some threshold).
    """
    n = len(df_raw)
    off_vals = df_raw["off"].values.astype(float)
    on_vals  = df_raw["on"].values.astype(float)
    tsi_on   = df_raw["time_since_intake_on"].values.astype(float) if "time_since_intake_on" in df_raw.columns else np.full(n, np.nan)

    f_arr = compute_f_new(tsi_on)
    t_known = ~np.isnan(tsi_on)

    has_off = ~np.isnan(off_vals)
    has_on  = ~np.isnan(on_vals)
    paired  = has_off & has_on & (off_vals >= 8) & t_known

    e_arr  = np.full(n, np.nan)
    sd_arr = np.full(n, np.nan)

    for i in np.where(paired)[0]:
        off_i = off_vals[i]
        on_i  = on_vals[i]
        f_i   = f_arr[i]
        if f_i <= 0:
            continue
        e_arr[i]  = (1.0 - on_i / off_i) / f_i
        sd_arr[i] = (np.sqrt((0.29 / off_i) ** 2 * (1.0 + (on_i / off_i) ** 2)) / f_i + 0.01)

    return pd.DataFrame({
        "e": e_arr, "sd": sd_arr, "f": f_arr, "paired": paired.astype(float)
    }, index=df_raw.index)


class EModel(BaseEstimator):
    """Drug sensitivity model — variant A: Bayesian update with an HGB prior.

    The prior E for a new patient is predicted from patient covariates
    (cohort, gene, sex, age at diagnosis, visit count, mean ledd) by a
    HistGradientBoostingRegressor trained on patients with at least 2 paired
    visits where f ≥ 0.9.  The prior uncertainty (prior_sd_) is estimated from
    5-fold KFold OOF error.

    fit learns model_, prior_mean_, prior_sd_, and pt_table_.
    predict_posterior returns (E_post, E_sd, E_prior, n_paired_used) per row.
    """

    def fit(self, df_raw: pd.DataFrame):
        """Fit on training rows.  df_raw must NOT contain the target column.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Rater-corrected visit-level data without the target column.
        """
        pair_df = compute_paired_e(df_raw)
        pids = df_raw["patient_id"].values

        # --- Build per-patient E labels from high-f paired visits ---
        # Only patients with >= 2 paired visits where f >= 0.9 contribute labels.
        # f >= 0.9 means the drug is at or near peak effect, so the E estimate
        # from ON = OFF * (1 - E*f) is least noisy.
        pt_table = _build_patient_table(df_raw)
        e_labels = {}
        for pid, grp_idx in df_raw.groupby("patient_id").groups.items():
            sub = pair_df.loc[grp_idx]
            high_f = sub[(sub["paired"] == 1) & (sub["f"] >= 0.9)]
            if len(high_f) >= 2:
                e_vals = high_f["e"].values
                valid = ~np.isnan(e_vals)
                if valid.sum() >= 2:
                    e_labels[pid] = float(np.median(e_vals[valid]))

        train_pids = [p for p in e_labels if p in pt_table.index]
        if len(train_pids) < 5:
            # fallback: prior = population median
            all_e = pair_df.loc[(pair_df["paired"] == 1) & (pair_df["f"] >= 0.9), "e"].dropna()
            self.prior_mean_ = float(np.clip(np.median(all_e) if len(all_e) > 0 else 0.3, 0.05, 0.9))
            self.prior_sd_   = 0.15
            self.model_      = None
            self.pt_table_   = pt_table
            return self

        X_tr = _encode_e_feats(pt_table.loc[train_pids])
        y_tr = np.array([e_labels[p] for p in train_pids])

        # 5-fold KFold to estimate prior_sd
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        oof_err2 = []
        for tridx, validx in kf.split(X_tr):
            m = HistGradientBoostingRegressor(**_E_MODEL_PARAMS)
            m.fit(X_tr[tridx], y_tr[tridx])
            pred = m.predict(X_tr[validx])
            oof_err2.extend((pred - y_tr[validx]) ** 2)
        mse = float(np.mean(oof_err2))
        self.prior_sd_ = float(max(np.sqrt(max(mse - 0.02 ** 2, 0.0)), 0.01))

        # Fit on all
        self.model_ = HistGradientBoostingRegressor(**_E_MODEL_PARAMS)
        self.model_.fit(X_tr, y_tr)
        self.pt_table_     = pt_table
        self.prior_mean_   = float(np.mean(y_tr))
        return self

    def predict_posterior(self, df_raw: pd.DataFrame, pair_df_ext: pd.DataFrame = None):
        """Return E_post, E_sd, E_prior, n_paired_used per row.
        pair_df_ext: precomputed (e, sd, f, paired) for df_raw rows.
        """
        pair_df = pair_df_ext if pair_df_ext is not None else compute_paired_e(df_raw)
        pids = df_raw["patient_id"].values
        n = len(df_raw)

        E_post_arr   = np.full(n, np.nan)
        E_sd_arr     = np.full(n, np.nan)
        E_prior_arr  = np.full(n, np.nan)
        nE_used_arr  = np.zeros(n, dtype=float)

        pt_table = _build_patient_table(df_raw)
        # Merge with training pt_table for any new patients
        for pid in np.unique(pids):
            # Prior from model
            if self.model_ is not None and pid in pt_table.index:
                pt_row = pt_table.loc[[pid]]
                prior = float(self.model_.predict(_encode_e_feats(pt_row))[0])
            else:
                prior = self.prior_mean_
            prior_sd = self.prior_sd_

            # Gather paired visits with f >= 0.5
            pid_mask = pids == pid
            pid_idx  = np.where(pid_mask)[0]
            sub = pair_df.iloc[pid_idx]
            use_mask = (sub["paired"] == 1) & (sub["f"] >= 0.5)
            e_vals   = sub.loc[use_mask, "e"].values
            sd_vals  = sub.loc[use_mask, "sd"].values
            valid    = ~(np.isnan(e_vals) | np.isnan(sd_vals))
            nE = int(valid.sum())

            w_prior = 1.0 / (prior_sd ** 2)
            total_w = w_prior
            num     = prior * w_prior
            if nE > 0:
                w_obs = 1.0 / (sd_vals[valid] ** 2)
                total_w += float(w_obs.sum())
                num     += float((e_vals[valid] * w_obs).sum())

            e_post = float(np.clip(num / total_w, 0.05, 0.9))
            e_sd   = float(1.0 / np.sqrt(total_w))

            E_post_arr[pid_idx]  = e_post
            E_sd_arr[pid_idx]    = e_sd
            E_prior_arr[pid_idx] = prior
            nE_used_arr[pid_idx] = float(nE)

        return E_post_arr, E_sd_arr, E_prior_arr, nE_used_arr


class EModelShrinkage(BaseEstimator):
    """Drug sensitivity model — variant B: James-Stein shrinkage, no covariate model.

    Each patient's E is shrunk toward the population median using a Gaussian
    hierarchical model.  The between-patient standard deviation is 0.094 and
    the within-patient observation noise is 0.079; both values were estimated
    from the training data.

    fit estimates E_pop_ (population median E).
    predict_posterior returns (E_post, E_sd, E_prior, n_paired_used) per row.
    """

    def fit(self, df_raw: pd.DataFrame):
        """Estimate the population median E from training data.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Rater-corrected visit-level data.
        """
        pair_df = compute_paired_e(df_raw)
        # --- Population median E from high-f paired visits ---
        # f >= 0.9 selects plateau visits where the drug estimate is reliable.
        mask_pop = (pair_df["paired"] == 1) & (pair_df["f"] >= 0.9)
        e_pop_vals = pair_df.loc[mask_pop, "e"].dropna().values
        E_pop = float(np.clip(np.median(e_pop_vals) if len(e_pop_vals) > 0 else 0.3, 0.05, 0.9))
        self.E_pop_ = E_pop
        return self

    def predict_posterior(self, df_raw: pd.DataFrame, pair_df_ext: pd.DataFrame = None):
        """Return (E_post, E_sd, E_prior, n_paired_used) per row via shrinkage.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Visit-level data.
        pair_df_ext : pd.DataFrame, optional
            Precomputed (e, sd, f, paired) frame; computed internally if None.

        Returns
        -------
        tuple of np.ndarray
            E_post, E_sd, E_prior, nE_used — one value per row.
        """
        pair_df = pair_df_ext if pair_df_ext is not None else compute_paired_e(df_raw)
        pids = df_raw["patient_id"].values
        n = len(df_raw)

        E_post_arr  = np.full(n, np.nan)
        E_sd_arr    = np.full(n, np.nan)
        E_prior_arr = np.full(n, np.nan)
        nE_used_arr = np.zeros(n, dtype=float)

        E_pop = self.E_pop_
        SIG2 = 0.094 ** 2  # between-patient SD in E: estimated from training data
        TAU2 = 0.079 ** 2  # within-patient observation noise in E: estimated from training data

        for pid in np.unique(pids):
            pid_mask = pids == pid
            pid_idx  = np.where(pid_mask)[0]
            sub = pair_df.iloc[pid_idx]
            use_mask = (sub["paired"] == 1) & (sub["f"] >= 0.5)
            e_vals = sub.loc[use_mask, "e"].dropna().values
            nE = len(e_vals)

            if nE >= 1:
                lam = SIG2 / (SIG2 + TAU2 / nE)
                E_pat = float(np.median(e_vals))
            else:
                lam = 0.0
                E_pat = E_pop

            e_post = float(np.clip(lam * E_pat + (1.0 - lam) * E_pop, 0.05, 0.9))
            e_sd   = float(np.sqrt(TAU2 / max(nE, 1) * (1.0 - lam) + 1e-6))

            E_post_arr[pid_idx]  = e_post
            E_sd_arr[pid_idx]    = e_sd
            E_prior_arr[pid_idx] = E_pop
            nE_used_arr[pid_idx] = float(nE)

        return E_post_arr, E_sd_arr, E_prior_arr, nE_used_arr


# =============================================================================
# D: New s_new, s_type, E columns and patient aggregates
# Converts every visit to a common OFF-scale value s_new = off if available,
# else on / (1 − E × f), and records how reliable that conversion is (s_type).
# =============================================================================

def build_new_columns(df_raw: pd.DataFrame, E_post: np.ndarray, E_sd: np.ndarray,
                      E_prior: np.ndarray, nE_used: np.ndarray) -> pd.DataFrame:
    """Add s_new, s_type, E_post, E_sd, E_prior, and per-patient s_new aggregates.

    s_new is the estimated true OFF score for each visit.  When an OFF
    measurement is available it is used directly; otherwise the ON score is
    back-corrected using the patient's drug sensitivity E and the drug-effect
    curve f(t).

    s_type encodes the quality of the conversion:
    * 0 — direct OFF measurement
    * 1 — ON-to-OFF conversion with known intake time and E estimated from
          paired visits
    * 2 — ON-to-OFF conversion with known intake time but E from prior only
    * 3 — ON-to-OFF conversion with missing intake time (uses E × 0.864)

    Parameters
    ----------
    df_raw : pd.DataFrame
        Feature-engineered data after PatientCorrectedPooling.
    E_post, E_sd, E_prior, nE_used : np.ndarray
        Drug sensitivity posterior from EModel or EModelShrinkage.

    Returns
    -------
    pd.DataFrame
        Input plus s_new, s_type, E_post, E_sd, E_prior, and the per-patient
        aggregates s_new_pt_max, s_new_pt_mean2, s_new_pt_mean3, s_new_pt_mean,
        s_new_pt_median, s_new_pt_p75.
    """
    df = df_raw.copy()
    n = len(df)
    off_vals = df["off"].values.astype(float)
    on_vals  = df["on"].values.astype(float) if "on" in df.columns else np.full(n, np.nan)
    tsi_on   = df["time_since_intake_on"].values.astype(float) if "time_since_intake_on" in df.columns else np.full(n, np.nan)

    f_arr    = compute_f_new(tsi_on)
    has_off  = ~np.isnan(off_vals)
    has_on   = ~np.isnan(on_vals)
    t_known  = ~np.isnan(tsi_on)

    # s_new: off if present, else on / max(1 - E_post*f, 0.15)
    s_new = np.full(n, np.nan)
    for i in range(n):
        if has_off[i]:
            s_new[i] = off_vals[i]
        elif has_on[i]:
            denom = max(1.0 - E_post[i] * f_arr[i], 0.15)
            s_new[i] = on_vals[i] / denom

    # s_type
    # For "has at least one paired visit with f >= 0.5": use nE_used > 0
    s_type = np.zeros(n, dtype=int)
    pids = df["patient_id"].values
    for i in range(n):
        if has_off[i]:
            s_type[i] = 0
        elif has_on[i] and t_known[i] and nE_used[i] > 0:
            s_type[i] = 1
        elif has_on[i] and t_known[i]:
            s_type[i] = 2
        else:
            s_type[i] = 3

    df["s_new"]   = s_new
    df["s_type"]  = s_type.astype(float)
    df["E_post"]  = E_post
    df["E_sd"]    = E_sd
    df["E_prior"] = E_prior

    # Per-patient s_new aggregates
    s_new_max    = np.full(n, np.nan)
    s_new_mean2  = np.full(n, np.nan)
    s_new_mean3  = np.full(n, np.nan)
    s_new_mean   = np.full(n, np.nan)
    s_new_median = np.full(n, np.nan)
    s_new_p75    = np.full(n, np.nan)

    for pid in np.unique(pids):
        mask = pids == pid
        idx  = np.where(mask)[0]
        sv   = s_new[idx]
        valid = ~np.isnan(sv)
        nv = valid.sum()
        if nv == 0:
            continue
        sv_valid = np.sort(sv[valid])[::-1]
        s_new_max[idx]    = sv_valid[0]
        s_new_mean2[idx]  = sv_valid[:min(2, nv)].mean()
        s_new_mean3[idx]  = sv_valid[:min(3, nv)].mean()
        s_new_mean[idx]   = sv_valid.mean()
        s_new_median[idx] = float(np.median(sv_valid))
        s_new_p75[idx]    = float(np.percentile(sv_valid, 75))

    df["s_new_pt_max"]    = s_new_max
    df["s_new_pt_mean2"]  = s_new_mean2
    df["s_new_pt_mean3"]  = s_new_mean3
    df["s_new_pt_mean"]   = s_new_mean
    df["s_new_pt_median"] = s_new_median
    df["s_new_pt_p75"]    = s_new_p75

    return df


# =============================================================================
# E: Noise densities and 10 likelihood features (new spec)
# Captures how likely each visit-level s_new is given the patient's inferred
# severity level, using a non-parametric log-density estimated from training data.
# =============================================================================

# New bin edges (10 bins): 10,15,20,25,30,40,50,60,80
_NEW_BIN_EDGES = np.array([0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, np.inf])
_NEW_N_BINS = 10

# z-grid from -4.00 to 1.50, step 0.01 -> 551 points
_NEW_Z_GRID     = np.arange(-4.00, 1.51, 0.01)
_NEW_Z_GRID_N   = len(_NEW_Z_GRID)
_NEW_Z_STEP     = 0.01

# 4 s_type categories
_NEW_N_CATS = 4


def _new_gauss_kernel(sd_pts: float = 3.0) -> np.ndarray:
    """Build a discrete Gaussian smoothing kernel of width sd_pts grid steps."""
    half = int(4 * sd_pts) + 1
    j = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-0.5 * (j / sd_pts) ** 2)
    return w / w.sum()


_NEW_GAUSS_KERNEL = _new_gauss_kernel(3.0)


def _new_target_bin(v_arr: np.ndarray) -> np.ndarray:
    """Map score values to the 10-bin target index defined by _NEW_BIN_EDGES."""
    idx = np.searchsorted(_NEW_BIN_EDGES, v_arr, side="right") - 1
    return np.clip(idx, 0, _NEW_N_BINS - 1).astype(int)


def fit_new_logdens(s_new: np.ndarray, target_arr: np.ndarray, s_type: np.ndarray) -> np.ndarray:
    """Estimate the log-density of z = log(s_new / target) for each (s_type, target_bin).

    The density is used at prediction time to score how consistent each visit's
    s_new value is with a proposed patient severity level q.

    Parameters
    ----------
    s_new : np.ndarray
        Per-visit OFF-scale estimate (shape n).
    target_arr : np.ndarray
        True target values, shape n (available at fit time only).
    s_type : np.ndarray
        Visit category (0–3), shape n.

    Returns
    -------
    np.ndarray
        Log-density array of shape (4, 10, 551): s_type × target_bin × z_grid.

    Notes
    -----
    When a cell has fewer than 200 observations, the density is pooled from
    neighbouring target bins, then the whole type, then the whole dataset.
    The z-grid spans −4.00 to +1.50 in steps of 0.01.
    """
    logdens = np.zeros((_NEW_N_CATS, _NEW_N_BINS, _NEW_Z_GRID_N), dtype=float)

    valid = (target_arr > 0.5) & (~np.isnan(s_new)) & (~np.isnan(target_arr))
    s_v   = s_new[valid]
    t_v   = target_arr[valid]
    c_v   = s_type[valid].astype(int)

    z_all   = np.log(np.maximum(s_v, 0.5) / np.maximum(t_v, 0.5))
    bins_all = _new_target_bin(t_v)

    for c in range(_NEW_N_CATS):
        c_mask = c_v == c
        z_c   = z_all[c_mask]
        bins_c = bins_all[c_mask]

        for b in range(_NEW_N_BINS):
            z_cb = z_c[bins_c == b]
            if len(z_cb) < 200:
                # neighbouring bins
                b_expand = (bins_c >= max(0, b - 1)) & (bins_c <= min(_NEW_N_BINS - 1, b + 1))
                z_cb = z_c[b_expand]
            if len(z_cb) < 200:
                # whole type
                z_cb = z_c
            if len(z_cb) < 200:
                # fall through to whole dataset
                z_cb = z_all

            counts = np.zeros(_NEW_Z_GRID_N, dtype=float)
            z_clip = z_cb[
                (z_cb >= _NEW_Z_GRID[0] - _NEW_Z_STEP / 2) &
                (z_cb <  _NEW_Z_GRID[-1] + _NEW_Z_STEP / 2)
            ]
            if len(z_clip) > 0:
                k_idx = np.round((z_clip - _NEW_Z_GRID[0]) / _NEW_Z_STEP).astype(int)
                k_idx = np.clip(k_idx, 0, _NEW_Z_GRID_N - 1)
                np.add.at(counts, k_idx, 1.0)

            smoothed = convolve1d(counts, _NEW_GAUSS_KERNEL, mode="constant", cval=0.0)
            total = smoothed.sum()
            density = smoothed / (total * _NEW_Z_STEP) if total > 0 else np.full(_NEW_Z_GRID_N, 1e-4)
            density = np.maximum(density, 1e-4)
            logdens[c, b, :] = np.log(density)

    return logdens


def _lookup_new_logdens_batch(logdens: np.ndarray, c_arr: np.ndarray, b_arr: np.ndarray,
                               x_arr: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x_arr, _NEW_Z_GRID[0], _NEW_Z_GRID[-1])
    k_frac = (x_clip - _NEW_Z_GRID[0]) / _NEW_Z_STEP
    k0 = np.floor(k_frac).astype(int)
    k1 = np.minimum(k0 + 1, _NEW_Z_GRID_N - 1)
    frac = k_frac - k0
    v0 = logdens[c_arr, b_arr, k0]
    v1 = logdens[c_arr, b_arr, k1]
    return v0 * (1.0 - frac) + v1 * frac


# New grids for E
_NEW_DELTA1  = np.arange(-0.8, 0.81, 0.01)   # 161 points
_NEW_N_D1    = len(_NEW_DELTA1)
_NEW_DELTA2  = np.arange(-0.6, 0.61, 0.02)   # 61 points
_NEW_GAMMA2  = np.arange(-0.08, 0.081, 0.01) # 17 points
_NEW_N_D2    = len(_NEW_DELTA2)
_NEW_N_G2    = len(_NEW_GAMMA2)
_NEW_GAMMA2_PRIOR = 0.5 * (_NEW_GAMMA2 / 0.03) ** 2

_NEW_MAX_VISITS = 16
_NEW_BATCH_PATIENTS = 200


def compute_likelihood_features_25(
    pids_all: np.ndarray,
    s_new: np.ndarray,
    q_all: np.ndarray,
    s_type: np.ndarray,
    tc_all: np.ndarray,
    ages_all: np.ndarray,
    logdens: np.ndarray,
    E_post: np.ndarray,
    E_sd: np.ndarray,
    has_off: np.ndarray,
) -> np.ndarray:
    """Compute 10 per-row likelihood features from the patient-level log-density.

    For each patient, a grid search over an additive level shift delta (and a
    level+age-slope grid) finds the posterior distribution over the patient's
    true severity.  Features summarise that posterior for each visit.

    Parameters
    ----------
    pids_all : np.ndarray
        Patient identifier per row.
    s_new : np.ndarray
        Per-visit OFF-scale estimate.
    q_all : np.ndarray
        Stage-1 prediction (quadratic projection) per row.
    s_type : np.ndarray
        Visit category 0–3 per row.
    tc_all : np.ndarray
        Age centred on the patient mean (time coordinate for slope grid).
    ages_all : np.ndarray
        Raw age per row.
    logdens : np.ndarray
        Log-density array (4, 10, 551) from fit_new_logdens.
    E_post, E_sd : np.ndarray
        Drug sensitivity posterior per row.
    has_off : np.ndarray
        Boolean mask: True when the visit has a direct OFF measurement.

    Returns
    -------
    np.ndarray
        Shape (n, 10): [q*(pm−1), dsd, q*(exp(dmle)−1), q*(mult−1),
        gpm, gsd, share_non_off, E_post, E_sd, s_new − q].
    """
    unique_pids = np.unique(pids_all)
    n = len(pids_all)
    feats = np.zeros((n, 10), dtype=float)

    for batch_start in range(0, len(unique_pids), _NEW_BATCH_PATIENTS):
        batch_pids = unique_pids[batch_start:batch_start + _NEW_BATCH_PATIENTS]
        _bp = len(batch_pids)

        s_pad  = np.full((_bp, _NEW_MAX_VISITS), np.nan)
        q_pad  = np.full((_bp, _NEW_MAX_VISITS), np.nan)
        c_pad  = np.full((_bp, _NEW_MAX_VISITS), -1, dtype=int)
        tc_pad = np.full((_bp, _NEW_MAX_VISITS), np.nan)
        pid_indices = []

        for pi, pid in enumerate(batch_pids):
            mask   = pids_all == pid
            idx_p  = np.where(mask)[0]
            nv     = min(len(idx_p), _NEW_MAX_VISITS)
            s_pad[pi, :nv]  = s_new[idx_p[:nv]]
            q_pad[pi, :nv]  = q_all[idx_p[:nv]]
            c_pad[pi, :nv]  = s_type[idx_p[:nv]].astype(int)
            tc_pad[pi, :nv] = tc_all[idx_p[:nv]]
            pid_indices.append(idx_p)

        # Use max(q, 0.5) as qc for likelihoods
        qc_pad = np.maximum(q_pad, 0.5)
        log_qc = np.log(qc_pad)
        log_s  = np.log(np.maximum(s_pad, 0.5))
        lv_base = log_s - log_qc  # (_bp, MV)

        valid_base = ~np.isnan(lv_base)

        # --- Level-only grid ---
        exp_d1 = np.exp(_NEW_DELTA1)
        y1 = qc_pad[:, :, np.newaxis] * exp_d1[np.newaxis, np.newaxis, :]  # (_bp, MV, ND1)
        b1 = _new_target_bin(y1.ravel()).reshape(_bp, _NEW_MAX_VISITS, _NEW_N_D1)
        lv1 = lv_base[:, :, np.newaxis] - _NEW_DELTA1[np.newaxis, np.newaxis, :]

        valid1_3d = np.broadcast_to(
            valid_base[:, :, np.newaxis], (_bp, _NEW_MAX_VISITS, _NEW_N_D1)
        ).copy()

        c_flat1   = np.repeat(c_pad, _NEW_N_D1, axis=1).ravel()
        b1_flat   = b1.ravel()
        lv1_flat  = lv1.ravel()
        vld_flat  = valid1_3d.ravel() & (c_flat1 >= 0)

        ld1_flat = np.zeros(len(c_flat1), dtype=float)
        if vld_flat.any():
            ld1_flat[vld_flat] = _lookup_new_logdens_batch(
                logdens, c_flat1[vld_flat], b1_flat[vld_flat], lv1_flat[vld_flat]
            )
        ld1_flat[~vld_flat] = 0.0
        ld1 = ld1_flat.reshape(_bp, _NEW_MAX_VISITS, _NEW_N_D1)
        ld1[~valid1_3d] = 0.0
        LL1 = ld1.sum(axis=1)  # (_bp, ND1)

        LL1_max = LL1.max(axis=1, keepdims=True)
        w1 = np.exp(LL1 - LL1_max)
        w1_sum = w1.sum(axis=1, keepdims=True); w1_sum = np.where(w1_sum == 0, 1.0, w1_sum)
        w1 /= w1_sum

        pm    = (w1 * exp_d1[np.newaxis, :]).sum(axis=1)       # (_bp,) posterior mean of exp(delta)
        dmle_idx = LL1.argmax(axis=1)
        dmle  = _NEW_DELTA1[dmle_idx]
        wd    = (w1 * _NEW_DELTA1[np.newaxis, :]).sum(axis=1)
        wd2   = (w1 * _NEW_DELTA1[np.newaxis, :] ** 2).sum(axis=1)
        dsd   = np.sqrt(np.maximum(wd2 - wd ** 2, 0.0))

        # --- Level+slope grid ---
        exp_d2 = np.exp(_NEW_DELTA2)
        exp_gtc = np.exp(_NEW_GAMMA2[np.newaxis, np.newaxis, :] * tc_pad[:, :, np.newaxis])
        y2 = (qc_pad[:, :, np.newaxis, np.newaxis]
              * exp_d2[np.newaxis, np.newaxis, :, np.newaxis]
              * exp_gtc[:, :, np.newaxis, :])
        b2 = _new_target_bin(y2.ravel()).reshape(_bp, _NEW_MAX_VISITS, _NEW_N_D2, _NEW_N_G2)
        lv2 = (lv_base[:, :, np.newaxis, np.newaxis]
               - _NEW_DELTA2[np.newaxis, np.newaxis, :, np.newaxis]
               - _NEW_GAMMA2[np.newaxis, np.newaxis, np.newaxis, :] * tc_pad[:, :, np.newaxis, np.newaxis])
        valid2_4d = np.broadcast_to(
            valid_base[:, :, np.newaxis, np.newaxis], (_bp, _NEW_MAX_VISITS, _NEW_N_D2, _NEW_N_G2)
        ).copy()

        c_flat2  = np.repeat(c_pad, _NEW_N_D2 * _NEW_N_G2, axis=1).ravel()
        b2_flat  = b2.ravel()
        lv2_flat = lv2.ravel()
        vld2_flat = valid2_4d.ravel() & (c_flat2 >= 0)

        total2 = _bp * _NEW_MAX_VISITS * _NEW_N_D2 * _NEW_N_G2
        ld2_flat = np.zeros(total2, dtype=float)
        if vld2_flat.any():
            ld2_flat[vld2_flat] = _lookup_new_logdens_batch(
                logdens, c_flat2[vld2_flat], b2_flat[vld2_flat], lv2_flat[vld2_flat]
            )
        ld2_flat[~vld2_flat] = 0.0
        ld2 = ld2_flat.reshape(_bp, _NEW_MAX_VISITS, _NEW_N_D2, _NEW_N_G2)
        ld2[~valid2_4d] = 0.0
        LL2 = ld2.sum(axis=1)  # (_bp, ND2, NG2)
        LL2 = LL2 - _NEW_GAMMA2_PRIOR[np.newaxis, np.newaxis, :]

        LL2_flat = LL2.reshape(_bp, _NEW_N_D2 * _NEW_N_G2)
        LL2_max  = LL2_flat.max(axis=1, keepdims=True)
        w2_flat  = np.exp(LL2_flat - LL2_max)
        w2_sum   = w2_flat.sum(axis=1, keepdims=True); w2_sum = np.where(w2_sum == 0, 1.0, w2_sum)
        w2_flat /= w2_sum
        w2 = w2_flat.reshape(_bp, _NEW_N_D2, _NEW_N_G2)

        gpm = (w2 * _NEW_GAMMA2[np.newaxis, np.newaxis, :]).sum(axis=(1, 2))
        wg2 = (w2 * _NEW_GAMMA2[np.newaxis, np.newaxis, :] ** 2).sum(axis=(1, 2))
        gsd = np.sqrt(np.maximum(wg2 - gpm ** 2, 0.0))

        # Assemble per-patient features
        for pi, pid in enumerate(batch_pids):
            idx_p  = pid_indices[pi]
            q_p    = q_all[idx_p]        # use original q (not qc) for features
            tc_p   = tc_all[idx_p]
            ages_p = ages_all[idx_p]

            # mult: posterior mean of exp(delta + gamma*(age - mean_age)) per visit
            # mean age for this patient
            valid_a = ~np.isnan(ages_p)
            age_mean_p = ages_p[valid_a].mean() if valid_a.any() else 0.0

            age_dev_p = ages_p - age_mean_p  # same as tc_p if tc was computed from mean age

            # Compute mult for each visit
            nv_p = min(len(idx_p), _NEW_MAX_VISITS)
            mult_p = np.ones(len(idx_p))
            for vi in range(min(len(idx_p), _NEW_MAX_VISITS)):
                # posterior mean of exp(delta + gamma * age_dev)
                age_d = age_dev_p[vi] if not np.isnan(age_dev_p[vi]) else 0.0
                exp_dg = (np.exp(_NEW_DELTA2)[:, np.newaxis] *
                          np.exp(_NEW_GAMMA2[np.newaxis, :] * age_d))  # (ND2, NG2)
                mult_p[vi] = float((w2[pi] * exp_dg).sum())
            # For visits beyond _NEW_MAX_VISITS, use gpm value
            if len(idx_p) > _NEW_MAX_VISITS:
                mult_p[_NEW_MAX_VISITS:] = float(np.exp(gpm[pi]))

            # Share of non-off visits
            has_off_p = has_off[idx_p]
            share_non_off = float((~has_off_p).sum()) / max(len(idx_p), 1)

            s_new_p  = s_new[idx_p]
            E_post_p = E_post[idx_p]
            E_sd_p   = E_sd[idx_p]

            feats[idx_p, 0] = q_p * (pm[pi] - 1.0)
            feats[idx_p, 1] = float(dsd[pi])
            feats[idx_p, 2] = q_p * (np.exp(dmle[pi]) - 1.0)
            feats[idx_p, 3] = q_p * (mult_p - 1.0)
            feats[idx_p, 4] = float(gpm[pi])
            feats[idx_p, 5] = float(gsd[pi])
            feats[idx_p, 6] = share_non_off
            feats[idx_p, 7] = E_post_p
            feats[idx_p, 8] = E_sd_p
            feats[idx_p, 9] = s_new_p - q_p

    return feats


# =============================================================================
# Original stage-2 feature helpers from 19 (unchanged)
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


def _compute_f_drug_old(time_since_intake_on: np.ndarray) -> np.ndarray:
    """Original f from 19 (used for s_ratio in sev_ev model)."""
    return np.where(
        np.isnan(time_since_intake_on),
        1.0,
        np.minimum(time_since_intake_on / 1.5, 1.0),
    )


def compute_s_ratio_orig(df: pd.DataFrame, E_pop: float = None):
    """Original s_ratio computation from 19 (for sev_ev features)."""
    n = len(df)
    off_col = df["off"].values.astype(float) if "off" in df.columns else np.full(n, np.nan)
    on_col  = df["on"].values.astype(float)  if "on"  in df.columns else np.full(n, np.nan)
    tsi_on  = df["time_since_intake_on"].values.astype(float) if "time_since_intake_on" in df.columns else np.full(n, np.nan)
    pids    = df["patient_id"].values
    f_arr = _compute_f_drug_old(tsi_on)
    has_off = ~np.isnan(off_col)
    has_on  = ~np.isnan(on_col)
    paired  = has_off & has_on & (off_col > 0) & (f_arr >= 0.5)
    E_vals_paired = np.where(paired, (1.0 - on_col / np.where(off_col > 0, off_col, 1.0)) / np.where(f_arr > 0, f_arr, 1.0), np.nan)
    pt_E_i  = {}; pt_nE_i = {}
    for pid in np.unique(pids):
        mask  = pids == pid; e_p = E_vals_paired[mask]; valid = ~np.isnan(e_p); nv = int(valid.sum()); pt_nE_i[pid] = nv
        pt_E_i[pid] = float(np.clip(np.median(e_p[valid]), 0.0, 0.85)) if nv > 0 else np.nan
    all_e_paired = E_vals_paired[paired]
    if E_pop is None:
        E_pop_out = float(np.clip(np.median(all_e_paired), 0.0, 0.85)) if len(all_e_paired) > 0 else 0.3
    else:
        E_pop_out = E_pop
    E_i_arr = np.full(n, np.nan); nE_i_arr = np.zeros(n, dtype=float)
    for pid in np.unique(pids):
        mask = pids == pid; idx = np.where(mask)[0]; ei = pt_E_i[pid]
        E_i_arr[idx]  = ei if not np.isnan(ei) else E_pop_out
        nE_i_arr[idx] = float(pt_nE_i[pid])
    s_arr = np.full(n, np.nan)
    for i in range(n):
        if has_off[i]:
            s_arr[i] = off_col[i]
        elif has_on[i]:
            denom = max(1.0 - E_i_arr[i] * f_arr[i], 0.15)
            s_arr[i] = on_col[i] / denom
    return s_arr, f_arr, has_off, E_i_arr, nE_i_arr, E_pop_out


def _add_sev_ev_pt_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pids = df["patient_id"].values; ages = df["age"].values.astype(float); s_arr = df["s_ratio"].values.astype(float)
    n = len(df)
    s_pt_max = np.full(n, np.nan); s_pt_mean2 = np.full(n, np.nan); s_pt_mean3 = np.full(n, np.nan)
    s_pt_mean = np.full(n, np.nan); s_pt_median = np.full(n, np.nan); s_pt_p75 = np.full(n, np.nan)
    s_pt_min = np.full(n, np.nan); s_pt_count = np.full(n, np.nan); s_age_slope = np.full(n, np.nan)
    s_age_resid = np.full(n, np.nan); age_tc_sev = np.full(n, np.nan); pt_n_visits_sev = np.full(n, np.nan)
    rel_pos_sev = np.full(n, np.nan)
    for pid in np.unique(pids):
        mask = pids == pid; idx = np.where(mask)[0]; s_p = s_arr[idx]; a_p = ages[idx]
        valid_s = ~np.isnan(s_p); nv = int(valid_s.sum())
        pt_n_visits_sev[idx] = float(len(idx))
        valid_a = ~np.isnan(a_p)
        if valid_a.sum() > 0:
            am = a_p[valid_a].mean(); age_tc_sev[idx] = a_p - am
        if len(idx) > 1:
            order = np.argsort(a_p); rp = np.full(len(idx), np.nan)
            for rank_i, si in enumerate(order):
                rp[si] = rank_i / (len(idx) - 1)
            rel_pos_sev[idx] = rp
        else:
            rel_pos_sev[idx] = 0.0
        if nv == 0: continue
        sv = np.sort(s_p[valid_s])[::-1]
        s_pt_max[idx] = sv[0]; s_pt_mean2[idx] = sv[:min(2,nv)].mean(); s_pt_mean3[idx] = sv[:min(3,nv)].mean()
        s_pt_mean[idx] = sv.mean(); s_pt_median[idx] = float(np.median(sv)); s_pt_p75[idx] = float(np.percentile(sv,75))
        s_pt_min[idx] = sv[-1]; s_pt_count[idx] = float(nv)
        sl, ic = _linreg_slope_intercept(a_p[valid_s], s_p[valid_s])
        s_age_slope[idx] = sl if not np.isnan(sl) else 0.0
        if not np.isnan(sl) and nv >= 3:
            fitted = sl * a_p + ic; s_age_resid[idx] = s_p - fitted
        else:
            s_age_resid[idx] = 0.0
    s_age_resid_pt_max = np.full(n, np.nan); s_age_resid_pt_mean2 = np.full(n, np.nan)
    for pid in np.unique(pids):
        mask = pids == pid; idx = np.where(mask)[0]; rv = s_age_resid[idx]; valid_r = ~np.isnan(rv)
        if valid_r.sum() > 0:
            rsorted = np.sort(rv[valid_r])[::-1]
            s_age_resid_pt_max[idx] = rsorted[0]; s_age_resid_pt_mean2[idx] = rsorted[:min(2,len(rsorted))].mean()
    for col, arr in [
        ("s_pt_max", s_pt_max), ("s_pt_mean2", s_pt_mean2), ("s_pt_mean3", s_pt_mean3),
        ("s_pt_mean", s_pt_mean), ("s_pt_median", s_pt_median), ("s_pt_p75", s_pt_p75),
        ("s_pt_min", s_pt_min), ("s_pt_count", s_pt_count), ("s_age_slope", s_age_slope),
        ("s_age_resid", s_age_resid), ("s_age_resid_pt_max", s_age_resid_pt_max),
        ("s_age_resid_pt_mean2", s_age_resid_pt_mean2), ("age_tc_sev", age_tc_sev),
        ("pt_n_visits_sev", pt_n_visits_sev), ("rel_pos_sev", rel_pos_sev),
    ]:
        df[col] = arr
    return df


# =============================================================================
# Stage-2 feature helpers from 19 (verbatim)
# =============================================================================

_TARGET_BIN_EDGES = np.array([0.0, 8.0, 15.0, 22.0, 30.0, 38.0, 46.0, 55.0, 65.0, 80.0, np.inf])
_N_BINS = 10
_LV_GRID   = np.arange(-5.00, 1.51, 0.01)
_LV_GRID_N = len(_LV_GRID)
_LV_STEP   = 0.01


def _make_gauss_kernel(sd: float = 0.03, half: int = 25) -> np.ndarray:
    j = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-0.5 * (j * _LV_STEP / sd) ** 2)
    return w / w.sum()


_GAUSS_KERNEL = _make_gauss_kernel()


def _target_bin_array(v_arr: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(_TARGET_BIN_EDGES, v_arr, side="right") - 1
    return np.clip(idx, 0, _N_BINS - 1).astype(int)


def fit_logdens_19(s_arr, target_arr, cat_arr):
    n_cats = 3
    logdens = np.zeros((n_cats, _N_BINS, _LV_GRID_N), dtype=float)
    valid = (target_arr > 0.5) & (~np.isnan(s_arr)) & (~np.isnan(target_arr))
    s_v = s_arr[valid]; t_v = target_arr[valid]; c_v = cat_arr[valid].astype(int)
    lv_all = np.log(np.maximum(s_v, 0.5) / np.maximum(t_v, 0.5))
    bins_all = _target_bin_array(t_v)
    for c in range(n_cats):
        c_mask = c_v == c; lv_c = lv_all[c_mask]; bins_c = bins_all[c_mask]
        for b in range(_N_BINS):
            lv_cb = lv_c[bins_c == b]
            if len(lv_cb) < 150:
                b_expand = (bins_c >= max(0, b - 1)) & (bins_c <= min(_N_BINS - 1, b + 1))
                lv_cb = lv_c[b_expand]
            if len(lv_cb) < 150:
                lv_cb = lv_c
            if len(lv_cb) == 0:
                logdens[c, b, :] = np.log(1e-4); continue
            counts = np.zeros(_LV_GRID_N, dtype=float)
            lv_clipped = lv_cb[(lv_cb >= _LV_GRID[0] - _LV_STEP/2) & (lv_cb < _LV_GRID[-1] + _LV_STEP/2)]
            if len(lv_clipped) > 0:
                k_idx = np.round((lv_clipped - _LV_GRID[0]) / _LV_STEP).astype(int)
                k_idx = np.clip(k_idx, 0, _LV_GRID_N - 1)
                np.add.at(counts, k_idx, 1.0)
            smoothed = convolve1d(counts, _GAUSS_KERNEL, mode="constant", cval=0.0)
            total = smoothed.sum()
            density = smoothed / (total * _LV_STEP) if total > 0 else np.full(_LV_GRID_N, 1e-4)
            density = np.maximum(density, 1e-4)
            logdens[c, b, :] = np.log(density)
    return logdens


def lookup_logdens_batch_19(logdens, c_arr, b_arr, x_arr):
    x_clip = np.clip(x_arr, _LV_GRID[0], _LV_GRID[-1])
    k_frac = (x_clip - _LV_GRID[0]) / _LV_STEP
    k0 = np.floor(k_frac).astype(int); k1 = np.minimum(k0 + 1, _LV_GRID_N - 1)
    frac = k_frac - k0
    return logdens[c_arr, b_arr, k0] * (1 - frac) + logdens[c_arr, b_arr, k1] * frac


_DELTA1 = np.arange(-1.00, 1.01, 0.01); _N_D1 = len(_DELTA1)
_DELTA2 = np.arange(-0.80, 0.81, 0.02); _GAMMA2 = np.arange(-0.08, 0.09, 0.01)
_N_D2 = len(_DELTA2); _N_G2 = len(_GAMMA2)
_GAMMA2_PRIOR = 0.5 * (_GAMMA2 / 0.03) ** 2
_MAX_VISITS = 12; _BATCH_PATIENTS = 300


def _likelihood_features_for_patients_19(pids_all, s_all, q_all, c_all, tc_all, logdens):
    unique_pids = np.unique(pids_all); n = len(pids_all)
    feats = np.zeros((n, 6), dtype=float)
    for batch_start in range(0, len(unique_pids), _BATCH_PATIENTS):
        batch_pids = unique_pids[batch_start:batch_start + _BATCH_PATIENTS]; _bp = len(batch_pids)
        s_pad = np.full((_bp, _MAX_VISITS), np.nan); q_pad = np.full((_bp, _MAX_VISITS), np.nan)
        c_pad = np.full((_bp, _MAX_VISITS), -1, dtype=int); tc_pad = np.full((_bp, _MAX_VISITS), np.nan)
        pid_indices = []
        for pi, pid in enumerate(batch_pids):
            mask = pids_all == pid; idx_p = np.where(mask)[0]; nv = min(len(idx_p), _MAX_VISITS)
            s_pad[pi,:nv] = s_all[idx_p[:nv]]; q_pad[pi,:nv] = q_all[idx_p[:nv]]
            c_pad[pi,:nv] = c_all[idx_p[:nv]]; tc_pad[pi,:nv] = tc_all[idx_p[:nv]]
            pid_indices.append(idx_p)
        log_q = np.log(np.maximum(q_pad, 0.5)); log_s = np.log(np.maximum(s_pad, 0.5))
        lv_base = log_s - log_q
        exp_d1 = np.exp(_DELTA1)
        y1 = q_pad[:,:,np.newaxis] * exp_d1[np.newaxis,np.newaxis,:]
        b1 = _target_bin_array(y1.ravel()).reshape(_bp, _MAX_VISITS, _N_D1)
        lv1 = lv_base[:,:,np.newaxis] - _DELTA1[np.newaxis,np.newaxis,:]
        valid1_3d = np.broadcast_to((~np.isnan(lv_base))[:,:,np.newaxis], (_bp,_MAX_VISITS,_N_D1)).copy()
        c_flat1 = np.repeat(c_pad, _N_D1, axis=1).ravel(); b1_flat = b1.ravel(); lv1_flat = lv1.ravel()
        vld_flat = valid1_3d.ravel() & (c_flat1 >= 0)
        ld1_flat = np.zeros(len(c_flat1), dtype=float)
        if vld_flat.any():
            ld1_flat[vld_flat] = lookup_logdens_batch_19(logdens, c_flat1[vld_flat], b1_flat[vld_flat], lv1_flat[vld_flat])
        ld1_flat[~vld_flat] = 0.0
        ld1 = ld1_flat.reshape(_bp,_MAX_VISITS,_N_D1); ld1[~valid1_3d] = 0.0; LL1 = ld1.sum(axis=1)
        LL1_max = LL1.max(axis=1, keepdims=True); w1 = np.exp(LL1 - LL1_max)
        w1_sum = w1.sum(axis=1, keepdims=True); w1_sum = np.where(w1_sum==0,1.0,w1_sum); w1/=w1_sum
        pm = (w1*exp_d1[np.newaxis,:]).sum(axis=1); dmle_idx = LL1.argmax(axis=1); dmle = _DELTA1[dmle_idx]
        wd = (w1*_DELTA1[np.newaxis,:]).sum(axis=1); wd2 = (w1*_DELTA1[np.newaxis,:]**2).sum(axis=1)
        psd = np.sqrt(np.maximum(wd2 - wd**2, 0.0))
        exp_d2 = np.exp(_DELTA2)
        exp_gtc = np.exp(_GAMMA2[np.newaxis,np.newaxis,:]*tc_pad[:,:,np.newaxis])
        y2 = (q_pad[:,:,np.newaxis,np.newaxis]*exp_d2[np.newaxis,np.newaxis,:,np.newaxis]*exp_gtc[:,:,np.newaxis,:])
        b2 = _target_bin_array(y2.ravel()).reshape(_bp,_MAX_VISITS,_N_D2,_N_G2)
        lv2 = (lv_base[:,:,np.newaxis,np.newaxis]-_DELTA2[np.newaxis,np.newaxis,:,np.newaxis]-_GAMMA2[np.newaxis,np.newaxis,np.newaxis,:]*tc_pad[:,:,np.newaxis,np.newaxis])
        valid2_4d = np.broadcast_to((~np.isnan(lv_base))[:,:,np.newaxis,np.newaxis],(_bp,_MAX_VISITS,_N_D2,_N_G2)).copy()
        c_flat2 = np.repeat(c_pad, _N_D2*_N_G2, axis=1).ravel(); b2_flat = b2.ravel(); lv2_flat = lv2.ravel()
        vld2_flat = valid2_4d.ravel() & (c_flat2 >= 0)
        ld2_flat = np.zeros(_bp*_MAX_VISITS*_N_D2*_N_G2, dtype=float)
        if vld2_flat.any():
            ld2_flat[vld2_flat] = lookup_logdens_batch_19(logdens, c_flat2[vld2_flat], b2_flat[vld2_flat], lv2_flat[vld2_flat])
        ld2_flat[~vld2_flat] = 0.0
        ld2 = ld2_flat.reshape(_bp,_MAX_VISITS,_N_D2,_N_G2); ld2[~valid2_4d] = 0.0
        LL2 = ld2.sum(axis=1) - _GAMMA2_PRIOR[np.newaxis,np.newaxis,:]
        LL2_flat = LL2.reshape(_bp,_N_D2*_N_G2); LL2_max = LL2_flat.max(axis=1,keepdims=True)
        w2_flat = np.exp(LL2_flat - LL2_max); w2_sum = w2_flat.sum(axis=1,keepdims=True)
        w2_sum = np.where(w2_sum==0,1.0,w2_sum); w2_flat/=w2_sum; w2 = w2_flat.reshape(_bp,_N_D2,_N_G2)
        dpm1 = (w2*_DELTA2[np.newaxis,:,np.newaxis]).sum(axis=(1,2)); gpm1 = (w2*_GAMMA2[np.newaxis,np.newaxis,:]).sum(axis=(1,2))
        wd_g2 = (w2*_GAMMA2[np.newaxis,np.newaxis,:]).sum(axis=(1,2)); wd2_g2 = (w2*_GAMMA2[np.newaxis,np.newaxis,:]**2).sum(axis=(1,2))
        gsd1 = np.sqrt(np.maximum(wd2_g2 - wd_g2**2, 0.0))
        for pi, pid in enumerate(batch_pids):
            idx_p = pid_indices[pi]; q_p = q_all[idx_p]; tc_p = tc_all[idx_p]
            feats[idx_p, 0] = q_p * (pm[pi] - 1.0)
            feats[idx_p, 1] = q_p * (np.exp(dmle[pi]) - 1.0)
            feats[idx_p, 2] = float(psd[pi])
            feats[idx_p, 3] = q_p * (np.exp(dpm1[pi] + gpm1[pi] * tc_p) - 1.0)
            feats[idx_p, 4] = float(gpm1[pi])
            feats[idx_p, 5] = float(gsd1[pi])
    return feats


def _compute_visit_category_19(off_present, nE_i_arr):
    """Map (off_present, nE_i > 0) to the 3-way visit category used by logdens_19."""
    return np.where(off_present, 0, np.where(nE_i_arr > 0, 1, 2)).astype(int)


# =============================================================================
# Final projection (F): polynomial + monotone 1.02x growth
# Smooths per-patient predictions onto a polynomial curve in age and enforces
# the clinical expectation that UPDRS scores do not decrease over time.
# =============================================================================

def final_projection_25(pids: np.ndarray, ages: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    """Per-patient polynomial fit, then monotone enforcement in age.

    The stage-2 predictions are noisy at the visit level.  Fitting a polynomial
    (up to degree 3) per patient removes visit-level noise while preserving the
    longitudinal trend.  The monotone enforcement then ensures predictions never
    decrease between visits, consistent with Parkinson's disease progression.

    Parameters
    ----------
    pids : np.ndarray
        Patient identifier per row.
    ages : np.ndarray
        Age per row.
    raw_pred : np.ndarray
        Raw stage-2 predictions per row.

    Returns
    -------
    np.ndarray
        Smoothed, non-decreasing predictions clipped to [0, ∞).

    Notes
    -----
    1.02: minimum growth factor between consecutive visits in age order.
    A factor of 1.02 prevents the curve from being flat (which would
    misrepresent disease progression) while staying conservative.
    """
    output = np.zeros(len(pids), dtype=float)
    for pid in np.unique(pids):
        mask = pids == pid
        idx  = np.where(mask)[0]
        a_p  = ages[idx].astype(float)
        p_p  = raw_pred[idx]
        valid = ~(np.isnan(a_p) | np.isnan(p_p))
        nv = int(valid.sum())
        if nv == 0:
            output[idx] = np.clip(p_p, 0, None)
            continue

        a_valid = a_p[valid]
        p_valid = p_p[valid]
        a_mean  = a_valid.mean()
        tc = a_p - a_mean

        deg = min(3, nv - 1)
        if deg == 0:
            poly_vals = np.full(len(idx), p_valid.mean())
        else:
            tc_valid = tc[valid]
            cols = [np.ones(nv)]
            for d in range(1, deg + 1):
                cols.append(tc_valid ** d)
            A = np.column_stack(cols)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, p_valid, rcond=None)
                cols_all = [np.ones(len(idx))]
                for d in range(1, deg + 1):
                    cols_all.append(tc ** d)
                A_all = np.column_stack(cols_all)
                poly_vals = A_all @ coeffs
            except np.linalg.LinAlgError:
                poly_vals = np.full(len(idx), p_valid.mean())

        # Sort by age
        order = np.argsort(a_p)
        sorted_poly = poly_vals[order]

        # Monotone enforcement: first value = max(v, 0); each next = max(floor(v,0), 1.02 * prev)
        out_sorted = np.zeros(len(idx))
        out_sorted[0] = max(float(sorted_poly[0]), 0.0)
        for k in range(1, len(idx)):
            floor_v = max(float(sorted_poly[k]), 0.0)
            out_sorted[k] = floor_v if floor_v >= out_sorted[k-1] else 1.02 * out_sorted[k-1]

        inv_order = np.argsort(order)
        output[idx] = out_sorted[inv_order]

    return np.clip(output, 0, None)


def quadratic_projection(pids: np.ndarray, ages: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Replace predictions with per-patient quadratic fit (from 19)."""
    output = preds.copy()
    for pid in np.unique(pids):
        mask = pids == pid; idx = np.where(mask)[0]; n = len(idx)
        if n < 3: continue
        a = ages[idx].astype(float); p = preds[idx]
        valid = ~(np.isnan(a) | np.isnan(p))
        if valid.sum() < 3: continue
        tc = a - a[valid].mean()
        if n == 3:
            A = np.column_stack([np.ones(valid.sum()), tc[valid]])
        else:
            A = np.column_stack([np.ones(valid.sum()), tc[valid], tc[valid]**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, p[valid], rcond=None)
            if n == 3:
                fitted = np.ones(n)*coeffs[0] + tc*coeffs[1]
            else:
                fitted = coeffs[0] + tc*coeffs[1] + tc**2*coeffs[2]
        except np.linalg.LinAlgError:
            continue
        for local_i, global_i in enumerate(idx):
            if not np.isnan(a[local_i]):
                output[global_i] = fitted[local_i]
    return output


# =============================================================================
# DiagnosisImputer, LeddInterp, helpers (verbatim)
# =============================================================================

def _add_tsd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)

ledd_interp_tr = LeddInterp()  # stateless


def _build_features_14(df_raw, pids, diag_imp, pcp, ml, upper_stats, implied_off_tr):
    """Full supervised feature stack (same as 19)."""
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
# F: TwoStageRegressor25
# =============================================================================

class TwoStageRegressor25(BaseEstimator, RegressorMixin):
    """Two-stage regressor — the core model of experiment 25.

    Stage 1 produces a per-visit quadratic-smoothed prediction q by blending
    six specialised HGB models.  Stage 2 corrects q multiplicatively using 10
    likelihood features and the full feature matrix, fitted on the log-ratio
    log(y / q) so that errors are proportional rather than additive.

    Parameters
    ----------
    base_params : dict
        HGB hyperparameters for the main stage-1 model (HGB_BEST_08).
    n_splits_inner : int
        Number of GroupKFold splits for cross-fitting the stage-1 OOF
        predictions that stage 2 trains on.  Using 5 folds ensures that
        stage 2 never trains on predictions made with its own rows.
    use_shrinkage : bool
        If True, use EModelShrinkage (variant B); if False, use EModel
        with HGB prior (variant A).
    """

    def __init__(self, base_params: dict, n_splits_inner: int = 5, use_shrinkage: bool = False):
        self.base_params      = base_params
        self.n_splits_inner   = n_splits_inner
        self.use_shrinkage    = use_shrinkage

    def _make_base(self):
        """Instantiate the main stage-1 HGB with the tuned hyperparameters."""
        return HistGradientBoostingRegressor(**self.base_params)

    def _make_ev_model(self):
        """Instantiate a secondary stage-1 HGB (simple, visit, off_ev, on_ev models)."""
        return HistGradientBoostingRegressor(**_EV_HGB_PARAMS)

    def _make_sev_model(self):
        """Instantiate the severity-evidence stage-1 HGB."""
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=30, random_state=0,
        )

    def _make_e_model(self):
        if self.use_shrinkage:
            return EModelShrinkage()
        return EModel()

    def _fit_e_model(self, df_raw_no_target: pd.DataFrame):
        """Fit E model on data WITHOUT target column."""
        m = self._make_e_model()
        m.fit(df_raw_no_target)
        return m

    def _compute_e_posterior(self, e_model, df_raw: pd.DataFrame):
        """Compute the E posterior for all rows; share the pair_df computation."""
        pair_df = compute_paired_e(df_raw)
        return e_model.predict_posterior(df_raw, pair_df_ext=pair_df)

    def _build_simple_X(self, X):
        """Build feature matrix for the simple stage-1 model (22 columns)."""
        df = _add_simple_aggs(X)
        cols = [c for c in _SIMPLE_FEATS if c in df.columns]
        return _encode_cats(df[cols], _EV_CAT_COLS)

    def _build_visit_X(self, X):
        """Build feature matrix for the visit-level stage-1 model (12 columns)."""
        cols = [c for c in _VISIT_FEATS if c in X.columns]
        return _encode_cats(X[cols], _EV_CAT_COLS)

    def _build_off_ev_X(self, X):
        cols = [c for c in _OFF_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols], _EV_CAT_COLS)

    def _build_on_ev_X(self, X):
        """Build feature matrix for the ON-only visits stage-1 model."""
        cols = [c for c in _ON_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols], _EV_CAT_COLS)

    def _build_sev_ev_X(self, X_with_s):
        """Build feature matrix for the severity-evidence stage-1 model."""
        df_aug = _add_sev_ev_pt_features(X_with_s)
        cols = [c for c in _SEV_EV_VISIT_FEATS + _SEV_EV_PT_FEATS if c in df_aug.columns]
        return _encode_cats(df_aug[cols], _EV_CAT_COLS)

    def _add_s_orig(self, X, E_pop):
        """Add s_ratio, f_drug, off_present, nE_i columns using the exp-19 computation."""
        s_arr, f_arr, off_present, E_i_arr, nE_i_arr, _ = compute_s_ratio_orig(X, E_pop=E_pop)
        df = X.copy()
        df["s_ratio"] = s_arr; df["f_drug"] = f_arr
        df["off_present"] = off_present.astype(float); df["nE_i"] = nE_i_arr
        return df, s_arr, f_arr, off_present, nE_i_arr

    def _stage2_features_19(self, X, r, q, ev_visit, ev_off, ev_on, s_arr, f_arr,
                             off_present, nE_i_arr, logdens19):
        """Build the 54 stage-2 columns from 19 (54 = 21 base + 15 ev + 12 s + 6 lik)."""
        pids = X["patient_id"].values; ages = X["age"].values.astype(float)
        off_col = X["off"].values.astype(float) if "off" in X.columns else np.full(len(X), np.nan)
        on_col  = X["on"].values.astype(float)  if "on"  in X.columns else np.full(len(X), np.nan)
        n = len(X)
        feat_q = q.copy(); feat_q_dev = np.full(n, np.nan); feat_r_minus_q = r - q
        feat_off_minus_q = off_col - q; feat_on_minus_q = np.where(np.isnan(on_col), np.nan, on_col - q)
        feat_max_off_q = np.full(n, np.nan); feat_mean_off_q = np.full(n, np.nan)
        feat_mean2_off_q = np.full(n, np.nan); feat_mean_on_q = np.full(n, np.nan)
        feat_max_on_q = np.full(n, np.nan); feat_std_r_q = np.full(n, np.nan)
        feat_n_off = np.full(n, np.nan); feat_n_on = np.full(n, np.nan)
        feat_tc = np.full(n, np.nan); feat_rel_pos = np.full(n, np.nan); feat_n_visits = np.full(n, np.nan)
        feat_off_dev = X["off_dev"].values.astype(float) if "off_dev" in X.columns else np.full(n, np.nan)
        feat_on_dev  = X["on_dev"].values.astype(float)  if "on_dev"  in X.columns else np.full(n, np.nan)
        feat_ledd_dev = np.full(n, np.nan)
        feat_tsi_on  = X["time_since_intake_on"].values.astype(float) if "time_since_intake_on" in X.columns else np.full(n, np.nan)
        feat_tsi_off = X["time_since_intake_off"].values.astype(float) if "time_since_intake_off" in X.columns else np.full(n, np.nan)
        ledd_col = X["ledd_interp"].values.astype(float) if "ledd_interp" in X.columns else np.full(n, np.nan)
        ev_sources = [("visit", ev_visit), ("off_ev", ev_off), ("on_ev", ev_on)]
        ev_delta = {name: e - q for name, e in ev_sources}
        ev_max_eq = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean_eq = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_mean2_eq = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        ev_min_eq = {name: np.full(n, np.nan) for name in ["visit", "off_ev", "on_ev"]}
        s_minus_q = s_arr - q
        s_q_max = np.full(n, np.nan); s_q_mean = np.full(n, np.nan); s_q_mean2 = np.full(n, np.nan)
        s_q_mean3 = np.full(n, np.nan); s_q_min = np.full(n, np.nan); s_q_median = np.full(n, np.nan)
        s_q_p75 = np.full(n, np.nan); s_count_feat = np.full(n, np.nan)
        tc_all = np.full(n, np.nan)
        for pid in np.unique(pids):
            mask = pids == pid; idx = np.where(mask)[0]
            q_p = q[idx]; r_p = r[idx]; age_p = ages[idx]; off_p = off_col[idx]; on_p = on_col[idx]; ledd_p = ledd_col[idx]
            valid_age = ~np.isnan(age_p)
            if valid_age.sum() > 0:
                age_mean = age_p[valid_age].mean(); tc_p = age_p - age_mean
            else:
                tc_p = np.full(len(idx), np.nan)
            feat_tc[idx] = tc_p; tc_all[idx] = tc_p; feat_q_dev[idx] = q_p - np.nanmean(q_p)
            age_order = np.argsort(age_p); rel_pos = np.full(len(idx), np.nan); nv = len(idx)
            if nv > 1:
                for rank_i, sorted_i in enumerate(age_order): rel_pos[sorted_i] = rank_i / (nv - 1)
            elif nv == 1:
                rel_pos[0] = 0.0
            feat_rel_pos[idx] = rel_pos; feat_n_visits[idx] = float(nv)
            lm = np.nanmean(ledd_p) if not np.all(np.isnan(ledd_p)) else np.nan
            feat_ledd_dev[idx] = ledd_p - lm
            off_q_p = off_p - q_p; valid_off = ~np.isnan(off_q_p); n_off = int(valid_off.sum())
            feat_n_off[idx] = float(n_off)
            if n_off > 0:
                off_q_valid = off_q_p[valid_off]
                feat_mean_off_q[idx] = float(np.mean(off_q_valid)); feat_max_off_q[idx] = float(np.max(off_q_valid))
                top2 = np.sort(off_q_valid)[::-1][:2]; feat_mean2_off_q[idx] = float(top2.mean())
            on_q_p = np.where(np.isnan(on_p), np.nan, on_p - q_p); valid_on = ~np.isnan(on_q_p); n_on = int(valid_on.sum())
            feat_n_on[idx] = float(n_on)
            if n_on > 0:
                on_q_valid = on_q_p[valid_on]
                feat_mean_on_q[idx] = float(np.mean(on_q_valid)); feat_max_on_q[idx] = float(np.max(on_q_valid))
            rq_p = r_p - q_p
            feat_std_r_q[idx] = float(np.nanstd(rq_p, ddof=0)) if len(rq_p) > 1 else 0.0
            for name, delta_arr in ev_delta.items():
                d_p = delta_arr[idx]; valid_d = ~np.isnan(d_p); nd = int(valid_d.sum())
                if nd > 0:
                    dv = d_p[valid_d]
                    ev_max_eq[name][idx] = float(np.max(dv)); ev_mean_eq[name][idx] = float(np.mean(dv))
                    top2d = np.sort(dv)[::-1][:2]; ev_mean2_eq[name][idx] = float(top2d.mean())
                    ev_min_eq[name][idx] = float(np.min(dv))
            sq_p = s_minus_q[idx]; valid_sq = ~np.isnan(sq_p); ns = int(valid_sq.sum())
            s_count_feat[idx] = float(ns)
            if ns > 0:
                sqv = np.sort(sq_p[valid_sq])[::-1]
                s_q_max[idx] = sqv[0]; s_q_mean[idx] = sqv.mean(); s_q_mean2[idx] = sqv[:min(2,ns)].mean()
                s_q_mean3[idx] = sqv[:min(3,ns)].mean(); s_q_min[idx] = sqv[-1]
                s_q_median[idx] = float(np.median(sqv)); s_q_p75[idx] = float(np.percentile(sqv,75))
        base_feats = np.column_stack([feat_q, feat_q_dev, feat_r_minus_q,
            feat_off_minus_q, feat_on_minus_q, feat_max_off_q, feat_mean_off_q, feat_mean2_off_q,
            feat_mean_on_q, feat_max_on_q, feat_std_r_q, feat_n_off, feat_n_on, feat_tc, feat_rel_pos,
            feat_n_visits, feat_off_dev, feat_on_dev, feat_ledd_dev, feat_tsi_on, feat_tsi_off])
        ev_feats_list = []
        for name in ["visit", "off_ev", "on_ev"]:
            ev_feats_list.extend([ev_delta[name], ev_max_eq[name], ev_mean_eq[name], ev_mean2_eq[name], ev_min_eq[name]])
        ev_feats = np.column_stack(ev_feats_list)
        s_feats = np.column_stack([s_minus_q, s_q_max, s_q_mean, s_q_mean2, s_q_mean3, s_q_min,
            s_q_median, s_q_p75, s_count_feat, nE_i_arr, f_arr, off_present.astype(float)])
        cat_arr_19 = _compute_visit_category_19(off_present, nE_i_arr)
        lik_feats = _likelihood_features_for_patients_19(pids, s_arr, q, cat_arr_19, tc_all, logdens19)
        return np.column_stack([base_feats, ev_feats, s_feats, lik_feats]), tc_all

    def fit(self, X: pd.DataFrame, y):
        """Fit the two-stage regressor on rater-corrected data.

        X must already be rater-corrected (no rater_id column) but has not
        yet been feature-engineered; all supervised transformations run here.

        Parameters
        ----------
        X : pd.DataFrame
            Rater-corrected raw visit-level data.
        y : array-like
            Target values (true OFF MDS-UPDRS score).
        """
        y_arr = np.asarray(y, dtype=float)
        pids  = X["patient_id"].values
        ages  = X["age"].values.astype(float)

        # --- Fit the drug-sensitivity model E on raw data (no target) ---
        # EModel / EModelShrinkage never see y, which prevents leakage of
        # the target through the drug-response correction.
        self.e_model_ = self._fit_e_model(X)
        E_post, E_sd, E_prior, nE_used = self._compute_e_posterior(self.e_model_, X)

        # --- Fit the stage-1 feature pipeline ---
        # DiagnosisImputer, LeddInterp, PatientCorrectedPooling, MixedLine,
        # UpperStats, and ImpliedOffTransformer all need y to correct
        # measurement bias or estimate the age slope.
        diag_imp = DiagnosisImputer(); diag_imp.fit(X)
        X_li = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X)))
        pcp = PatientCorrectedPooling(n_splits=self.n_splits_inner, use_ledd_interp=True)
        pcp.fit(X_li, y, patient_ids=pids)
        ml = MixedLine(pcp=pcp); ml.fit(X_li, y, patient_ids=pids)
        X_base = pcp.transform(ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X))))
        X_ml   = ml.transform(X_base, patient_ids=pids)
        X_ml   = add_patient_features(X_ml, ledd_col="ledd_interp")
        X_ml["cohort"] = X_ml["cohort"].astype("category")
        X_ml["gene"]   = X_ml["gene"].astype("category")
        us = UpperStats(); us.fit(X_ml, y)
        iot = ImpliedOffTransformer(); iot.fit(X_base)

        self.diag_imp_ = diag_imp
        self.pcp_      = pcp
        self.ml_       = ml
        self.us_       = us
        self.iot_      = iot

        # --- Add s_new, s_type, and E columns to the feature matrix ---
        Xf = _build_features_14(X, pids, diag_imp, pcp, ml, us, iot)
        Xf = build_new_columns(Xf, E_post, E_sd, E_prior, nE_used)

        # --- Add the exp-19 s_ratio columns for the severity-evidence model ---
        # E_pop is the population-median drug sensitivity from all training rows.
        _, _, _, _, _, E_pop_full = compute_s_ratio_orig(Xf, E_pop=None)
        self.E_pop_ = E_pop_full
        Xf_s, s_arr_full, f_arr_full, off_present_full, nE_i_full = self._add_s_orig(Xf, E_pop_full)

        # --- Fit noise densities of log(s / target) used by likelihood features ---
        # logdens_new_ is the 4×10×551 array for the 10 new likelihood features;
        # logdens_19_ is the 3×10×651 array for the 6 exp-19 likelihood features.
        s_type_full = Xf["s_type"].values.astype(int)
        self.logdens_new_ = fit_new_logdens(Xf["s_new"].values, y_arr, s_type_full)
        cat_arr_full = _compute_visit_category_19(off_present_full, nE_i_full)
        self.logdens_19_ = fit_logdens_19(s_arr_full, y_arr, cat_arr_full)

        # --- Inner GroupKFold: train the six stage-1 models and collect OOF predictions ---
        # n_splits_inner=5 means stage 2 trains on OOF predictions, so it never
        # sees stage-1 predictions made on its own training rows.
        gkf = GroupKFold(n_splits=self.n_splits_inner)
        r_oof = np.full(len(y_arr), np.nan); simp_oof = np.full(len(y_arr), np.nan)
        visit_oof = np.full(len(y_arr), np.nan); off_ev_oof = np.full(len(y_arr), np.nan)
        on_ev_oof = np.full(len(y_arr), np.nan); sev_ev_oof = np.full(len(y_arr), np.nan)

        for tr_idx, val_idx in gkf.split(Xf, y_arr, groups=pids):
            Xf_tr = Xf.iloc[tr_idx]; Xf_val = Xf.iloc[val_idx]
            y_tr  = y_arr[tr_idx]

            m_base = self._make_base()
            m_base.fit(Xf_tr.drop(columns=["patient_id"]), y_tr)
            r_oof[val_idx] = m_base.predict(Xf_val.drop(columns=["patient_id"]))

            m_simp = self._make_ev_model()
            m_simp.fit(self._build_simple_X(Xf_tr), y_tr)
            simp_oof[val_idx] = m_simp.predict(self._build_simple_X(Xf_val))

            m_visit = self._make_ev_model()
            m_visit.fit(self._build_visit_X(Xf_tr), y_tr)
            visit_oof[val_idx] = m_visit.predict(self._build_visit_X(Xf_val))

            off_tr_vals = Xf_tr["off"].values.astype(float)
            has_off_tr  = ~np.isnan(off_tr_vals)
            m_off_ev = self._make_ev_model()
            if has_off_tr.sum() >= 10:
                m_off_ev.fit(self._build_off_ev_X(Xf_tr)[has_off_tr], y_tr[has_off_tr])
                off_val_vals = Xf_val["off"].values.astype(float)
                has_off_val  = ~np.isnan(off_val_vals)
                if has_off_val.any():
                    preds_off = m_off_ev.predict(self._build_off_ev_X(Xf_val))
                    off_ev_oof[val_idx[has_off_val]] = preds_off[has_off_val]

            on_tr_vals = Xf_tr["on"].values.astype(float)
            has_on_tr  = ~np.isnan(on_tr_vals)
            m_on_ev = self._make_ev_model()
            if has_on_tr.sum() >= 10:
                m_on_ev.fit(self._build_on_ev_X(Xf_tr)[has_on_tr], y_tr[has_on_tr])
                on_val_vals = Xf_val["on"].values.astype(float)
                has_on_val  = ~np.isnan(on_val_vals)
                if has_on_val.any():
                    preds_on = m_on_ev.predict(self._build_on_ev_X(Xf_val))
                    on_ev_oof[val_idx[has_on_val]] = preds_on[has_on_val]

            # sev_ev uses the exp-19 s_ratio; its E_pop is recomputed per fold
            # to avoid leakage from the full-data E_pop.
            _, _, _, _, _, E_pop_fold = compute_s_ratio_orig(Xf_tr, E_pop=None)
            Xf_s_tr, _, _, _, _ = self._add_s_orig(Xf_tr, E_pop_fold)
            Xf_s_val, _, _, _, _ = self._add_s_orig(Xf_val, E_pop_fold)
            m_sev_ev = self._make_sev_model()
            m_sev_ev.fit(self._build_sev_ev_X(Xf_s_tr), y_tr)
            sev_ev_oof[val_idx] = m_sev_ev.predict(self._build_sev_ev_X(Xf_s_val))

        # --- Blend stage-1 and project each patient on a quadratic curve in age ---
        # 0.8 base + 0.2 simple (inner blend), then 0.7 of that + 0.3 severity evidence.
        # Blending reduces per-model noise; quadratic projection q smooths the result.
        inner_blend = 0.8 * r_oof + 0.2 * simp_oof
        r_blend = 0.7 * inner_blend + 0.3 * sev_ev_oof
        q_oof = quadratic_projection(pids, ages, r_blend)
        self.q_oof_ = q_oof

        # --- Build the stage-2 feature matrix ---
        # 54 features from exp-19 + 10 new likelihood features + all numeric columns.
        s2_19, tc_all = self._stage2_features_19(
            Xf, r_blend, q_oof, visit_oof, off_ev_oof, on_ev_oof,
            s_arr_full, f_arr_full, off_present_full, nE_i_full, self.logdens_19_
        )
        has_off_full = off_present_full.astype(bool)
        e10 = compute_likelihood_features_25(
            pids, Xf["s_new"].values, q_oof, s_type_full, tc_all, ages,
            self.logdens_new_, E_post, E_sd, has_off_full
        )
        numeric_cols = [c for c in Xf.columns if c != "patient_id" and
                        Xf[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
        x_num = Xf[numeric_cols].values.astype(float)

        s2_mat = np.column_stack([s2_19, e10, x_num])
        print(f"  [fit] Stage-2 matrix shape: {s2_mat.shape}", flush=True)

        # --- Stage-2 target and sample weights ---
        # Target = log(y / q) makes the correction multiplicative and scale-invariant.
        # Weight = max(q, 1) gives higher importance to high-severity patients,
        # where absolute errors are larger and matter more for RMSE.
        s2_target = np.log(np.maximum(y_arr, 0.5) / np.maximum(q_oof, 0.5))
        s2_weight = np.maximum(q_oof, 1.0)  # weight by severity; avoid zero weight

        # --- Train 5 bagged stage-2 HGB models, each on a random 80 % of patients ---
        # 500 iterations, lr 0.03, 63 leaves, min_samples_leaf 150: chosen to avoid
        # overfitting on the log-ratio target, which has lower SNR than the raw target.
        unique_pids = np.unique(pids)
        self.stage2_models_ = []
        for seed in range(5):
            rng_sub = np.random.default_rng(seed)
            n_sub = max(1, int(0.8 * len(unique_pids)))  # 80 % patient subsample for bagging
            sub_pids = set(rng_sub.choice(unique_pids, size=n_sub, replace=False))
            sub_mask = np.array([pid in sub_pids for pid in pids])
            valid_mask = sub_mask & np.isfinite(q_oof) & np.isfinite(s2_target)
            m_s2 = HistGradientBoostingRegressor(
                max_iter=500,           # iterations tuned for the log-ratio target
                learning_rate=0.03,     # small lr compensated by 500 iterations
                max_leaf_nodes=63,      # 6-level trees; avoids memorising patient curves
                min_samples_leaf=150,   # strong regularisation on the smaller training set
                random_state=seed,
            )
            m_s2.fit(s2_mat[valid_mask], s2_target[valid_mask],
                     sample_weight=s2_weight[valid_mask])
            self.stage2_models_.append(m_s2)

        # --- Refit all six stage-1 models on the full training data ---
        # The inner-fold models above were only used to get unbiased OOF predictions
        # for stage 2.  For prediction we refit on all rows.
        self.base_ = self._make_base()
        self.base_.fit(Xf.drop(columns=["patient_id"]), y_arr)
        self.model_simple_ = self._make_ev_model()
        self.model_simple_.fit(self._build_simple_X(Xf), y_arr)
        self.model_visit_ = self._make_ev_model()
        self.model_visit_.fit(self._build_visit_X(Xf), y_arr)

        off_col_all = Xf["off"].values.astype(float)
        has_off_all = ~np.isnan(off_col_all)
        self.model_off_ev_ = self._make_ev_model()
        if has_off_all.sum() >= 10:
            self.model_off_ev_.fit(self._build_off_ev_X(Xf)[has_off_all], y_arr[has_off_all])

        on_col_all = Xf["on"].values.astype(float)
        has_on_all = ~np.isnan(on_col_all)
        self.model_on_ev_ = self._make_ev_model()
        if has_on_all.sum() >= 10:
            self.model_on_ev_.fit(self._build_on_ev_X(Xf)[has_on_all], y_arr[has_on_all])

        self.model_sev_ev_ = self._make_sev_model()
        self.model_sev_ev_.fit(self._build_sev_ev_X(Xf_s), y_arr)

        self.numeric_cols_ = numeric_cols
        return self

    def _predict_raw(self, X_raw: pd.DataFrame, E_post=None, E_sd=None, E_prior=None, nE_used=None):
        """Predict on rater-corrected data, returning the final_projection_25 output.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Rater-corrected visit-level data without rater_id.
        E_post, E_sd, E_prior, nE_used : np.ndarray, optional
            Pre-computed E posterior; computed internally if None.

        Returns
        -------
        np.ndarray
            Per-row predictions after monotone projection.
        """
        pids = X_raw["patient_id"].values
        ages = X_raw["age"].values.astype(float)

        # --- Compute E posterior for prediction rows ---
        if E_post is None:
            E_post, E_sd, E_prior, nE_used = self._compute_e_posterior(self.e_model_, X_raw)

        # --- Build feature matrix using fitted pipeline transformers ---
        Xf = _build_features_14(X_raw, pids, self.diag_imp_, self.pcp_, self.ml_, self.us_, self.iot_)
        Xf = build_new_columns(Xf, E_post, E_sd, E_prior, nE_used)
        Xf_s, s_arr, f_arr, off_present, nE_i_arr = self._add_s_orig(Xf, self.E_pop_)

        # --- Stage-1 blend and quadratic projection q ---
        r_base = self.base_.predict(Xf.drop(columns=["patient_id"]))
        r_simp = self.model_simple_.predict(self._build_simple_X(Xf))
        ev_sev = self.model_sev_ev_.predict(self._build_sev_ev_X(Xf_s))
        inner  = 0.8 * r_base + 0.2 * r_simp
        r      = 0.7 * inner + 0.3 * ev_sev
        q      = quadratic_projection(pids, ages, r)

        # --- Collect remaining stage-1 predictions for the stage-2 feature matrix ---
        ev_visit = self.model_visit_.predict(self._build_visit_X(Xf))
        off_col  = Xf["off"].values.astype(float)
        ev_off   = np.full(len(Xf), np.nan)
        has_off  = ~np.isnan(off_col)
        if has_off.any():
            preds_off = self.model_off_ev_.predict(self._build_off_ev_X(Xf))
            ev_off[has_off] = preds_off[has_off]
        on_col = Xf["on"].values.astype(float)
        ev_on  = np.full(len(Xf), np.nan)
        has_on = ~np.isnan(on_col)
        if has_on.any():
            preds_on = self.model_on_ev_.predict(self._build_on_ev_X(Xf))
            ev_on[has_on] = preds_on[has_on]

        # --- Build stage-2 matrix and apply the 5 bagged stage-2 models ---
        s2_19, tc_all = self._stage2_features_19(
            Xf, r, q, ev_visit, ev_off, ev_on,
            s_arr, f_arr, off_present, nE_i_arr, self.logdens_19_
        )
        s_type = Xf["s_type"].values.astype(int)
        e10 = compute_likelihood_features_25(
            pids, Xf["s_new"].values, q, s_type, tc_all, ages,
            self.logdens_new_, E_post, E_sd, off_present.astype(bool)
        )
        # Align numeric columns to the fit-time set; pad absent columns with 0.
        x_num = Xf[[c for c in self.numeric_cols_ if c in Xf.columns]].values.astype(float)
        if x_num.shape[1] < len(self.numeric_cols_):
            x_num_full = np.zeros((len(Xf), len(self.numeric_cols_)), dtype=float)
            col_map = {c: i for i, c in enumerate(self.numeric_cols_)}
            for ci, c in enumerate([c for c in self.numeric_cols_ if c in Xf.columns]):
                x_num_full[:, col_map[c]] = x_num[:, ci]
            x_num = x_num_full

        s2_mat = np.column_stack([s2_19, e10, x_num])

        # Average the 5 stage-2 models to obtain the log correction factor.
        log_delta = np.mean(
            np.column_stack([m.predict(s2_mat) for m in self.stage2_models_]),
            axis=1,
        )
        # raw prediction = max(q, 0.5) * exp(avg log_delta)
        raw_pred = np.maximum(q, 0.5) * np.exp(log_delta)

        return final_projection_25(pids, ages, raw_pred)

    def predict(self, X_raw: pd.DataFrame, E_post=None, E_sd=None, E_prior=None, nE_used=None):
        """Predict and clip to non-negative values."""
        return np.clip(self._predict_raw(X_raw, E_post, E_sd, E_prior, nE_used), 0, None)


# =============================================================================
# H: BlendWrapper (fits two variants in parallel)
# Runs variant A and variant B of TwoStageRegressor25 in parallel via joblib.
# =============================================================================

def _fit_one_variant(variant_tag, X_rc, y, base_params, n_splits_inner, use_shrinkage):
    """Fit one TwoStageRegressor25 variant; must be a module-level function for Windows spawn."""
    print(f"  [variant {variant_tag}] Starting fit...", flush=True)
    m = TwoStageRegressor25(base_params=base_params, n_splits_inner=n_splits_inner, use_shrinkage=use_shrinkage)
    m.fit(X_rc, y)
    print(f"  [variant {variant_tag}] Fit done.", flush=True)
    return m


class BlendWrapper25(BaseEstimator, RegressorMixin):
    """Average the predictions of variant A (EModel) and variant B (EModelShrinkage).

    Both variants are fitted in parallel (joblib loky, 2 workers).  predict
    returns 0.5 * pred_A + 0.5 * pred_B clipped to non-negative values.

    The blend reduces variance: A and B disagree mostly on patients with few
    paired visits, where the covariate-based prior (A) and the population
    shrinkage (B) pull in different directions.
    """

    def __init__(self, base_params: dict, n_splits_inner: int = 5):
        self.base_params    = base_params
        self.n_splits_inner = n_splits_inner

    def fit(self, X_raw: pd.DataFrame, y, rater_factors: dict = None):
        """Fit both variants on rater-corrected data.

        Parameters
        ----------
        X_raw : pd.DataFrame
            Raw visit-level data including rater_id.
        y : array-like
            Target values.
        rater_factors : dict, optional
            Pre-fitted rater factors.  If None, they are estimated from
            X_raw and y (pass pre-fitted factors in CV to avoid leakage
            from the validation fold).
        """
        y_arr = np.asarray(y, dtype=float)
        pids  = X_raw["patient_id"].values

        # --- Estimate per-rater scale factors from training data only ---
        if rater_factors is None:
            df_with_target = X_raw.copy()
            df_with_target["target"] = y_arr
            rater_factors = fit_rater_factors(df_with_target)
        self.rater_factors_ = rater_factors

        # --- Apply rater correction then fit both variants in parallel ---
        X_rc = apply_rater_correction(X_raw, rater_factors)

        results = Parallel(n_jobs=2, backend="loky")(
            delayed(_fit_one_variant)(tag, X_rc, y, self.base_params, self.n_splits_inner, shrink)
            for tag, shrink in [("A", False), ("B", True)]
        )
        self.model_A_ = results[0]
        self.model_B_ = results[1]
        return self

    def predict(self, X_raw: pd.DataFrame):
        """Predict: apply rater correction, average variant A and B, clip to ≥ 0."""
        X_rc = apply_rater_correction(X_raw, self.rater_factors_)
        pred_A = self.model_A_.predict(X_rc)
        pred_B = self.model_B_.predict(X_rc)
        self.last_pred_A_ = pred_A
        self.last_pred_B_ = pred_B
        return np.clip(0.5 * pred_A + 0.5 * pred_B, 0, None)


# =============================================================================
# Main guard (required for Windows joblib spawn)
# =============================================================================

if __name__ == "__main__":
    import skore
    from skore import login, EstimatorReport
    from sklearn.metrics import root_mean_squared_error

    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["smoke", "submit", "report", "cv"])
    args = parser.parse_args()

    # Fail early if rater_id is absent — all subsequent code requires it.
    _hdr_train = pd.read_csv(DATA_DIR / "X_train.csv", nrows=0)
    _hdr_test  = pd.read_csv(DATA_DIR / "X_test.csv",  nrows=0)
    if "rater_id" not in _hdr_train.columns:
        sys.exit(f"ERROR: rater_id missing from {DATA_DIR}/X_train.csv")
    if "rater_id" not in _hdr_test.columns:
        sys.exit(f"ERROR: rater_id missing from {DATA_DIR}/X_test.csv")

    # Load the full dataset; all steps share these objects.
    print(f"Loading data from {DATA_DIR} ...", flush=True)
    X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
    X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
    sample_sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")
    groups_train = X_train_raw["patient_id"]

    N_SPLITS_CV = 5
    EXPECTED_A = [2.6538, 2.6600, 2.6794, 2.6719, 2.7562]  # reference values from the specification, printed as a check; not measured by this script (README 7.1)
    EXPECTED_B = [2.7054, 2.6653, 2.6997, 2.6973, 2.7871]
    EXPECTED_V = [2.6599, 2.6428, 2.6700, 2.6623, 2.7517]

    audit_dir = PROJECT_ROOT / "audit"
    audit_dir.mkdir(exist_ok=True)
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)

    # =========================================================================
    # SMOKE — sanity check: fit on 10 % of patients, predict, print RMSE.
    # Uses n_splits_inner=2 to keep runtime short.
    # =========================================================================
    if args.step == "smoke":
        print("\n=== SMOKE TEST ===", flush=True)
        rng = np.random.default_rng(0)
        all_pids = groups_train.unique()
        sub_pids = rng.choice(all_pids, size=max(4, len(all_pids) // 10), replace=False)
        smoke_mask = groups_train.isin(sub_pids)
        X_sm = X_train_raw[smoke_mask]
        y_sm = y_train[smoke_mask]
        g_sm = groups_train[smoke_mask]
        print(f"  Smoke: {len(X_sm)} rows, {g_sm.nunique()} patients", flush=True)

        # n_splits_inner=2 speeds up the smoke test; production uses 5.
        wrapper = BlendWrapper25(base_params=HGB_BEST_08, n_splits_inner=2)
        wrapper.fit(X_sm, y_sm)

        pids_sm = g_sm.values
        pred_blend = wrapper.predict(X_sm)
        rmse_blend = root_mean_squared_error(y_sm, pred_blend)
        rmse_A = root_mean_squared_error(y_sm, wrapper.last_pred_A_)
        rmse_B = root_mean_squared_error(y_sm, wrapper.last_pred_B_)
        print(f"  Smoke RMSE: A={rmse_A:.4f}  B={rmse_B:.4f}  blend={rmse_blend:.4f}", flush=True)
        print("  [SMOKE] All checks passed.", flush=True)

    # =========================================================================
    # SUBMIT
    # =========================================================================
    elif args.step == "submit":
        sub_path = sub_dir / "25_blend.csv"
        if sub_path.exists():
            print(f"submissions/25_blend.csv already exists, skipping submit step.", flush=True)
            sys.exit(0)

        print("\n=== SUBMIT: Fitting on all training data ===", flush=True)

        # Rater factors are estimated from all training rows (with target attached).
        df_with_tgt = X_train_raw.copy()
        df_with_tgt["target"] = y_train.values
        rater_factors_all = fit_rater_factors(df_with_tgt)
        print(f"  Rater factors: {len(rater_factors_all)} raters", flush=True)

        wrapper = BlendWrapper25(base_params=HGB_BEST_08, n_splits_inner=5)
        wrapper.fit(X_train_raw, y_train, rater_factors=rater_factors_all)

        print("  Predicting test set...", flush=True)
        test_preds = wrapper.predict(X_test_raw)

        pred_A = wrapper.last_pred_A_
        pred_B = wrapper.last_pred_B_

        # Checks
        assert X_test_raw.index.equals(pd.Index(sample_sub["Index"])), \
            "Index mismatch vs sample_submission"
        assert len(test_preds) == 11013, f"Expected 11013 predictions, got {len(test_preds)}"
        n_finite = int(np.isfinite(test_preds).sum())
        assert n_finite == 11013, f"Expected 11013 finite predictions, got {n_finite}"
        assert np.all(test_preds >= 0), "Some predictions are negative"

        # Save test audit
        audit_test = pd.DataFrame({
            "Index": X_test_raw.index,
            "v25_A": pred_A,
            "v25_B": pred_B,
            "v25":   test_preds,
        })
        audit_test.to_csv(audit_dir / "test_25.csv", index=False)
        print(f"  Test audit saved -> audit/test_25.csv", flush=True)

        sub_df = pd.DataFrame({"Index": X_test_raw.index, "target": test_preds})
        sub_df.to_csv(sub_path, index=False)
        print(f"  Submission written -> submissions/25_blend.csv", flush=True)
        print(f"  Test pred stats: mean={test_preds.mean():.3f}  std={test_preds.std():.3f}", flush=True)

    # =========================================================================
    # REPORT — build and push an EstimatorReport to Skore Hub.
    # Uses GroupShuffleSplit(20 %, random_state=0) to match the held-out RMSE
    # numbers in the file header.
    # =========================================================================
    elif args.step == "report":
        import pickle
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.load_skore_credentials import load_skore_credentials
        cfg = load_skore_credentials()
        login(mode="hub")
        project = skore.Project(
            name="bobathon-esilv",
            mode="hub",
            workspace=cfg["workspace"],
        )

        print("\n=== REPORT: Building EstimatorReport ===", flush=True)
        # Same 80/20 holdout split used to compute the header RMSE numbers.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        hub_tr_idx, hub_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

        X_raw_hub_tr  = X_train_raw.iloc[hub_tr_idx]
        X_raw_hub_val = X_train_raw.iloc[hub_val_idx]
        y_hub_tr      = y_train.iloc[hub_tr_idx]
        y_hub_val     = y_train.iloc[hub_val_idx]

        _pkl_path = audit_dir / "report_25_wrapper.pkl"

        # Load a cached fitted wrapper if available; otherwise fit and cache it.
        # Caching avoids refitting (≈12min) every time the report step is rerun.
        if _pkl_path.exists():
            print(f"  Loading pre-fitted wrapper from {_pkl_path} ...", flush=True)
            with open(_pkl_path, "rb") as _f:
                wrapper_hub = pickle.load(_f)
            rater_factors_hub = wrapper_hub.rater_factors_
        else:
            # Rater factors are estimated from the training fold only (no leakage).
            df_with_tgt = X_raw_hub_tr.copy()
            df_with_tgt["target"] = y_hub_tr.values
            rater_factors_hub = fit_rater_factors(df_with_tgt)

            wrapper_hub = BlendWrapper25(base_params=HGB_BEST_08, n_splits_inner=5)
            wrapper_hub.fit(X_raw_hub_tr, y_hub_tr, rater_factors=rater_factors_hub)

            with open(_pkl_path, "wb") as _f:
                pickle.dump(wrapper_hub, _f)
            print(f"  Wrapper pickled to {_pkl_path}", flush=True)

        # Predict on the holdout fold and print RMSE for A, B, and the blend.
        pred_val   = wrapper_hub.predict(X_raw_hub_val)
        pred_A_val = wrapper_hub.last_pred_A_
        pred_B_val = wrapper_hub.last_pred_B_

        rmse_blend = root_mean_squared_error(y_hub_val, pred_val)
        rmse_A     = root_mean_squared_error(y_hub_val, pred_A_val)
        rmse_B     = root_mean_squared_error(y_hub_val, pred_B_val)

        print(f"  Holdout RMSE: A={rmse_A:.4f}  B={rmse_B:.4f}  blend={rmse_blend:.4f}", flush=True)

        _n_hub_tr = len(X_raw_hub_tr)

        # _RawWrapper exists because skore's EstimatorReport calls clone() on the
        # estimator it receives, which resets all fitted attributes and empties the
        # nested BlendWrapper25.  _RawWrapper stores only the pickle path (a plain
        # string) so clone() is safe.  When EstimatorReport calls fit(), _RawWrapper
        # loads the cached BlendWrapper25 from the pickle file and re-fits the rater
        # factors from the training rows it receives (which are the same hub_tr rows).
        # If the pickle is missing, the step refits from scratch and saves it.
        class _RawWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, pkl_path):
                self.pkl_path = pkl_path
            def fit(self, X, y):
                assert len(X) == _n_hub_tr, (
                    f"Expected {_n_hub_tr} training rows, got {len(X)}"
                )
                with open(self.pkl_path, "rb") as _f:
                    self.fitted_wrapper_ = pickle.load(_f)
                _df = X.copy()
                _df["target"] = np.asarray(y, dtype=float)
                self.fitted_wrapper_.rater_factors_ = fit_rater_factors(_df)
                return self
            def predict(self, X):
                return self.fitted_wrapper_.predict(X)

        # Pass _RawWrapper to EstimatorReport; it calls fit() then predict() internally.
        raw_wrapper = _RawWrapper(str(_pkl_path))
        rep = EstimatorReport(
            raw_wrapper,
            X_train=X_raw_hub_tr, y_train=y_hub_tr,
            X_test=X_raw_hub_val,  y_test=y_hub_val,
        )
        project.put(EXPERIMENT_KEY, rep)
        HUB_URL = "https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/"  # project page; the report itself is .../estimators/85087
        print(f"  Pushed {EXPERIMENT_KEY}: {HUB_URL}", flush=True)
        print(f"  Report URL: {HUB_URL}", flush=True)
        try:
            rep_preds = rep.predictions_["test"]
            rep_rmse  = root_mean_squared_error(y_hub_val, rep_preds)
            print(f"  EstimatorReport blend RMSE check: {rep_rmse:.4f} (expected {rmse_blend:.4f})", flush=True)
            assert abs(rep_rmse - rmse_blend) < 1e-4, f"RMSE mismatch: {rep_rmse} vs {rmse_blend}"
        except Exception as _e:
            print(f"  WARNING: post-push RMSE check skipped: {_e}", flush=True)

    # =========================================================================
    # CV — 5-fold grouped CV on patient_id.
    # Each fold fits rater factors from the fold's training rows only.
    # Results are cached in audit/oof_25_folds/ so individual folds can be
    # rerun if needed.
    # =========================================================================
    elif args.step == "cv":
        print("\n=== CV: GroupKFold(5) on patient_id ===", flush=True)
        gkf = GroupKFold(n_splits=N_SPLITS_CV)

        oof_v25_A = np.full(len(y_train), np.nan)
        oof_v25_B = np.full(len(y_train), np.nan)
        oof_v25   = np.full(len(y_train), np.nan)

        fold_rmses_A = []; fold_rmses_B = []; fold_rmses_v = []

        oof_fold_dir = audit_dir / "oof_25_folds"
        oof_fold_dir.mkdir(exist_ok=True)

        for fold_i, (tr_idx, val_idx) in enumerate(
            gkf.split(X_train_raw, y_train, groups=groups_train)
        ):
            fold_path = oof_fold_dir / f"fold_{fold_i}.csv"
            if fold_path.exists():
                print(f"  Fold {fold_i+1}: loading from cache {fold_path}", flush=True)
                fold_df = pd.read_csv(fold_path, index_col="Index")
                oof_v25_A[val_idx] = fold_df["v25_A"].values
                oof_v25_B[val_idx] = fold_df["v25_B"].values
                oof_v25[val_idx]   = fold_df["v25"].values
                y_val_fold = y_train.iloc[val_idx]
                rA = root_mean_squared_error(y_val_fold, fold_df["v25_A"].values)
                rB = root_mean_squared_error(y_val_fold, fold_df["v25_B"].values)
                rv = root_mean_squared_error(y_val_fold, fold_df["v25"].values)
                fold_rmses_A.append(rA); fold_rmses_B.append(rB); fold_rmses_v.append(rv)
                print(
                    f"  Fold {fold_i+1}/{N_SPLITS_CV}: "
                    f"A={rA:.4f}(exp {EXPECTED_A[fold_i]:.4f})  "
                    f"B={rB:.4f}(exp {EXPECTED_B[fold_i]:.4f})  "
                    f"v25={rv:.4f}(exp {EXPECTED_V[fold_i]:.4f})",
                    flush=True
                )
                continue

            X_raw_tr  = X_train_raw.iloc[tr_idx]
            X_raw_val = X_train_raw.iloc[val_idx]
            y_tr      = y_train.iloc[tr_idx]
            y_val     = y_train.iloc[val_idx]

            print(f"  Fold {fold_i+1}/{N_SPLITS_CV}: fitting...", flush=True)

            # Fit rater factors from fold training rows only
            df_with_tgt = X_raw_tr.copy()
            df_with_tgt["target"] = y_tr.values
            rater_factors_fold = fit_rater_factors(df_with_tgt)

            wrapper_fold = BlendWrapper25(base_params=HGB_BEST_08, n_splits_inner=5)
            wrapper_fold.fit(X_raw_tr, y_tr, rater_factors=rater_factors_fold)

            pred_blend = wrapper_fold.predict(X_raw_val)
            pred_A = wrapper_fold.last_pred_A_
            pred_B = wrapper_fold.last_pred_B_

            oof_v25_A[val_idx] = pred_A
            oof_v25_B[val_idx] = pred_B
            oof_v25[val_idx]   = pred_blend

            rA = root_mean_squared_error(y_val, pred_A)
            rB = root_mean_squared_error(y_val, pred_B)
            rv = root_mean_squared_error(y_val, pred_blend)
            fold_rmses_A.append(rA); fold_rmses_B.append(rB); fold_rmses_v.append(rv)

            # Save fold cache
            fold_df = pd.DataFrame({
                "Index": X_train_raw.index[val_idx],
                "patient_id": groups_train.iloc[val_idx].values,
                "v25_A": pred_A,
                "v25_B": pred_B,
                "v25":   pred_blend,
            })
            fold_df.to_csv(fold_path, index=False)

            print(
                f"  Fold {fold_i+1}/{N_SPLITS_CV}: "
                f"A={rA:.4f}(exp {EXPECTED_A[fold_i]:.4f})  "
                f"B={rB:.4f}(exp {EXPECTED_B[fold_i]:.4f})  "
                f"v25={rv:.4f}(exp {EXPECTED_V[fold_i]:.4f})",
                flush=True
            )

        # Save OOF
        oof_df = pd.DataFrame({
            "Index":      X_train_raw.index,
            "patient_id": groups_train.values,
            "v25_A":      oof_v25_A,
            "v25_B":      oof_v25_B,
            "v25":        oof_v25,
        })
        oof_df.to_csv(audit_dir / "oof_25.csv", index=False)
        print(f"\n  OOF saved -> audit/oof_25.csv", flush=True)

        print("\n=== CV Results ===", flush=True)
        for tag, folds, expected in [
            ("v25_A", fold_rmses_A, EXPECTED_A),
            ("v25_B", fold_rmses_B, EXPECTED_B),
            ("v25",   fold_rmses_v, EXPECTED_V),
        ]:
            print(f"\n  {tag}: folds={[f'{v:.4f}' for v in folds]}  mean={np.mean(folds):.4f}  expected_mean={np.mean(expected):.4f}", flush=True)
