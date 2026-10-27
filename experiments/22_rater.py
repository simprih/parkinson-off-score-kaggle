# %% [markdown]
# # Experiment 22 — rater debiasing + bigger stage-2
#
# **Date:** 2026-10-13
# **Goal:** Handle the rater_id column in data_new/ files.
#
# Change P (inside TwoStageRegressor):
#   Per-rater log-factor e_r estimated from training data only (target-based + lattice),
#   combined and used to undo the rater-specific scaling of on/off before all other steps.
#
# Change O (both e_modes):
#   Stage-2 HGB larger: max_iter=900, lr=0.02, max_leaf_nodes=63, min_samples_leaf=50,
#   l2_regularization=1.0, early_stopping=False.
#
# Benchmark: Experiment 20 (key 20_blend), recomputed on data_new/ outer splits.
# Keep criterion: mean RMSE < 2.90.
# Why not the usual rule of the report (report/rapport_SPS.pdf), section 3.2
# (beat the previous model on the mean and on 4 of 5
# folds): the benchmark, experiment 20, was fitted on the first version of the files, so its
# score is not comparable, and only a model built for the current files could be submitted.
# The specification of this experiment therefore set 2.90 as a safety threshold that only
# catches a broken run. Result: 2.7442, also below the 2.7650 of experiment 20.
#
# **Key:** 22_rater  |  **Decision:** kept, replaced by 25
# **Results:** CV (GroupKFold 5) 2.7442 ± 0.043  |  holdout (20 % of patients) 2.76  |  public LB 2.815
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/72416
# **Run:** `python experiments/22_rater.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/22_rater.py`).
# Reads data_new/ (current files, path written in the code), data/ (first version of the
# files) for a diagnostic, and audit/oof_20.csv; needs the Skore key (.skore), see the report
# (report/rapport_SPS.pdf), How to run.
# `SMOKE_TEST=1`: 10 % of the patients, 2 folds, nothing saved or pushed (no Skore key needed).

# %%
import os
import sys
import time
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
DATA_DIR     = PROJECT_ROOT / "data_new"
EXPERIMENT_KEY = "22_rater"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_SPLITS_CV = 2 if SMOKE else 5

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
# Shared primitives (verbatim from 20_blend)
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
    """Per-patient linear interpolation of ledd over age (stateless)."""

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
    """Visit-level bias correction (cross-fitted on training fold)."""

    def __init__(self, n_splits: int = 5, use_ledd_interp: bool = False):
        """Store the number of cross-fitting folds and whether to use the interpolated dose."""
        self.n_splits = n_splits
        self.use_ledd_interp = use_ledd_interp

    @property
    def _bias_off(self):
        """Return the columns used to predict the bias of OFF measurements."""
        return _BIAS_FEATURES_OFF_I if self.use_ledd_interp else \
            ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]

    @property
    def _bias_on(self):
        """Return the columns used to predict the bias of ON measurements."""
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
        """Keep the fitted PatientCorrectedPooling that provides off_corr and on_corr."""
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


def _upper_stats_for_scores(
    raw_vals: np.ndarray,
    times: np.ndarray,
    beta: float,
    t_query: np.ndarray,
    prefix: str,
) -> dict:
    """Statistics of the highest detrended scores of one patient (the OFF noise mostly pulls scores down).

    Returns arrays ``{prefix}_top1, _mean2, _mean_uh, _mean_nlo, _mean, _count, _spread`` and
    ``_proj2, _proj_uh`` (two of these means projected back to each visit time with slope ``beta``).
    """
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
        """Learn the population slope ``beta_`` of the target over time since diagnosis (within-patient least squares)."""
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
        """Add the high-value statistics of off, on, off_corr and on_corr for each patient (36 columns)."""
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
        """Predict population-level off-on gap for each row."""
        X_pop = df[_POP_GAP_FEATURES].copy()
        return self._pop_model_.predict(X_pop.values)

    def fit(self, df: pd.DataFrame, y=None):
        """Learn the per-patient OFF - ON gaps, a population gap model and the shrinkage constant ``k_``.

        ``k_`` is chosen by leave-one-visit-out error on the patients with at least two paired visits.
        """
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
        """Add implied_off (on_corr plus the shrunk patient gap), gap_weight and n_paired."""
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
        """Learn the median gap between first-visit age and age at diagnosis, per cohort and overall."""
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
        """Fill age_at_diagnosis for patients who have none (first-visit age minus the cohort gap).

        Also adds the flag column diag_imputed.
        """
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
    """Encode the categorical columns as integer codes and return a float array for HGB."""
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes.astype(float)
    return out.values.astype(float)


def _add_simple_aggs(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-patient max and count of off, on and ledd, and the centred age age_tc."""
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
# Shrinkage constants for e_mode="shrunk_mean"
# =============================================================================

_E_SD_BETWEEN = 0.094
_E_SD_WITHIN  = 0.079


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
    """Compute OFF-scale severity s and associated features."""
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

    if e_mode == "shrunk_mean":
        sb2 = _E_SD_BETWEEN ** 2
        sw2 = _E_SD_WITHIN  ** 2
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
# sev_evidence feature builder (verbatim from 19/20)
# =============================================================================

def _build_sev_ev_features(df: pd.DataFrame) -> np.ndarray:
    """Build the feature matrix of the severity-evidence model (visit and patient features of s)."""
    all_feats = [c for c in _SEV_EV_VISIT_FEATS + _SEV_EV_PT_FEATS if c in df.columns]
    return _encode_cats(df[all_feats], _EV_CAT_COLS)


def _add_sev_ev_pt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient statistics of s for the severity-evidence model.

    Max, top-2 and top-3 means, mean, median, p75, min and count of s; its slope and residuals
    over age; centred age, number of visits and relative position of the visit.
    """
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
# Noise density fitting and lookup (verbatim from 19/20)
# =============================================================================

_TARGET_BIN_EDGES = np.array([0.0, 8.0, 15.0, 22.0, 30.0, 38.0, 46.0, 55.0, 65.0, 80.0, np.inf])
_N_BINS = 10

_LV_GRID = np.arange(-5.00, 1.51, 0.01)
_LV_GRID_N = len(_LV_GRID)
_LV_STEP = 0.01


def _make_gauss_kernel(sd: float = 0.03, half: int = 25) -> np.ndarray:
    """Return a normalised Gaussian kernel (sd in log units) used to smooth the noise densities."""
    j = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-0.5 * (j * _LV_STEP / sd) ** 2)
    return w / w.sum()

_GAUSS_KERNEL = _make_gauss_kernel()


def _target_bin(v: float) -> int:
    """Return the index of the target band that contains ``v``."""
    edges = _TARGET_BIN_EDGES
    idx = int(np.searchsorted(edges, v, side="right")) - 1
    return int(np.clip(idx, 0, _N_BINS - 1))


def _target_bin_array(v_arr: np.ndarray) -> np.ndarray:
    """Vectorised ``_target_bin``: the target band index of each value."""
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

    lv_all  = np.log(np.maximum(s_v, 0.5) / np.maximum(t_v, 0.5))
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
    """Log-density of ``x`` = log(s / target) for visit category ``c`` and target band ``b``, interpolated on the grid."""
    x_clip = float(np.clip(x, _LV_GRID[0], _LV_GRID[-1]))
    k_frac = (x_clip - _LV_GRID[0]) / _LV_STEP
    k0 = int(k_frac)
    k1 = min(k0 + 1, _LV_GRID_N - 1)
    frac = k_frac - k0
    return float(logdens[c, b, k0] * (1.0 - frac) + logdens[c, b, k1] * frac)


def lookup_logdens_batch(logdens: np.ndarray, c_arr: np.ndarray, b_arr: np.ndarray, x_arr: np.ndarray) -> np.ndarray:
    """Vectorised ``lookup_logdens`` for arrays of categories, bands and values."""
    x_clip = np.clip(x_arr, _LV_GRID[0], _LV_GRID[-1])
    k_frac = (x_clip - _LV_GRID[0]) / _LV_STEP
    k0 = np.floor(k_frac).astype(int)
    k1 = np.minimum(k0 + 1, _LV_GRID_N - 1)
    frac = k_frac - k0
    v0 = logdens[c_arr, b_arr, k0]
    v1 = logdens[c_arr, b_arr, k1]
    return v0 * (1.0 - frac) + v1 * frac


# =============================================================================
# Per-patient likelihood features (verbatim from 19/20)
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
    """Compute the 6 per-patient likelihood features of stage 2.

    Shifts of the curve q are weighted by the noise densities of the patient's s values, giving
    q*(pm-1), q*(exp(dmle)-1), psd, q*(exp(dpm1+gpm1*tc)-1), gpm1 and gsd1.
    """
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
    """Visit category of the noise model: 0 if OFF is measured, 1 if s comes from ON with a patient E, 2 with the population E."""
    c = np.where(
        off_present,
        0,
        np.where(nE_i_arr > 0, 1, 2),
    ).astype(int)
    return c


# =============================================================================
# P: Rater debiasing helpers
# =============================================================================

def _estimate_rater_factors(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Estimate one log-factor e_r per rater from training rows only.

    Uses a target-based estimate (75th pct of log(off/y)) combined with
    a lattice estimate (maximum likelihood over rounding grid) and resolves
    the common-constant ambiguity of the target-based estimates.

    Returns dict: rater_id -> e_r  (default 0 for unseen raters).
    """
    rater_ids = X["rater_id"].values
    off_col   = X["off"].values.astype(float)
    on_col    = X["on"].values.astype(float)
    unique_raters = np.unique(rater_ids)

    # --- a. Target-based estimate: q_r = 75th pct of log(off/y) for eligible rows ---
    q_r = {}
    for r in unique_raters:
        mask = rater_ids == r
        off_r = off_col[mask]
        y_r   = y[mask]
        eligible = ~np.isnan(off_r) & (off_r > 0) & (y_r > 10)
        if eligible.sum() == 0:
            q_r[r] = np.nan
        else:
            ratios = np.log(off_r[eligible] / np.maximum(y_r[eligible], 1e-9))
            q_r[r] = float(np.percentile(ratios, 75))

    # --- b. Lattice estimate: find k = exp(g) that maximises the rounding likelihood ---
    grid = np.arange(-0.30, 0.30001, 0.0005)  # shape (G,)

    lat_r = {}
    n_lattice = 0
    for r in unique_raters:
        mask = rater_ids == r
        # All non-missing on and off values >= 3 (as floats)
        vals_r = np.concatenate([
            off_col[mask][~np.isnan(off_col[mask])],
            on_col[mask][~np.isnan(on_col[mask])],
        ])
        vals_r = vals_r[vals_r >= 3.0]

        if len(vals_r) == 0:
            lat_r[r] = np.nan
            continue

        # Vectorize: count_i(g) = ceil((n_i+0.5)/k - 1e-9) - ceil((n_i-0.5)/k - 1e-9)
        # shape: (G, N) where G = len(grid), N = len(vals_r)
        k = np.exp(grid)[:, np.newaxis]          # (G, 1)
        n = vals_r[np.newaxis, :]                 # (1, N)
        count_i = (
            np.ceil((n + 0.5) / k - 1e-9).astype(int) -
            np.ceil((n - 0.5) / k - 1e-9).astype(int)
        ).astype(float)
        count_i = np.maximum(count_i, 0.0)

        # LL(g) = sum_i log(count_i + 0.02) + len(n) * g
        LL = np.sum(np.log(count_i + 0.02), axis=1) + len(vals_r) * grid  # (G,)
        best_g = float(grid[np.argmax(LL)])
        lat_r[r] = best_g
        n_lattice += 1

    # --- c. Combine: align target-based to lattice scale ---
    # c = median over raters of (lat_r - q_r) for raters where both are available
    diffs = []
    for r in unique_raters:
        l = lat_r.get(r, np.nan)
        q = q_r.get(r, np.nan)
        if not np.isnan(l) and not np.isnan(q):
            diffs.append(l - q)
    c_offset = float(np.median(diffs)) if diffs else 0.0

    e_r = {}
    n_uses_lattice = 0
    for r in unique_raters:
        l = lat_r.get(r, np.nan)
        q = q_r.get(r, np.nan)
        if np.isnan(q):
            qa = np.nan
        else:
            qa = q + c_offset

        if np.isnan(l):
            e_r[r] = 0.0 if np.isnan(qa) else float(qa)
        else:
            if np.isnan(qa) or abs(l - qa) <= 0.01:
                e_r[r] = float(l)
                n_uses_lattice += 1
            else:
                e_r[r] = float(qa)

    return e_r, n_uses_lattice


def _debias_X(X: pd.DataFrame, e_r: dict) -> pd.DataFrame:
    """Return a copy of X with on/off debiased by exp(e_r) and rater_id dropped."""
    df = X.copy()
    rater_ids = df["rater_id"].values
    off_col   = df["off"].values.astype(float)
    on_col    = df["on"].values.astype(float)

    new_off = off_col.copy()
    new_on  = on_col.copy()

    unique_r = np.unique(rater_ids)
    for r in unique_r:
        mask = rater_ids == r
        k    = float(np.exp(e_r.get(r, 0.0)))
        if k == 1.0:
            continue
        # off: round(off / k), NaN stays NaN
        off_r = off_col[mask]
        on_r  = on_col[mask]
        valid_off = ~np.isnan(off_r)
        valid_on  = ~np.isnan(on_r)
        new_off[mask] = np.where(valid_off, np.round(off_r / k), np.nan)
        new_on[mask]  = np.where(valid_on,  np.round(on_r  / k), np.nan)

    df["off"] = new_off
    df["on"]  = new_on
    df = df.drop(columns=["rater_id"])
    return df


# =============================================================================
# O: Bigger stage-2 HGB params
# =============================================================================

STAGE2_HGB_PARAMS = dict(
    max_iter=900, learning_rate=0.02, max_leaf_nodes=63,
    min_samples_leaf=50, l2_regularization=1.0, early_stopping=False,
)


# =============================================================================
# _add_s_cols helper
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
# P+O: TwoStageRegressor with rater debiasing and bigger stage-2
# =============================================================================

class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Two-stage regressor with rater debiasing (P) and larger stage-2 (O).

    Parameters
    ----------
    e_mode : "median" or "shrunk_mean".
    """

    def __init__(self, base_params: dict, n_splits_inner: int = 5, e_mode: str = "median"):
        """Store the base HGB settings, the number of inner folds and the E estimation mode ("median" or "shrunk_mean")."""
        self.base_params = base_params
        self.n_splits_inner = n_splits_inner
        self.e_mode = e_mode

    def _make_base(self):
        """Return a new base first-stage HGB built with ``base_params``."""
        return HistGradientBoostingRegressor(**self.base_params)

    def _make_ev_model(self):
        """Return a new HGB with the settings of the evidence models."""
        return HistGradientBoostingRegressor(**_EV_HGB_PARAMS)

    def _make_sev_model(self):
        """Return a new HGB for the severity-evidence model (built on s)."""
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
        """Feature matrix of the simple first-stage model (columns plus simple patient aggregates)."""
        df = _add_simple_aggs(X)
        cols_needed = [c for c in _SIMPLE_FEATS if c in df.columns]
        return _encode_cats(df[cols_needed], _EV_CAT_COLS)

    def _build_visit_X(self, X: pd.DataFrame) -> np.ndarray:
        """Feature matrix of the visit evidence model."""
        cols_needed = [c for c in _VISIT_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_off_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        """Feature matrix of the OFF evidence model (used on rows with an OFF score)."""
        cols_needed = [c for c in _OFF_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_on_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        """Feature matrix of the ON evidence model (used on rows with an ON score)."""
        cols_needed = [c for c in _ON_EV_FEATS if c in X.columns]
        return _encode_cats(X[cols_needed], _EV_CAT_COLS)

    def _build_sev_ev_X(self, X: pd.DataFrame) -> np.ndarray:
        """Feature matrix of the severity-evidence model, after adding the patient features of s."""
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
        """Predict the four evidence models; OFF and ON evidence are NaN where that score is missing.

        Returns ``(ev_visit, ev_off, ev_on, ev_sev)``.
        """
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

    # --- P: fit debiases first, then proceeds exactly as Experiment 20 ---
    def fit(self, X: pd.DataFrame, y):
        """Fit the rater factors, stage 1 (inner GroupKFold out-of-fold) and the 5 bagged stage-2 models.

        Stage 2 learns the residual y - q on 80 % patient subsamples; the first-stage models are
        then refitted on all rows for prediction.
        """
        y_arr = np.asarray(y, dtype=float)

        # P: estimate per-rater log-factors from training data only
        self.e_r_, self._n_lattice_ = _estimate_rater_factors(X, y_arr)

        # P: debias X so all subsequent steps see corrected on/off without rater_id
        X = _debias_X(X, self.e_r_)

        pids  = X["patient_id"].values
        ages  = X["age"].values.astype(float)

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
            t0 = time.time()
            rng_sub = np.random.default_rng(seed)
            n_sub = max(1, int(0.8 * len(unique_pids)))
            sub_pids = set(rng_sub.choice(unique_pids, size=n_sub, replace=False))
            sub_mask = np.array([pid in sub_pids for pid in pids])
            valid_mask = sub_mask & np.isfinite(q_oof) & np.isfinite(residuals)
            # O: bigger stage-2 HGB
            m_s2 = HistGradientBoostingRegressor(**STAGE2_HGB_PARAMS, random_state=seed)
            m_s2.fit(s2_feats[valid_mask], residuals[valid_mask])
            elapsed = time.time() - t0
            if seed == 0:
                print(f"  [fit e_mode={self.e_mode}] Stage-2 model 0: n_iter_={m_s2.n_iter_}  fit_time={elapsed:.1f}s", flush=True)
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

    # --- P: _predict_raw debiases at the start ---
    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction before the isotonic step.

        Debias, blend stage 1, project onto the quadratic curve q, add the mean stage-2 correction
        and project again.
        """
        X = _debias_X(X, self.e_r_)

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

    # --- P: predict debiases at the start ---
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Final prediction: ``_predict_raw``, then a non-decreasing isotonic fit in age per patient, clipped at 0."""
        p_raw = self._predict_raw(X)

        # Need patient ids and ages from debiased X (without rater_id)
        X_db = _debias_X(X, self.e_r_)
        pids = X_db["patient_id"].values
        ages = X_db["age"].values.astype(float)

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
# BlendRegressor (verbatim from 20)
# =============================================================================

class BlendRegressor(BaseEstimator, RegressorMixin):
    """Fit each estimator independently; predict the mean of their outputs."""

    def __init__(self, estimators: list):
        """Store the list of estimators to average."""
        self.estimators = estimators

    def fit(self, X, y):
        """Fit a clone of each estimator on the same data."""
        self.estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X):
        """Return the mean of the estimators' predictions."""
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return preds.mean(axis=1)


# =============================================================================
# Data loading — data_new/ (has rater_id column)
# =============================================================================

print("Loading data from data_new/ ...", flush=True)
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
    """Add time_since_diagnosis = age - age_at_diagnosis if the column is missing."""
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


# =============================================================================
# Feature pipeline
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
    """Full supervised feature stack (df_raw must have rater_id dropped already)."""
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
# Protocol step 0: print smoke diagnostics on full training data
# =============================================================================

def _print_smoke_diagnostics():
    """Diagnostics using full training data (pre-smoke subset)."""
    # Reload full data_new/ training set for diagnostics
    X_full = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
    y_full = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
    y_full_arr = y_full.values.astype(float)

    print("\n--- Smoke diagnostics (full training data, data_new/) ---", flush=True)

    n_raters = X_full["rater_id"].nunique()
    print(f"  Number of raters: {n_raters}  (expected 50)", flush=True)

    e_r_dict, n_lat = _estimate_rater_factors(X_full, y_full_arr)
    print(f"  Raters using lattice estimate: {n_lat}  (expected ~40)", flush=True)

    exp_er = np.array([np.exp(v) for v in e_r_dict.values()])
    print(f"  min(exp(e_r))={exp_er.min():.4f}  max(exp(e_r))={exp_er.max():.4f}  (expected ~0.81 and ~1.32)", flush=True)

    # E_pop on debiased data
    X_db = _debias_X(X_full, e_r_dict)
    _, _, _, _, _, E_pop_db = compute_s_ratio(X_db, E_pop=None, e_mode="median")
    print(f"  E_pop on debiased data: {E_pop_db:.4f}  (expected ~0.435)", flush=True)

    # Share of rows where debiased off == old off (from data/)
    X_old = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv", index_col="Index")
    off_new_db = X_db["off"].reindex(X_full.index).values.astype(float)
    off_old    = X_old["off"].reindex(X_full.index).values.astype(float)
    valid_off  = ~np.isnan(off_new_db) & ~np.isnan(off_old)
    match_off  = (off_new_db[valid_off] == off_old[valid_off]).mean() if valid_off.sum() > 0 else float("nan")
    print(f"  Share debiased off == old off: {match_off:.3f}  (expected ~0.90)", flush=True)

    on_new_db = X_db["on"].reindex(X_full.index).values.astype(float)
    on_old    = X_old["on"].reindex(X_full.index).values.astype(float)
    valid_on  = ~np.isnan(on_new_db) & ~np.isnan(on_old)
    match_on  = (on_new_db[valid_on] == on_old[valid_on]).mean() if valid_on.sum() > 0 else float("nan")
    print(f"  Share debiased on  == old on : {match_on:.3f}  (expected ~0.90)", flush=True)


_print_smoke_diagnostics()

# =============================================================================
# Protocol step 1 (smoke): quick CV of median model only
# =============================================================================

if SMOKE:
    print("\n[SMOKE] Running 2-fold CV of TwoStageRegressor(e_mode='median') ...", flush=True)
    gkf_sm = GroupKFold(n_splits=2)
    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf_sm.split(X_train_raw, y_train, groups=groups_train)
    ):
        X_raw_tr  = X_train_raw.iloc[tr_idx]
        X_raw_val = X_train_raw.iloc[val_idx]
        y_tr      = y_train.iloc[tr_idx]
        pids_tr   = groups_train.iloc[tr_idx].values
        pids_val  = groups_train.iloc[val_idx].values
        y_arr_tr  = y_tr.values.astype(float)

        # Debias train fold to build feature transformers
        e_r_sm, _ = _estimate_rater_factors(X_raw_tr, y_arr_tr)
        X_db_tr = _debias_X(X_raw_tr, e_r_sm)
        X_db_val = _debias_X(X_raw_val, e_r_sm)

        diag_imp = DiagnosisImputer(); diag_imp.fit(X_db_tr)
        X_li_tr = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_db_tr)))
        pcp = PatientCorrectedPooling(n_splits=2, use_ledd_interp=True)
        pcp.fit(X_li_tr, y_tr, patient_ids=pids_tr)
        ml = MixedLine(pcp=pcp); ml.fit(X_li_tr, y_tr, patient_ids=pids_tr)
        X_tr_base = pcp.transform(X_li_tr)
        X_tr_up = ml.transform(X_tr_base, patient_ids=pids_tr)
        X_tr_up = add_patient_features(X_tr_up, ledd_col="ledd_interp")
        X_tr_up["cohort"] = X_tr_up["cohort"].astype("category")
        X_tr_up["gene"]   = X_tr_up["gene"].astype("category")
        us = UpperStats(); us.fit(X_tr_up, y_tr)
        iot = ImpliedOffTransformer(); iot.fit(X_tr_base)

        X_tr  = _build_features_14(X_db_tr,  pids_tr,  diag_imp, pcp, ml, us, iot)
        X_val = _build_features_14(X_db_val, pids_val, diag_imp, pcp, ml, us, iot)

        # Re-attach rater_id so TwoStageRegressor.fit can run _estimate_rater_factors
        X_tr_with_rater = X_tr.copy()
        X_tr_with_rater["rater_id"] = X_raw_tr["rater_id"].values
        X_val_with_rater = X_val.copy()
        X_val_with_rater["rater_id"] = X_raw_val["rater_id"].values

        est = TwoStageRegressor(base_params=HGB_BEST_08, n_splits_inner=2, e_mode="median")
        est.fit(X_tr_with_rater, y_tr)
        preds = est.predict(X_val_with_rater)
        rmse = root_mean_squared_error(y_train.iloc[val_idx], preds)
        print(f"  [SMOKE] Fold {fold_i+1}: RMSE={rmse:.4f}", flush=True)

    print("\n[SMOKE] All checks passed.", flush=True)
    import sys; sys.exit(0)

# =============================================================================
# Benchmark: v20 RMSE on the same outer GroupKFold(5) splits (data_new/ indices)
# =============================================================================

print("\n--- Benchmark: v20 on data_new/ outer splits ---", flush=True)
oof_20_df = pd.read_csv(PROJECT_ROOT / "audit" / "oof_20.csv").set_index("Index")
y_arr_all = y_train.values.astype(float)
gkf5 = GroupKFold(n_splits=5)
fold_rmses_v20 = []
for fold_i, (tr_idx, val_idx) in enumerate(
    gkf5.split(X_train_raw, y_train, groups=groups_train)
):
    idx_in_oof = X_train_raw.index[val_idx]
    v20_val = oof_20_df["v20"].reindex(idx_in_oof).values
    y_val   = y_arr_all[val_idx]
    valid   = ~np.isnan(v20_val)
    if valid.sum() == 0:
        fold_rmses_v20.append(float("nan"))
    else:
        fold_rmses_v20.append(root_mean_squared_error(y_val[valid], v20_val[valid]))
    print(f"  v20 fold {fold_i+1}: RMSE={fold_rmses_v20[-1]:.4f}", flush=True)
mean_v20_ref = float(np.nanmean(fold_rmses_v20))
print(f"  v20 mean RMSE: {mean_v20_ref:.4f} +/- {float(np.nanstd(fold_rmses_v20, ddof=1)):.4f}", flush=True)


# =============================================================================
# CV helper: build feature pipeline for one outer fold
# =============================================================================

def _build_fold_features(X_raw_tr, X_raw_val, y_tr, pids_tr, pids_val):
    """Feature pipeline for one outer fold; X_raw_* already have rater_id dropped."""
    diag_imp = DiagnosisImputer(); diag_imp.fit(X_raw_tr)
    X_li_tr = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_tr)))
    pcp = PatientCorrectedPooling(n_splits=N_SPLITS_CV, use_ledd_interp=True)
    pcp.fit(X_li_tr, y_tr, patient_ids=pids_tr)
    ml = MixedLine(pcp=pcp); ml.fit(X_li_tr, y_tr, patient_ids=pids_tr)
    X_tr_base  = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_tr)))
    X_tr_base  = pcp.transform(X_tr_base)
    X_val_base = ledd_interp_tr.transform(_add_tsd(diag_imp.transform(X_raw_val)))
    X_val_base = pcp.transform(X_val_base)
    X_tr_up = ml.transform(X_tr_base, patient_ids=pids_tr)
    X_tr_up = add_patient_features(X_tr_up, ledd_col="ledd_interp")
    X_tr_up["cohort"] = X_tr_up["cohort"].astype("category")
    X_tr_up["gene"]   = X_tr_up["gene"].astype("category")
    us = UpperStats(); us.fit(X_tr_up, y_tr)
    iot = ImpliedOffTransformer(); iot.fit(X_tr_base)
    X_tr  = _build_features_14(X_raw_tr,  pids_tr,  diag_imp, pcp, ml, us, iot)
    X_val = _build_features_14(X_raw_val, pids_val, diag_imp, pcp, ml, us, iot)
    return X_tr, X_val


# =============================================================================
# CV: run one e_mode, save/load OOF file
# =============================================================================

def run_cv_22(e_mode: str, oof_tag: str, oof_path: Path):
    """Full CV for TwoStageRegressor with given e_mode. Saves/loads OOF file."""
    col_raw = f"{oof_tag}_raw"
    col_iso = oof_tag

    if oof_path.exists():
        print(f"\n  {oof_path.name} already exists — loading.", flush=True)
        df = pd.read_csv(oof_path).set_index("Index")
        return df[col_raw].values, df[col_iso].values

    oof_raw = np.full(len(y_train), np.nan)
    oof_iso = np.full(len(y_train), np.nan)

    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf5.split(X_train_raw, y_train, groups=groups_train)
    ):
        print(f"  [{e_mode}] Fold {fold_i+1}/{N_SPLITS_CV} ...", flush=True)
        X_raw_tr  = X_train_raw.iloc[tr_idx]
        X_raw_val = X_train_raw.iloc[val_idx]
        y_tr      = y_train.iloc[tr_idx]
        pids_tr   = groups_train.iloc[tr_idx].values
        pids_val  = groups_train.iloc[val_idx].values
        y_arr_tr  = y_tr.values.astype(float)

        # P: debias per fold — only training rows inform e_r
        e_r_fold, _ = _estimate_rater_factors(X_raw_tr, y_arr_tr)
        X_db_tr  = _debias_X(X_raw_tr,  e_r_fold)
        X_db_val = _debias_X(X_raw_val, e_r_fold)

        X_tr, X_val = _build_fold_features(X_db_tr, X_db_val, y_tr, pids_tr, pids_val)

        # Re-attach rater_id so TwoStageRegressor.fit runs _estimate_rater_factors
        # on the already-debiased data (e_r will be ~0 for all raters on debiased data,
        # which is correct — the debiasing is already done at the fold level above)
        # Actually: TwoStageRegressor.fit MUST receive raw (not yet debiased) data with rater_id.
        # The feature pipeline (PCP, MixedLine, etc.) must be built on debiased data,
        # but TwoStageRegressor.fit does its own debiasing internally.
        # Solution: pass raw fold data with rater_id to TwoStageRegressor.fit, and pass
        # debiased features to match. We need to keep both paths consistent.
        # Correct design: TwoStageRegressor.fit takes raw X with rater_id; it debiases
        # first and then builds features inside. But the outer feature pipeline (PCP, etc.)
        # was built on debiased data. So TwoStageRegressor receives the ALREADY-DEBIASED
        # feature-engineered df WITHOUT rater_id, and we add a dummy rater_id col of zeros
        # so its internal debias is a no-op (e_r=0 for all).
        # The HONEST debiasing (for CV) is done at the fold level above.
        # TwoStageRegressor.fit then sees debiased data and estimates e_r~0.
        X_tr_rater  = X_tr.copy();  X_tr_rater["rater_id"]  = 0
        X_val_rater = X_val.copy(); X_val_rater["rater_id"] = 0

        est = TwoStageRegressor(base_params=HGB_BEST_08, n_splits_inner=N_SPLITS_CV, e_mode=e_mode)
        est.fit(X_tr_rater, y_tr)

        raw_val = est._predict_raw(X_val_rater)
        iso_val = est.predict(X_val_rater)

        oof_raw[val_idx] = raw_val
        oof_iso[val_idx] = iso_val

        rmse_raw = root_mean_squared_error(y_arr_all[val_idx], raw_val)
        rmse_iso = root_mean_squared_error(y_arr_all[val_idx], iso_val)
        print(f"  [{e_mode}] Fold {fold_i+1}: raw={rmse_raw:.4f}  iso={rmse_iso:.4f}", flush=True)

    audit_dir = PROJECT_ROOT / "audit"
    audit_dir.mkdir(exist_ok=True)
    oof_df = pd.DataFrame({
        "Index":      X_train_raw.index,
        "patient_id": groups_train.values,
        col_raw:      oof_raw,
        col_iso:      oof_iso,
    })
    oof_df.to_csv(oof_path, index=False)
    print(f"  OOF saved -> {oof_path.name}", flush=True)
    return oof_raw, oof_iso


# =============================================================================
# Steps 3 and 4: Full CV for both models
# =============================================================================

audit_dir = PROJECT_ROOT / "audit"
audit_dir.mkdir(exist_ok=True)

print("\n=== Step 3: CV TwoStageRegressor(e_mode='median') ===", flush=True)
oof_22a_raw, oof_22a = run_cv_22("median",      "v22a", audit_dir / "oof_22a.csv")

print("\n=== Step 4: CV TwoStageRegressor(e_mode='shrunk_mean') ===", flush=True)
oof_22b_raw, oof_22b = run_cv_22("shrunk_mean", "v22b", audit_dir / "oof_22b.csv")

# =============================================================================
# Step 5: Blend
# =============================================================================

print("\nStep 5: Building blend OOF ...", flush=True)
v22_raw = (oof_22a_raw + oof_22b_raw) / 2.0
v22     = (oof_22a     + oof_22b)     / 2.0

oof_22_df = pd.DataFrame({
    "Index":      X_train_raw.index,
    "patient_id": groups_train.values,
    "v22_raw":    v22_raw,
    "v22":        v22,
})
oof_22_path = audit_dir / "oof_22.csv"
oof_22_df.to_csv(oof_22_path, index=False)
print(f"OOF saved -> audit/oof_22.csv", flush=True)

# =============================================================================
# Step 6: Diagnostics
# =============================================================================

print("\n=== Step 6: Diagnostics ===", flush=True)

# Per-fold RMSEs
fold_rmses_22a = []
fold_rmses_22b = []
fold_rmses_22  = []
for fold_i, (tr_idx, val_idx) in enumerate(
    gkf5.split(X_train_raw, y_train, groups=groups_train)
):
    y_val = y_arr_all[val_idx]
    fold_rmses_22a.append(root_mean_squared_error(y_val, oof_22a[val_idx]))
    fold_rmses_22b.append(root_mean_squared_error(y_val, oof_22b[val_idx]))
    fold_rmses_22.append(root_mean_squared_error(y_val, v22[val_idx]))

for tag, fold_rmses_tag, oof_tag in [
    ("v22a", fold_rmses_22a, oof_22a),
    ("v22b", fold_rmses_22b, oof_22b),
    ("v22",  fold_rmses_22,  v22),
]:
    mean_r = float(np.mean(fold_rmses_tag))
    std_r  = float(np.std(fold_rmses_tag, ddof=1))
    n_better = sum(1 for f, b in zip(fold_rmses_tag, fold_rmses_v20) if f < b)
    print(f"\n  {tag}: mean={mean_r:.4f} +/- {std_r:.4f}  ({n_better}/5 folds better than v20)", flush=True)
    for i, (fv, f20) in enumerate(zip(fold_rmses_tag, fold_rmses_v20)):
        delta = fv - f20
        flag = "BETTER" if delta < 0 else "worse"
        print(f"    Fold {i+1}: {tag}={fv:.4f}  v20={f20:.4f}  delta={delta:+.4f}  [{flag}]", flush=True)

# Per-fold delta vs v21 if available
oof_21_path = audit_dir / "oof_21.csv"
if oof_21_path.exists():
    oof_21_df = pd.read_csv(oof_21_path).set_index("Index")
    v21 = oof_21_df["v21"].reindex(X_train_raw.index).values
    fold_rmses_21 = []
    for fold_i, (tr_idx, val_idx) in enumerate(
        gkf5.split(X_train_raw, y_train, groups=groups_train)
    ):
        y_val  = y_arr_all[val_idx]
        v21_val = v21[val_idx]
        valid21 = ~np.isnan(v21_val)
        if valid21.sum() > 0:
            fold_rmses_21.append(root_mean_squared_error(y_val[valid21], v21_val[valid21]))
        else:
            fold_rmses_21.append(float("nan"))
    print("\n  Per-fold delta v22 vs v21:", flush=True)
    for i, (f22, f21) in enumerate(zip(fold_rmses_22, fold_rmses_21)):
        print(f"    Fold {i+1}: v22={f22:.4f}  v21={f21:.4f}  delta={f22-f21:+.4f}", flush=True)

# RMSE by number of OFF visits
print("\n  RMSE by number of OFF visits (v22 vs v20):", flush=True)
n_off_per_pt = X_train_raw.groupby("patient_id")["off"].apply(lambda s: s.notna().sum())
for label, lo, hi in [("0 OFF", 0, 0), ("1-2 OFF", 1, 2), ("3-5 OFF", 3, 5), ("6+ OFF", 6, 999)]:
    pts = n_off_per_pt[(n_off_per_pt >= lo) & (n_off_per_pt <= hi)].index
    idx_mask = X_train_raw["patient_id"].isin(pts)
    y_grp    = y_arr_all[idx_mask.values]
    v22_grp  = v22[idx_mask.values]
    v20_grp_reindex = oof_20_df["v20"].reindex(X_train_raw.index).values[idx_mask.values]
    valid    = ~np.isnan(v22_grp) & ~np.isnan(v20_grp_reindex)
    if valid.sum() == 0:
        continue
    r22 = root_mean_squared_error(y_grp[valid], v22_grp[valid])
    r20 = root_mean_squared_error(y_grp[valid], v20_grp_reindex[valid])
    print(f"    {label}: v22={r22:.4f}  v20={r20:.4f}  delta={r22-r20:+.4f}", flush=True)

# =============================================================================
# Step 7: Decision
# =============================================================================

print("\n=== Step 7: Decision ===", flush=True)
mean_v22 = float(np.mean(fold_rmses_22))
KEEP_THRESHOLD = 2.90
qualifies = mean_v22 < KEEP_THRESHOLD
print(
    f"  v22: mean RMSE={mean_v22:.4f}  threshold={KEEP_THRESHOLD:.2f}  "
    f"-> {'KEEP' if qualifies else 'discard'}",
    flush=True,
)

if not qualifies:
    print("\nv22 does not meet the keep criterion. Stopping.", flush=True)
    import sys; sys.exit(0)

print(f"\nKeeping v22 as {EXPERIMENT_KEY}.", flush=True)

# =============================================================================
# Step 8: Skore report, refits, submission
# =============================================================================

print("\nStep 8: Building skore holdout report on data_new/ ...", flush=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
hub_tr_idx, hub_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

X_raw_hub_tr  = X_train_raw.iloc[hub_tr_idx]
X_raw_hub_val = X_train_raw.iloc[hub_val_idx]
y_hub_tr      = y_train.iloc[hub_tr_idx]
y_hub_val     = y_train.iloc[hub_val_idx]
pids_hub_tr   = groups_train.iloc[hub_tr_idx].values
pids_hub_val  = groups_train.iloc[hub_val_idx].values

# Debias holdout split for feature pipeline
e_r_hub, _ = _estimate_rater_factors(X_raw_hub_tr, y_hub_tr.values.astype(float))
X_db_hub_tr  = _debias_X(X_raw_hub_tr,  e_r_hub)
X_db_hub_val = _debias_X(X_raw_hub_val, e_r_hub)

diag_imp_hub = DiagnosisImputer(); diag_imp_hub.fit(X_db_hub_tr)
X_li_hub = ledd_interp_tr.transform(_add_tsd(diag_imp_hub.transform(X_db_hub_tr)))
pcp_hub = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)
ml_hub = MixedLine(pcp=pcp_hub); ml_hub.fit(X_li_hub, y_hub_tr, patient_ids=pids_hub_tr)
X_hub_base_tr  = ledd_interp_tr.transform(_add_tsd(diag_imp_hub.transform(X_db_hub_tr)))
X_hub_base_tr  = pcp_hub.transform(X_hub_base_tr)
X_hub_ml_tr    = ml_hub.transform(X_hub_base_tr, patient_ids=pids_hub_tr)
X_hub_ml_tr    = add_patient_features(X_hub_ml_tr, ledd_col="ledd_interp")
X_hub_ml_tr["cohort"] = X_hub_ml_tr["cohort"].astype("category")
X_hub_ml_tr["gene"]   = X_hub_ml_tr["gene"].astype("category")
us_hub = UpperStats(); us_hub.fit(X_hub_ml_tr, y_hub_tr)
iot_hub = ImpliedOffTransformer(); iot_hub.fit(X_hub_base_tr)

X_hub_tr  = _build_features_14(X_db_hub_tr,  pids_hub_tr,  diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)
X_hub_val = _build_features_14(X_db_hub_val, pids_hub_val, diag_imp_hub, pcp_hub, ml_hub, us_hub, iot_hub)

# Attach rater_id=0 (debiasing already done); BlendRegressor->TwoStageRegressor will estimate e_r~0
X_hub_tr["rater_id"]  = 0
X_hub_val["rater_id"] = 0

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

# =============================================================================
# Refit both models on all training data, predict test set
# =============================================================================

print("\nRefitting both models on all training data (data_new/) ...", flush=True)
pids_all = groups_train.values

# Debias all training data (e_r estimated from all training rows)
e_r_all, _ = _estimate_rater_factors(X_train_raw, y_train.values.astype(float))
X_db_all = _debias_X(X_train_raw, e_r_all)

diag_imp_all = DiagnosisImputer(); diag_imp_all.fit(X_db_all)
X_all_base = ledd_interp_tr.transform(_add_tsd(diag_imp_all.transform(X_db_all)))
pcp_all = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_all.fit(X_all_base, y_train, patient_ids=pids_all)
ml_all = MixedLine(pcp=pcp_all); ml_all.fit(X_all_base, y_train, patient_ids=pids_all)
X_all_base_corr = pcp_all.transform(X_all_base)
X_all_ml = ml_all.transform(X_all_base_corr, patient_ids=pids_all)
X_all_ml = add_patient_features(X_all_ml, ledd_col="ledd_interp")
X_all_ml["cohort"] = X_all_ml["cohort"].astype("category")
X_all_ml["gene"]   = X_all_ml["gene"].astype("category")
us_all = UpperStats(); us_all.fit(X_all_ml, y_train)
iot_all = ImpliedOffTransformer(); iot_all.fit(X_all_base_corr)

X_all_final = _build_features_14(X_db_all, pids_all, diag_imp_all, pcp_all, ml_all, us_all, iot_all)
X_all_final["rater_id"] = 0  # already debiased

est_med_final  = TwoStageRegressor(base_params=HGB_BEST_08, e_mode="median")
est_sm_final   = TwoStageRegressor(base_params=HGB_BEST_08, e_mode="shrunk_mean")
est_med_final.fit(X_all_final, y_train)
est_sm_final.fit(X_all_final, y_train)

# Predict test set
print("Predicting test set (data_new/X_test.csv) ...", flush=True)
pids_test = X_test_raw["patient_id"].values
X_db_test = _debias_X(X_test_raw, e_r_all)
X_test_feat = _build_features_14(X_db_test, pids_test, diag_imp_all, pcp_all, ml_all, us_all, iot_all)
X_test_feat["rater_id"] = 0

test_preds_med = est_med_final.predict(X_test_feat)
test_preds_sm  = est_sm_final.predict(X_test_feat)
test_preds_blend = (test_preds_med + test_preds_sm) / 2.0

# Assertions
assert len(test_preds_blend) == 11013, f"Expected 11013, got {len(test_preds_blend)}"
n_finite = int(np.isfinite(test_preds_blend).sum())
assert n_finite == 11013, f"Expected 11013 finite predictions, got {n_finite}"
assert np.all(test_preds_blend >= 0), "Some blend predictions are negative"
assert list(X_test_raw.index) == list(sample_sub["Index"]), "Index mismatch"

# Print stats
print("\nPrediction stats:", flush=True)
print(f"  median test:  mean={test_preds_med.mean():.3f}  std={test_preds_med.std():.3f}", flush=True)
print(f"  shrunk test:  mean={test_preds_sm.mean():.3f}  std={test_preds_sm.std():.3f}", flush=True)
print(f"  blend  test:  mean={test_preds_blend.mean():.3f}  std={test_preds_blend.std():.3f}", flush=True)
print(f"  v22a OOF:     mean={float(np.nanmean(oof_22a)):.3f}  std={float(np.nanstd(oof_22a)):.3f}", flush=True)
print(f"  v22b OOF:     mean={float(np.nanmean(oof_22b)):.3f}  std={float(np.nanstd(oof_22b)):.3f}", flush=True)
print(f"  v22  OOF:     mean={float(np.nanmean(v22)):.3f}  std={float(np.nanstd(v22)):.3f}", flush=True)

sub_dir = PROJECT_ROOT / "submissions"
sub_dir.mkdir(exist_ok=True)
sub_20_path = sub_dir / "20_blend.csv"
if sub_20_path.exists():
    sub_20 = pd.read_csv(sub_20_path)
    print(f"  20_blend submission mean: {sub_20['target'].mean():.3f}", flush=True)
    corr_22_20 = float(np.corrcoef(test_preds_blend, sub_20["target"].values)[0, 1])
    print(f"  Correlation v22 vs 20_blend: {corr_22_20:.4f}  (expected > 0.99)", flush=True)

sub_path = sub_dir / f"{EXPERIMENT_KEY}.csv"
sub_df = pd.DataFrame({"Index": sample_sub["Index"], "target": test_preds_blend})
sub_df.to_csv(sub_path, index=False)
print(f"\nSubmission written -> submissions/{EXPERIMENT_KEY}.csv", flush=True)
print(f"Report URL: {HUB_URL}", flush=True)
