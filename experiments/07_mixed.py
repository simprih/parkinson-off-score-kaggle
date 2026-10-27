# %% [markdown]
# # Experiment 07 — mixed
#
# **Key:** 07_mixed  |  **Date:** 2026-10-08  |  **Kept:** yes, replaced by 08
# **Idea:** Bayesian linear mixed model per patient (MixedLine transformer).
# **CV (GroupKFold 5):** 4.05
# Posterior line values let ON measurements inform the patient slope without biasing the intercept.
# **CV detail (run output):** variant B (MixedLine features + HGB) 4.0548 ± 0.0439; control A 4.2643 ± 0.0574.
# **Skore Hub:** https://skore.probabl.ai/sps/bobathon-esilv/estimators/57212 (holdout report of variant B);
# grouped CV reports: cross-validations/57200 (A) and cross-validations/57206 (B).
# **Run:** `python experiments/07_mixed.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/07_mixed.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see the report (report/rapport_SPS.pdf), How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "07_mixed"

# %% — Open skore Hub project
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% — Helpers (unchanged from 05)
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
    """Patient-level aggregated features identical to experiment 03."""
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


# %% — PatientCorrectedPooling (unchanged from 05)
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


# %% — MixedLine transformer
class MixedLine(BaseEstimator, TransformerMixin):
    """Empirical-Bayes linear mixed model over corrected OFF/ON measurements.

    For each patient the latent parameter vector is theta = [a, b, d_off, d_on]:
    * a, b  : intercept and slope of target vs t (t = age - age_at_diagnosis)
    * d_off : patient-level offset in corrected OFF scores
    * d_on  : patient-level offset in corrected ON scores

    Measurement model
    -----------------
    OFF visit: z_off[i] ~ a + b*t[i] + d_off  +  eps_off,  Var = s2_off
    ON  visit: z_on[i]  ~ a + b*t[i] + d_on   +  eps_on,   Var = s2_on

    Prior (estimated from training patients)
    ----------------------------------------
    theta ~ N(mu, S)  where mu and S are the empirical mean and covariance of
    the per-patient OLS estimates.

    Posterior per patient
    ---------------------
    Given stacked observation vector z and design matrix X (rows [1,t,1,0] or
    [1,t,0,1]) with diagonal noise R:
        V = inv(inv(S) + X' R^{-1} X)
        m = V (inv(S) mu + X' R^{-1} z)
    Prediction at each visit: m[0] + m[1]*t.

    Features added
    --------------
    ml_pred, ml_std, ml_slope, ml_d_off, ml_d_on
    """

    def __init__(self, pcp: PatientCorrectedPooling):
        self.pcp = pcp

    # ------------------------------------------------------------------ #
    # Internal: per-patient sufficient statistics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _patient_params(pids, times, z_off, z_on, y):
        """Compute (a, b, d_off, d_on) for every patient in this fold."""
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
        return params   # pid -> (a, b, d_off, d_on)

    @staticmethod
    def _empirical_prior(params: dict):
        """Return mu (4,) and S (4,4) from per-patient parameter dict."""
        pids = list(params.keys())
        P = np.array([params[p] for p in pids], dtype=float)   # (N, 4)
        mu = np.nanmean(P, axis=0)                              # (4,)
        # Guard: replace NaN mu entries (all-missing column) with 0
        mu = np.where(np.isnan(mu), 0.0, mu)

        # Replace each missing entry with the column mean for covariance
        P_filled = P.copy()
        for j in range(4):
            col = P[:, j]
            col_mean = np.nanmean(col)
            fill_val = float(col_mean) if not np.isnan(col_mean) else 0.0
            P_filled[np.isnan(col), j] = fill_val

        S = np.cov(P_filled.T, ddof=1)   # (4, 4)
        # Replace any remaining NaN in S with 0 (diagonal already regularised)
        S = np.where(np.isnan(S), 0.0, S)
        return mu, S

    @staticmethod
    def _noise_variances(pids, times, z_off, z_on, params):
        """Pooled within-patient residual variances for OFF and ON."""
        ss_off, n_off = 0.0, 0
        ss_on,  n_on  = 0.0, 0
        unique_pids = np.unique(pids)
        for pid in unique_pids:
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

    # ------------------------------------------------------------------ #
    # Posterior per patient
    # ------------------------------------------------------------------ #

    def _posterior(self, t, z_off_p, z_on_p):
        """Return (m, V) for a single patient given its measurement vectors."""
        mu = self.mu_
        S  = self.S_
        s2_off = self.s2_off_
        s2_on  = self.s2_on_

        rows_X = []
        rows_z = []
        rows_r = []

        valid_off = ~np.isnan(z_off_p)
        for i, (ti, zi) in enumerate(zip(t, z_off_p)):
            if valid_off[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 1.0, 0.0])
                rows_z.append(zi)
                rows_r.append(s2_off)

        valid_on = ~np.isnan(z_on_p)
        for i, (ti, zi) in enumerate(zip(t, z_on_p)):
            if valid_on[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 0.0, 1.0])
                rows_z.append(zi)
                rows_r.append(s2_on)

        # Regularise S to ensure invertibility
        S_reg = S + np.eye(4) * 1e-6
        S_inv = np.linalg.inv(S_reg)

        if len(rows_X) == 0:
            # no measurements: posterior = prior
            V = np.linalg.inv(S_inv)
            m = mu.copy()
            return m, V

        X  = np.array(rows_X)    # (K, 4)
        z  = np.array(rows_z)    # (K,)
        r  = np.array(rows_r)    # (K,)
        R_inv = np.diag(1.0 / r)  # (K, K)

        precision = S_inv + X.T @ R_inv @ X
        V = np.linalg.inv(precision + np.eye(4) * 1e-8)
        m = V @ (S_inv @ mu + X.T @ R_inv @ z)
        return m, V

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #

    def fit(self, df_raw, y, patient_ids=None):
        """Estimate prior and noise variances from training fold."""
        if patient_ids is None:
            raise ValueError("patient_ids is required")

        z_off, z_on = self.pcp.get_corrections(df_raw)
        pids  = np.asarray(patient_ids)
        times = df_raw["time_since_diagnosis"].values.astype(float)
        y_arr = np.asarray(y, dtype=float)

        params = self._patient_params(pids, times, z_off, z_on, y_arr)
        mu, S  = self._empirical_prior(params)
        s2_off, s2_on = self._noise_variances(pids, times, z_off, z_on, params)

        self.mu_    = mu
        self.S_     = S
        self.s2_off_ = max(s2_off, 1e-6)
        self.s2_on_  = max(s2_on,  1e-6)
        return self

    def transform(self, df_raw, patient_ids=None):
        """Add ml_pred, ml_std, ml_slope, ml_d_off, ml_d_on to df_raw."""
        df   = df_raw.copy()
        pids = np.asarray(patient_ids if patient_ids is not None else df_raw["patient_id"].values)
        z_off, z_on = self.pcp.get_corrections(df_raw)
        times = df_raw["time_since_diagnosis"].values.astype(float)
        n     = len(df_raw)

        ml_pred  = np.full(n, np.nan)
        ml_std   = np.full(n, np.nan)
        ml_slope = np.full(n, np.nan)
        ml_d_off = np.full(n, np.nan)
        ml_d_on  = np.full(n, np.nan)

        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            t_p  = times[idx]
            zo_p = z_off[idx]
            zn_p = z_on[idx]

            m, V = self._posterior(t_p, zo_p, zn_p)

            raw_pred_p = m[0] + m[1] * t_p
            # Fall back to prior intercept where time is unknown
            pred_p = np.where(np.isnan(raw_pred_p), m[0], raw_pred_p)
            # posterior variance of prediction at each visit:
            # Var(m[0] + m[1]*t) = [1, t, 0, 0] V [1, t, 0, 0]'
            std_p  = np.array([
                np.sqrt(max(np.array([1.0, ti, 0.0, 0.0]) @ V @ np.array([1.0, ti if not np.isnan(ti) else 0.0, 0.0, 0.0]), 0.0))
                for ti in t_p
            ])

            ml_pred[idx]  = pred_p
            ml_std[idx]   = std_p
            ml_slope[idx] = m[1]
            ml_d_off[idx] = m[2]
            ml_d_on[idx]  = m[3]

        df["ml_pred"]  = ml_pred
        df["ml_std"]   = ml_std
        df["ml_slope"] = ml_slope
        df["ml_d_off"] = ml_d_off
        df["ml_d_on"]  = ml_d_on
        return df


# %% — Load data
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

groups_train = X_train_raw["patient_id"]

gkf       = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_raw, y_train, groups=groups_train))


# %% — Feature builders
def _add_tsd(df):
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


def prepare_A(df: pd.DataFrame) -> pd.DataFrame:
    """Variant B of experiment 05: 03 patient features only."""
    X = _add_tsd(df)
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"]   = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


def prepare_B(df_raw, pcp, ml, pids):
    """05 corrected features + MixedLine features."""
    df = _add_tsd(df_raw)
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df)
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    df = df.drop(columns=["patient_id"])
    return df


# %% — Model A (control): variant B of 05
X_train_A = prepare_A(X_train_raw)
model_A    = HistGradientBoostingRegressor(random_state=0)
report_A   = skore.evaluate(model_A, X_train_A, y_train, splitter=cv_splits)

# %% — Models B and C: cross-validation with per-fold PCP + MixedLine
rmse_B, rmse_C = [], []
sanity_patients = []

for fold_i, (fold_tr_idx, fold_val_idx) in enumerate(cv_splits):
    X_tr_raw = _add_tsd(X_train_raw.iloc[fold_tr_idx])
    X_val_raw = _add_tsd(X_train_raw.iloc[fold_val_idx])
    y_tr  = y_train.iloc[fold_tr_idx]
    y_val = y_train.iloc[fold_val_idx]
    pids_tr  = groups_train.iloc[fold_tr_idx].values
    pids_val = groups_train.iloc[fold_val_idx].values

    pcp = PatientCorrectedPooling(n_splits=5)
    pcp.fit(X_tr_raw, y_tr, patient_ids=pids_tr)

    ml = MixedLine(pcp=pcp)
    ml.fit(X_tr_raw, y_tr, patient_ids=pids_tr)

    if fold_i == 0:
        # Print sanity check from first fold
        print("\n=== Sanity check (fold 0) ===")
        print(f"mu        = {ml.mu_}")
        print(f"sqrt(diag(S)) = {np.sqrt(np.diag(ml.S_))}")
        print(f"s2_off    = {ml.s2_off_:.4f}")
        print(f"s2_on     = {ml.s2_on_:.4f}")

        # Pick 3 validation patients
        unique_val_pids = np.unique(pids_val)[:3]
        z_off_tr, z_on_tr = pcp.get_corrections(X_tr_raw)
        times_tr = X_tr_raw["time_since_diagnosis"].values.astype(float)
        y_tr_arr = y_tr.values.astype(float)
        params_tr = MixedLine._patient_params(pids_tr, times_tr, z_off_tr, z_on_tr, y_tr_arr)

        z_off_val, z_on_val = pcp.get_corrections(X_val_raw)
        times_val = X_val_raw["time_since_diagnosis"].values.astype(float)
        val_pids_arr = pids_val

        for pid in unique_val_pids:
            mask_v = val_pids_arr == pid
            idx_v  = np.where(mask_v)[0]
            t_p    = times_val[idx_v]
            zo_p   = z_off_val[idx_v]
            zn_p   = z_on_val[idx_v]
            n_off_p = int((~np.isnan(zo_p)).sum())
            n_on_p  = int((~np.isnan(zn_p)).sum())
            y_p     = y_val.values[idx_v]

            prior_pred = float(np.mean(ml.mu_[0] + ml.mu_[1] * t_p))
            m_post, V_post = ml._posterior(t_p, zo_p, zn_p)
            post_pred = float(np.mean(m_post[0] + m_post[1] * t_p))
            mean_tgt  = float(np.nanmean(y_p))

            print(f"  pid={pid}: n_off={n_off_p}, n_on={n_on_p}, "
                  f"prior_pred={prior_pred:.2f}, post_pred={post_pred:.2f}, "
                  f"mean_target={mean_tgt:.2f}")
            sanity_patients.append(dict(pid=pid, n_off=n_off_p, n_on=n_on_p,
                                        prior_pred=prior_pred, post_pred=post_pred,
                                        mean_target=mean_tgt))

    # B features
    X_tr_B  = prepare_B(X_tr_raw,  pcp, ml, pids_tr)
    X_val_B = prepare_B(X_val_raw, pcp, ml, pids_val)

    m_B = HistGradientBoostingRegressor(random_state=0)
    m_B.fit(X_tr_B, y_tr)
    rmse_B.append(root_mean_squared_error(y_val, m_B.predict(X_val_B)))

    # C: MixedLine prediction alone
    z_off_val2, z_on_val2 = pcp.get_corrections(X_val_raw)
    times_val2 = X_val_raw["time_since_diagnosis"].values.astype(float)
    val_pids2  = pids_val
    pred_C = np.full(len(y_val), np.nan)
    for pid in np.unique(val_pids2):
        mask = val_pids2 == pid
        idx  = np.where(mask)[0]
        t_p  = times_val2[idx]
        m_post, _ = ml._posterior(t_p, z_off_val2[idx], z_on_val2[idx])
        raw_pred = m_post[0] + m_post[1] * t_p
        # Fall back to prior intercept where time is unknown
        pred_C[idx] = np.where(np.isnan(raw_pred), m_post[0], raw_pred)
    rmse_C.append(root_mean_squared_error(y_val.values, pred_C))

# ---- Sanity gate ----
rmse_C_mean = np.mean(rmse_C)
if rmse_C_mean > 8:
    raise RuntimeError(f"SANITY FAIL: C RMSE mean = {rmse_C_mean:.4f} > 8 — likely a bug.")

rmse_B_mean, rmse_B_std = float(np.mean(rmse_B)), float(np.std(rmse_B))
rmse_C_std = float(np.std(rmse_C))

print(f"\nB CV RMSE: {rmse_B_mean:.4f} ± {rmse_B_std:.4f}")
print(f"C CV RMSE: {rmse_C_mean:.4f} ± {rmse_C_std:.4f}")

# %% — Build full-data PCP + MixedLine for skore reports
X_train_tsd = _add_tsd(X_train_raw)
X_test_tsd  = _add_tsd(X_test_raw)

pcp_full = PatientCorrectedPooling(n_splits=5)
pcp_full.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

ml_full = MixedLine(pcp=pcp_full)
ml_full.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

X_train_B = prepare_B(X_train_tsd, pcp_full, ml_full, groups_train.values)
X_test_B  = prepare_B(X_test_tsd,  pcp_full, ml_full, X_test_tsd["patient_id"].values)

cv_splits_B = list(gkf.split(X_train_B, y_train, groups=groups_train))

model_B  = HistGradientBoostingRegressor(random_state=0)
report_B = skore.evaluate(model_B, X_train_B, y_train, splitter=cv_splits_B)

# ---- Summary ----
summary_A = report_A.metrics.summarize().frame()
col_mean_A = [c for c in summary_A.columns if c.endswith("_mean")][0]
col_std_A  = [c for c in summary_A.columns if c.endswith("_std")][0]
mean_A = float(summary_A.loc["rmse", col_mean_A])
std_A  = float(summary_A.loc["rmse", col_std_A])

summary_B = report_B.metrics.summarize().frame()
col_mean_B = [c for c in summary_B.columns if c.endswith("_mean")][0]
col_std_B  = [c for c in summary_B.columns if c.endswith("_std")][0]
mean_B_skore = float(summary_B.loc["rmse", col_mean_B])
std_B_skore  = float(summary_B.loc["rmse", col_std_B])

print(f"\nA (control)     RMSE: {mean_A:.4f} ± {std_A:.4f}")
print(f"B (MixedLine)   RMSE: {mean_B_skore:.4f} ± {std_B_skore:.4f}  (manual: {rmse_B_mean:.4f} ± {rmse_B_std:.4f})")
print(f"C (ML-direct)   RMSE: {rmse_C_mean:.4f} ± {rmse_C_std:.4f}")

# %% — Push comparison to Hub (ComparisonReport cannot be put; push individual reports)
from skore import ComparisonReport  # noqa: E402

project.put("07_mixed_A", report_A)
project.put("07_mixed_B", report_B)
comparison = ComparisonReport({
    "A_control": report_A,
    "B_mixed_features": report_B,
})
print(f"Pushed 07_mixed_A: https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/")
print(f"Pushed 07_mixed_B: https://skore.probabl.ai/sps/bobathon-esilv/cross-validations/")
print("Comparison (local):")
print(comparison)

# %% — Best variant: if B beats A, produce holdout report and submission
best_rmse = min(mean_A, mean_B_skore, rmse_C_mean)
best_name = (
    "A" if best_rmse == mean_A else
    "B" if best_rmse == mean_B_skore else "C"
)
print(f"\nBest variant: {best_name} with RMSE {best_rmse:.4f}")

if best_rmse < mean_A:
    from skore import EstimatorReport  # noqa: E402

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    X_hold_tr_raw = _add_tsd(X_train_raw.iloc[hold_tr_idx])
    X_hold_val_raw = _add_tsd(X_train_raw.iloc[hold_val_idx])
    y_hold_tr  = y_train.iloc[hold_tr_idx]
    y_hold_val = y_train.iloc[hold_val_idx]
    pids_hold_tr  = groups_train.iloc[hold_tr_idx].values
    pids_hold_val = groups_train.iloc[hold_val_idx].values

    pcp_hold = PatientCorrectedPooling(n_splits=5)
    pcp_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)

    ml_hold = MixedLine(pcp=pcp_hold)
    ml_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)

    X_hold_tr_B  = prepare_B(X_hold_tr_raw,  pcp_hold, ml_hold, pids_hold_tr)
    X_hold_val_B = prepare_B(X_hold_val_raw, pcp_hold, ml_hold, pids_hold_val)

    report_final = EstimatorReport(
        HistGradientBoostingRegressor(random_state=0),
        X_train=X_hold_tr_B,
        y_train=y_hold_tr,
        X_test=X_hold_val_B,
        y_test=y_hold_val,
    )
    project.put("07_mixed_final", report_final)
    print("Pushed 07_mixed_final to hub")

    # Fit on all train, predict test
    model_final = HistGradientBoostingRegressor(random_state=0)
    model_final.fit(X_train_B, y_train)
    preds_test = model_final.predict(X_test_B)

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission.to_csv(submissions_dir / "07_mixed.csv", index=False)
    print(f"Submission written: {len(submission)} rows -> submissions/07_mixed.csv")
else:
    print("No variant beats A — skipping holdout report and submission.")
