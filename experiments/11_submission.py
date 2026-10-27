# %% [markdown]
# # Experiment 11 — Kaggle submission (11_B pipeline)
#
# Fits the complete 11_B leakage-free pipeline on all available training data
# and predicts the test set.  No CV is repeated; hyperparameters are unchanged.
#
# Pipeline (Variant B):
#   LeddInterp → PatientCorrectedPooling (writes off_corr/on_corr) →
#   MixedLine → add_patient_features → UpperStats → PatientSmoothedRegressor
#
# Output: submissions/11_leakage_fix.csv  (Index, target — 11 013 rows)
#
# **Key:** 11_submission  |  **Date:** 2026-10-09  |  **Decision:** Kaggle submission file of 11_B
# **Results:** holdout (20 % of patients) 3.65 (Skore report `11_submission_B_holdout`); no CV,
# the pipeline is the one scored in 11_leakage_fix.py.
# **Skore Hub:** report key `11_submission_B_holdout` (id not listed in the report).
# **Run:** `python experiments/11_submission.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/11_submission.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see the report (report/rapport_SPS.pdf), How to run.
# A built-in check on 8 % of the patients runs first; `SMOKE_TEST=1` only skips the Skore report.

# %%
import math
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "11_submission"

SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# ── skore hub (skipped in smoke mode) ─────────────────────────────────────────
if not SMOKE:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline classes — identical to experiments/11_leakage_fix.py
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
    return slope, ym.mean() - slope * xm.mean()


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
        ledd_interp    = df["ledd"].values.astype(float).copy()
        ledd_missing   = np.where(np.isnan(df["ledd"].values), 1.0, 0.0)
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
            order        = np.argsort(a_p)
            a_sorted     = a_p[order]
            known_sorted = known[order]
            l_sorted     = l_p[order]
            known_ages   = a_sorted[known_sorted]
            known_ledds  = l_sorted[known_sorted]
            interp_vals  = np.interp(a_sorted, known_ages, known_ledds)
            ledd_interp[idx] = interp_vals[np.argsort(order)]
        df["ledd_interp"]  = ledd_interp
        df["ledd_missing"] = ledd_missing
        df["ledd_slope"]   = ledd_slope_col
        return df


class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction.  transform() writes off_corr and on_corr."""

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
        for tr_idx, val_idx in gkf.split(col_data, residuals, groups=patient_ids):
            tr_mask = has_score[tr_idx]
            if tr_mask.sum() < 10:
                continue
            tr_idx2 = tr_idx[tr_mask]
            model   = self._bias_corrector(
                col_data.iloc[tr_idx2].values, residuals[tr_idx2], cat_idx
            )
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
        off_vals   = df_raw["off"].values.astype(float)
        on_vals    = df_raw["on"].values.astype(float)
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
        return (
            np.where(np.isnan(off_vals), np.nan, off_vals + g_off),
            np.where(np.isnan(on_vals),  np.nan, on_vals  + g_on),
        )

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        off_corr, on_corr = self.get_corrections(df_raw)
        df["off_corr"] = off_corr   # exp-11 fix: expose per-visit corrected scores
        df["on_corr"]  = on_corr
        pids  = patient_ids if patient_ids is not None else df["patient_id"].values
        times = df["time_since_diagnosis"].values.astype(float)
        n     = len(df)
        fof = np.full(n, np.nan); fon = np.full(n, np.nan)
        fpf = np.full(n, np.nan); fpm = np.full(n, np.nan)
        nof = np.zeros(n, dtype=float); non_ = np.zeros(n, dtype=float)
        for pid in np.unique(pids):
            mask = pids == pid; idx = np.where(mask)[0]
            t = times[idx]; oc = off_corr[idx]; nc = on_corr[idx]
            vof = ~np.isnan(oc); nof_p = vof.sum()
            if nof_p >= 3:
                s, b = _linreg_slope_intercept(t, oc)
                fof[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif nof_p > 0:
                fof[idx] = np.nanmean(oc[vof])
            von = ~np.isnan(nc); non_p = von.sum()
            if non_p >= 3:
                s, b = _linreg_slope_intercept(t, nc)
                fon[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif non_p > 0:
                fon[idx] = np.nanmean(nc[von])
            av = np.concatenate([oc, nc]); at = np.concatenate([t, t]); vp = ~np.isnan(av)
            if vp.sum() >= 3:
                s, b = _linreg_slope_intercept(at, av)
                fpf[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif vp.sum() > 0:
                fpf[idx] = np.nanmean(av[vp])
            fpm[idx] = np.nanmean(av) if vp.sum() > 0 else np.nan
            nof[idx] = nof_p; non_[idx] = non_p
        df["off_corr_pt_fitted"]  = fof;  df["on_corr_pt_fitted"]   = fon
        df["pool_corr_pt_fitted"] = fpf;  df["pool_corr_pt_mean"]   = fpm
        df["pt_n_off_corr"]       = nof;  df["pt_n_on_corr"]        = non_
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
            vof = ~np.isnan(zo)
            d_off = float(np.mean(zo[vof] - fit_y[vof])) if vof.sum() > 0 else np.nan
            von = ~np.isnan(zn)
            d_on  = float(np.mean(zn[von] - fit_y[von])) if von.sum() > 0 else np.nan
            params[pid] = (float(a), float(s), d_off, d_on)
        return params

    @staticmethod
    def _empirical_prior(params):
        pids = list(params.keys())
        P  = np.array([params[p] for p in pids], dtype=float)
        mu = np.nanmean(P, axis=0); mu = np.where(np.isnan(mu), 0.0, mu)
        Pf = P.copy()
        for j in range(4):
            col = P[:, j]; cm = np.nanmean(col)
            Pf[np.isnan(col), j] = float(cm) if not np.isnan(cm) else 0.0
        S = np.cov(Pf.T, ddof=1); S = np.where(np.isnan(S), 0.0, S)
        return mu, S

    @staticmethod
    def _noise_variances(pids, times, z_off, z_on, params):
        ss_off, n_off = 0.0, 0; ss_on, n_on = 0.0, 0
        for pid in np.unique(pids):
            a, b, d_off, d_on = params[pid]
            if np.isnan(a): continue
            mask = pids == pid; t = times[mask]
            zo = z_off[mask]; zn = z_on[mask]; fit = a + b * t
            vof = ~np.isnan(zo)
            if vof.sum() > 0 and not np.isnan(d_off):
                res = zo[vof] - fit[vof] - d_off
                ss_off += float((res ** 2).sum()); n_off += int(vof.sum())
            von = ~np.isnan(zn)
            if von.sum() > 0 and not np.isnan(d_on):
                res = zn[von] - fit[von] - d_on
                ss_on += float((res ** 2).sum()); n_on += int(von.sum())
        return ss_off / max(n_off - 1, 1), ss_on / max(n_on - 1, 1)

    def _posterior(self, t, z_off_p, z_on_p):
        mu = self.mu_; S = self.S_; s2_off = self.s2_off_; s2_on = self.s2_on_
        rows_X = []; rows_z = []; rows_r = []
        vof = ~np.isnan(z_off_p)
        for i, (ti, zi) in enumerate(zip(t, z_off_p)):
            if vof[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 1.0, 0.0]); rows_z.append(zi); rows_r.append(s2_off)
        von = ~np.isnan(z_on_p)
        for i, (ti, zi) in enumerate(zip(t, z_on_p)):
            if von[i] and not np.isnan(ti):
                rows_X.append([1.0, ti, 0.0, 1.0]); rows_z.append(zi); rows_r.append(s2_on)
        S_reg = S + np.eye(4) * 1e-6; S_inv = np.linalg.inv(S_reg)
        if len(rows_X) == 0:
            return mu.copy(), np.linalg.inv(S_inv)
        X2 = np.array(rows_X); z2 = np.array(rows_z); r2 = np.array(rows_r)
        R_inv = np.diag(1.0 / r2)
        precision = S_inv + X2.T @ R_inv @ X2
        V = np.linalg.inv(precision + np.eye(4) * 1e-8)
        m = V @ (S_inv @ mu + X2.T @ R_inv @ z2)
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
        n = len(df_raw)
        ml_pred  = np.full(n, np.nan); ml_std   = np.full(n, np.nan)
        ml_slope = np.full(n, np.nan); ml_d_off = np.full(n, np.nan); ml_d_on = np.full(n, np.nan)
        for pid in np.unique(pids):
            mask = pids == pid; idx = np.where(mask)[0]
            t_p = times[idx]; zo_p = z_off[idx]; zn_p = z_on[idx]
            m, V = self._posterior(t_p, zo_p, zn_p)
            raw_pred_p = m[0] + m[1] * t_p
            pred_p = np.where(np.isnan(raw_pred_p), m[0], raw_pred_p)
            std_p  = np.array([
                np.sqrt(max(
                    np.array([1.0, ti, 0.0, 0.0]) @ V @
                    np.array([1.0, ti if not np.isnan(ti) else 0.0, 0.0, 0.0]), 0.0
                ))
                for ti in t_p
            ])
            ml_pred[idx] = pred_p;  ml_std[idx]   = std_p
            ml_slope[idx] = m[1];   ml_d_off[idx] = m[2];  ml_d_on[idx] = m[3]
        df["ml_pred"]  = ml_pred;  df["ml_std"]   = ml_std
        df["ml_slope"] = ml_slope; df["ml_d_off"] = ml_d_off; df["ml_d_on"] = ml_d_on
        return df


def _upper_stats_for_scores(raw_vals, times, beta, t_query, prefix):
    valid = ~np.isnan(raw_vals); n_val = valid.sum(); n_rows = len(t_query)
    empties = {f"{prefix}_{k}": np.full(n_rows, np.nan)
               for k in ["top1","mean2","mean_uh","mean_nlo","mean","count","spread","proj2","proj_uh"]}
    if n_val == 0:
        return empties
    v_vals = raw_vals[valid]; v_times = times[valid]; t_mean = v_times.mean()
    a = v_vals - beta * (v_times - t_mean)
    order = np.argsort(-a); a_sorted = a[order]; n = len(a_sorted)
    n_uh = math.ceil(n / 2)
    _t1 = a_sorted[0]; _m2 = a_sorted[:min(2, n)].mean(); _muh = a_sorted[:n_uh].mean()
    _mnlo = a_sorted[:-1].mean() if n > 1 else a_sorted[0]; _ma = a_sorted.mean(); _sp = _t1 - _ma
    out = {
        f"{prefix}_top1":    np.full(n_rows, _t1),
        f"{prefix}_mean2":   np.full(n_rows, _m2),
        f"{prefix}_mean_uh": np.full(n_rows, _muh),
        f"{prefix}_mean_nlo":np.full(n_rows, _mnlo),
        f"{prefix}_mean":    np.full(n_rows, _ma),
        f"{prefix}_count":   np.full(n_rows, float(n)),
        f"{prefix}_spread":  np.full(n_rows, _sp),
        f"{prefix}_proj2":   _m2  + beta * (t_query - t_mean),
        f"{prefix}_proj_uh": _muh + beta * (t_query - t_mean),
    }
    return out


class UpperStats(BaseEstimator, TransformerMixin):
    """Pooled within-patient OLS detrending; uses off_corr/on_corr when present."""

    def fit(self, X: pd.DataFrame, y):
        pids  = X["patient_id"].values
        times = (X["age"] - X["age_at_diagnosis"]).values.astype(float)
        y_arr = np.asarray(y, dtype=float)
        num, den = 0.0, 0.0
        for pid in np.unique(pids):
            mask  = pids == pid
            t_p   = times[mask]; y_p = y_arr[mask]
            valid = ~(np.isnan(t_p) | np.isnan(y_p))
            if valid.sum() < 2:
                continue
            tc = t_p[valid] - t_p[valid].mean()
            yc = y_p[valid] - y_p[valid].mean()
            num += float((tc * yc).sum()); den += float((tc ** 2).sum())
        self.beta_ = num / den if den > 0 else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df    = X.copy()
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
            mask = pids == pid; idx = np.where(mask)[0]; t_p = times[idx]
            for prefix, src in [
                ("off_up",      off_vals[idx]),
                ("on_up",       on_vals[idx]),
                ("off_corr_up", off_corr_vals[idx]),
                ("on_corr_up",  on_corr_vals[idx]),
            ]:
                res = _upper_stats_for_scores(src, t_p, beta, t_p, prefix)
                for cn, arr in res.items():
                    all_arrs[cn][idx] = arr
        for cn, arr in all_arrs.items():
            df[cn] = arr
        return df


class PatientSmoothedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base, alpha: float = 0.0):
        self.base = base; self.alpha = alpha

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
# Hyperparameters (unchanged from 11_B)
# ══════════════════════════════════════════════════════════════════════════════

HGB_BEST_08 = dict(
    learning_rate=0.03, max_iter=1000, max_leaf_nodes=63,
    min_samples_leaf=15, l2_regularization=0.5,
    early_stopping=True, random_state=0,
)
BEST_ALPHA_08 = 0.3

ledd_interp_tr = LeddInterp()   # stateless


def _add_tsd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


def _build_B_features(
    df_raw: pd.DataFrame,
    pids: np.ndarray,
    pcp: PatientCorrectedPooling,
    ml: MixedLine,
) -> pd.DataFrame:
    """Apply the 11_B base stack up to (not including) UpperStats.

    LeddInterp → PCP (writes off_corr/on_corr) → MixedLine → add_patient_features.
    Caller fits UpperStats on the returned frame, then calls us.transform().
    """
    df = ledd_interp_tr.transform(_add_tsd(df_raw))
    df = pcp.transform(df)
    df = ml.transform(df, patient_ids=pids)
    df = add_patient_features(df, ledd_col="ledd_interp")
    df["cohort"] = df["cohort"].astype("category")
    df["gene"]   = df["gene"].astype("category")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...", flush=True)
X_train_raw  = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train      = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw   = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")
sample_sub   = pd.read_csv(DATA_DIR / "sample_submission.csv")
groups_train = X_train_raw["patient_id"]
groups_test  = X_test_raw["patient_id"]

# ── guard: no train/test patient overlap ──────────────────────────────────────
overlap = set(groups_train.unique()) & set(groups_test.unique())
assert len(overlap) == 0, f"Train/test patient_id overlap: {overlap}"
print(f"  Train: {len(X_train_raw)} rows, {groups_train.nunique()} patients", flush=True)
print(f"  Test:  {len(X_test_raw)}  rows, {groups_test.nunique()}  patients", flush=True)
print(f"  No train/test patient overlap confirmed.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Smoke test: fit on 10% of patients, predict on unseen patients
# ══════════════════════════════════════════════════════════════════════════════

print("\n[SMOKE] Running pipeline smoke test...", flush=True)
rng = np.random.default_rng(42)
all_pids   = groups_train.unique()
n_smoke_tr = max(4, int(len(all_pids) * 0.08))
n_smoke_te = max(2, int(len(all_pids) * 0.04))
smoke_tr_pids = rng.choice(all_pids, size=n_smoke_tr, replace=False)
remaining     = np.setdiff1d(all_pids, smoke_tr_pids)
smoke_te_pids = rng.choice(remaining, size=n_smoke_te, replace=False)

mask_str = groups_train.isin(smoke_tr_pids)
mask_ste = groups_train.isin(smoke_te_pids)

Xs_tr_raw  = X_train_raw[mask_str]; ys_tr  = y_train[mask_str]; pids_str = groups_train[mask_str].values
Xs_te_raw  = X_train_raw[mask_ste];                              pids_ste = groups_train[mask_ste].values

X_li_str = ledd_interp_tr.transform(_add_tsd(Xs_tr_raw))
pcp_s = PatientCorrectedPooling(n_splits=2, use_ledd_interp=True)
pcp_s.fit(X_li_str, ys_tr, patient_ids=pids_str)
ml_s  = MixedLine(pcp=pcp_s); ml_s.fit(X_li_str, ys_tr, patient_ids=pids_str)
X_s_tr_b = _build_B_features(Xs_tr_raw, pids_str, pcp_s, ml_s)
us_s = UpperStats(); us_s.fit(X_s_tr_b, ys_tr)
X_s_tr = us_s.transform(X_s_tr_b)
X_s_te = us_s.transform(_build_B_features(Xs_te_raw, pids_ste, pcp_s, ml_s))

# Verify off_corr columns are distinct from raw off
assert "off_corr" in X_s_tr.columns, "off_corr column missing after PCP.transform()"
max_diff = float(np.nanmax(np.abs(X_s_tr["off_corr_up_top1"].values - X_s_tr["off_up_top1"].values)))
assert max_diff > 1e-9, "off_corr_up_top1 == off_up_top1: off_corr is still duplicating raw off"

est_s = PatientSmoothedRegressor(base=HistGradientBoostingRegressor(**HGB_BEST_08), alpha=BEST_ALPHA_08)
est_s.fit(X_s_tr, ys_tr)
preds_s = est_s.predict(X_s_te)
assert len(preds_s) == len(Xs_te_raw), "Smoke: prediction count mismatch"
assert np.all(np.isfinite(preds_s)), "Smoke: non-finite predictions"
print(f"  [SMOKE] PASSED — {len(preds_s)} unseen-patient predictions, off_corr distinct (max_diff={max_diff:.4f})", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Full fit on all training data
# ══════════════════════════════════════════════════════════════════════════════

print("\nFitting 11_B pipeline on all training data...", flush=True)
X_train_li = ledd_interp_tr.transform(_add_tsd(X_train_raw))

pcp_full = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
pcp_full.fit(X_train_li, y_train, patient_ids=groups_train.values)
print("  PatientCorrectedPooling fitted.", flush=True)

ml_full = MixedLine(pcp=pcp_full)
ml_full.fit(X_train_li, y_train, patient_ids=groups_train.values)
print("  MixedLine fitted.", flush=True)

X_train_b = _build_B_features(X_train_raw, groups_train.values, pcp_full, ml_full)
us_full   = UpperStats()
us_full.fit(X_train_b, y_train)
X_train_final = us_full.transform(X_train_b)
print(f"  Features built: {X_train_final.shape[1]} columns.", flush=True)

model_final = PatientSmoothedRegressor(
    base=HistGradientBoostingRegressor(**HGB_BEST_08),
    alpha=BEST_ALPHA_08,
)
model_final.fit(X_train_final, y_train)
print("  Final model fitted.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Test set feature construction and prediction
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding test features...", flush=True)
X_test_final = us_full.transform(
    _build_B_features(X_test_raw, groups_test.values, pcp_full, ml_full)
)
print(f"  Test features: {X_test_final.shape[1]} columns.", flush=True)
assert X_test_final.shape[1] == X_train_final.shape[1], \
    f"Column count mismatch: train={X_train_final.shape[1]}, test={X_test_final.shape[1]}"

preds_test = model_final.predict(X_test_final)

# ══════════════════════════════════════════════════════════════════════════════
# Assertions
# ══════════════════════════════════════════════════════════════════════════════

assert len(preds_test) == 11_013, \
    f"Expected 11013 predictions, got {len(preds_test)}"
assert list(X_test_raw.index) == list(sample_sub["Index"]), \
    "Index ordering does not match sample_submission.csv"
assert not np.any(np.isnan(preds_test)), "NaN predictions found"
assert np.all(np.isfinite(preds_test)),  "Non-finite predictions found"
print("  All assertions passed.", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Save submission CSV
# ══════════════════════════════════════════════════════════════════════════════

sub_dir = PROJECT_ROOT / "submissions"
sub_dir.mkdir(exist_ok=True)
sub_path = sub_dir / "11_leakage_fix.csv"

submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
submission.to_csv(sub_path, index=False, encoding="utf-8")
print(f"\nSubmission saved: {sub_path.resolve()}", flush=True)
print(f"  Rows: {len(submission)}  |  min={preds_test.min():.3f}  max={preds_test.max():.3f}  mean={preds_test.mean():.3f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Skore EstimatorReport — GroupShuffleSplit holdout (not a repeat of 5-fold CV)
# ══════════════════════════════════════════════════════════════════════════════

if not SMOKE:
    print("\nBuilding Skore holdout report...", flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    rep_tr_idx, rep_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    X_rep_raw_tr  = X_train_raw.iloc[rep_tr_idx]
    X_rep_raw_val = X_train_raw.iloc[rep_val_idx]
    y_rep_tr      = y_train.iloc[rep_tr_idx]
    y_rep_val     = y_train.iloc[rep_val_idx]
    pids_rep_tr   = groups_train.iloc[rep_tr_idx].values
    pids_rep_val  = groups_train.iloc[rep_val_idx].values

    X_li_rep_tr = ledd_interp_tr.transform(_add_tsd(X_rep_raw_tr))
    pcp_rep = PatientCorrectedPooling(n_splits=5, use_ledd_interp=True)
    pcp_rep.fit(X_li_rep_tr, y_rep_tr, patient_ids=pids_rep_tr)
    ml_rep  = MixedLine(pcp=pcp_rep)
    ml_rep.fit(X_li_rep_tr, y_rep_tr, patient_ids=pids_rep_tr)

    X_rep_tr_b  = _build_B_features(X_rep_raw_tr,  pids_rep_tr,  pcp_rep, ml_rep)
    X_rep_val_b = _build_B_features(X_rep_raw_val, pids_rep_val, pcp_rep, ml_rep)
    us_rep = UpperStats(); us_rep.fit(X_rep_tr_b, y_rep_tr)
    X_rep_tr_final  = us_rep.transform(X_rep_tr_b)
    X_rep_val_final = us_rep.transform(X_rep_val_b)

    rep = EstimatorReport(
        PatientSmoothedRegressor(
            base=HistGradientBoostingRegressor(**HGB_BEST_08),
            alpha=BEST_ALPHA_08,
        ),
        X_train=X_rep_tr_final, y_train=y_rep_tr,
        X_test=X_rep_val_final,  y_test=y_rep_val,
    )
    skore_key = "11_submission_B_holdout"
    project.put(skore_key, rep)
    HUB_URL = f"https://skore.probabl.ai/sps/bobathon-esilv/estimators/"
    print(f"  Skore report pushed: key='{skore_key}'", flush=True)
    print(f"  Hub URL: {HUB_URL}", flush=True)

print("\nDone.", flush=True)
