# %% [markdown]
# # Experiment 06 — response
#
# **Key:** 06_response  |  **Date:** 2026-10-08  |  **Kept:** partly
# **Idea:** Patient-specific drug-response factor (ON = OFF − r_i × effect(dose, time)).
# **CV (GroupKFold 5):** not scored
# The drug-response idea evolved into the E-model used from experiment 18 onwards.
# **Skore Hub:** report keys `06_response_A`, `06_response_B`, `06_response_final` (ids not listed in the README).
# **Run:** `python experiments/06_response.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/06_response.py`).
# Reads data/ (first version of the files, before 9 October); needs the Skore key (.skore),
# see README, How to run.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import skore
from skore import login
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.load_skore_credentials import load_skore_credentials  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_KEY = "06_response"

# %% [markdown]
# ## Open skore Hub project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Shared helpers (from 03 / 05)

# %%
def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray):
    """Simple OLS; returns (NaN, NaN) when fewer than 2 paired observations."""
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
                f"{col}_pt_mean": "mean",
                f"{col}_pt_median": "median",
                f"{col}_pt_std": "std",
                f"{col}_pt_count": lambda s: s.notna().sum(),
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
        df[f"{score_col}_pt_slope"] = df["patient_id"].map(slopes)
        df[f"{score_col}_pt_intercept"] = df["patient_id"].map(intercepts)
        df[f"{score_col}_pt_fitted"] = pd.Series(fitted)
    df["off_dev"] = df["off"] - df["off_pt_mean"]
    df["on_dev"] = df["on"] - df["on_pt_mean"]
    df["pt_n_visits"] = df.groupby("patient_id")["patient_id"].transform("count")
    df["pt_visit_rank"] = df.groupby("patient_id")["time_since_diagnosis"].rank(
        method="first", na_option="keep"
    )
    return df


# %% [markdown]
# ## PatientResponse transformer
#
# Steps (all learned on training fold only; OOF cross-fitting for training rows):
# 1. Population effect curves: h_off = HGB(time_since_intake_off, ledd) -> (target - off)
#                               h_on  = HGB(time_since_intake_on,  ledd) -> (target - on)
# 2. Patient response r_i: shrinkage RTO on visits with both off and on
# 3. Corrected measurements: off_corr = off + r_i * h_off
#                             on_corr  = on  + r_i * h_on
# 4. Weighted OLS pooled line per patient (time = age - age_at_diagnosis)
# 5. Features: r_i, h_off, h_on, r_i*h_off, r_i*h_on, off_corr, on_corr,
#              pooled_line_fitted, n_corr_measurements

# %%
_H_FEATURES_OFF = ["time_since_intake_off", "ledd"]
_H_FEATURES_ON  = ["time_since_intake_on",  "ledd"]


class PatientResponse(BaseEstimator, TransformerMixin):
    """Patient-specific drug-response features via cross-fitting.

    Parameters
    ----------
    n_splits : int
        Inner cross-fitting folds (grouped by patient_id).
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fit_hgb(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
        """Fit a small HGB; NaN in X is handled natively."""
        m = HistGradientBoostingRegressor(max_iter=50, random_state=0)
        m.fit(X, y)
        return m

    def _cross_fit_h(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        score_col: str,
        feat_cols: list,
        patient_ids: np.ndarray,
    ) -> np.ndarray:
        """OOF predictions of (target - score_col) from feat_cols.

        Predicted for every row (even where score_col is NaN) using the fold
        model fitted on rows where score_col is observed.
        """
        n = len(df)
        oof = np.full(n, np.nan)
        residuals = y.values - df[score_col].values.astype(float)  # NaN where missing
        has_score = ~np.isnan(df[score_col].values.astype(float))
        X_feat = df[feat_cols].values.astype(float)  # NaN kept (HGB handles it)

        gkf = GroupKFold(n_splits=self.n_splits)
        for tr_idx, val_idx in gkf.split(X_feat, residuals, groups=patient_ids):
            tr_mask = has_score[tr_idx]
            if tr_mask.sum() < 5:
                continue
            model = self._fit_hgb(X_feat[tr_idx][tr_mask], residuals[tr_idx][tr_mask])
            oof[val_idx] = model.predict(X_feat[val_idx])

        return oof

    def _fit_full_h(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        score_col: str,
        feat_cols: list,
    ) -> HistGradientBoostingRegressor:
        """Full-data HGB for score_col -> (target - score_col)."""
        residuals = y.values - df[score_col].values.astype(float)
        has_score = ~np.isnan(df[score_col].values.astype(float))
        X_feat = df[feat_cols].values.astype(float)
        return self._fit_hgb(X_feat[has_score], residuals[has_score])

    @staticmethod
    def _compute_r_i(
        df: pd.DataFrame,
        h_off: np.ndarray,
        h_on: np.ndarray,
        patient_ids: np.ndarray,
        k_shrink: float,
    ) -> np.ndarray:
        """Per-patient response r_i = RTO regression shrunk toward 1.

        Only visits with both off and on contribute.  Patients with no such
        visit get r_i = 1.
        """
        off_vals = df["off"].values.astype(float)
        on_vals  = df["on"].values.astype(float)

        # d = h_on - h_off per visit
        d = h_on - h_off  # shape (n,)

        n = len(df)
        r_out = np.ones(n)

        unique_pids = np.unique(patient_ids)
        for pid in unique_pids:
            mask = patient_ids == pid
            idx = np.where(mask)[0]
            # visits where both off and on are observed
            both = ~np.isnan(off_vals[idx]) & ~np.isnan(on_vals[idx])
            if not both.any():
                continue
            bidx = idx[both]
            diff = off_vals[bidx] - on_vals[bidx]  # observed off - on
            d_b  = d[bidx]
            num = np.nansum(diff * d_b) + k_shrink
            den = np.nansum(d_b ** 2) + k_shrink
            if den == 0:
                continue
            r_out[idx] = num / den  # same r_i for all visits of this patient

        return r_out

    @staticmethod
    def _weighted_pooled_line(
        df: pd.DataFrame,
        off_corr: np.ndarray,
        on_corr: np.ndarray,
        patient_ids: np.ndarray,
        w_off: float,
        w_on: float,
    ) -> np.ndarray:
        """Weighted OLS pooled line per patient; returns fitted value at each visit."""
        n = len(df)
        times = (df["age"].values - df["age_at_diagnosis"].values).astype(float)
        fitted = np.full(n, np.nan)

        unique_pids = np.unique(patient_ids)
        for pid in unique_pids:
            mask = patient_ids == pid
            idx  = np.where(mask)[0]
            t    = times[idx]

            oc = off_corr[idx]
            nc = on_corr[idx]

            # stack all corrected measurements with weights
            all_vals = np.concatenate([oc, nc])
            all_t    = np.concatenate([t, t])
            all_w    = np.concatenate([
                np.where(np.isnan(oc), 0.0, w_off),
                np.where(np.isnan(nc), 0.0, w_on),
            ])
            valid = ~np.isnan(all_vals)
            n_valid = valid.sum()

            if n_valid == 0:
                continue

            tv = all_t[valid]
            yv = all_vals[valid]
            wv = all_w[valid]
            W  = wv.sum()

            if n_valid < 3:
                # weighted mean
                slope = 0.0
                intercept = np.dot(wv, yv) / W if W > 0 else np.nanmean(yv)
            else:
                xm = np.dot(wv, tv) / W
                ym = np.dot(wv, yv) / W
                tc = tv - xm
                denom = np.dot(wv, tc ** 2)
                if denom == 0:
                    slope = 0.0
                    intercept = ym
                else:
                    slope = np.dot(wv, tc * (yv - ym)) / denom
                    intercept = ym - slope * xm

            fitted[idx] = slope * t + intercept

        return fitted

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #

    def fit(self, df_raw: pd.DataFrame, y: pd.Series, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids required")

        pids = np.asarray(patient_ids)

        # Step 1: OOF effect curves (for training rows)
        oof_h_off = self._cross_fit_h(df_raw, y, "off", _H_FEATURES_OFF, pids)
        oof_h_on  = self._cross_fit_h(df_raw, y, "on",  _H_FEATURES_ON,  pids)

        # Full-data effect curves (for test rows)
        self._model_h_off_ = self._fit_full_h(df_raw, y, "off", _H_FEATURES_OFF)
        self._model_h_on_  = self._fit_full_h(df_raw, y, "on",  _H_FEATURES_ON)

        # Step 2: k = median of sum(d^2) over training patients (using OOF h)
        d_oof = oof_h_on - oof_h_off
        sum_d2_per_patient = {}
        for pid in np.unique(pids):
            mask = pids == pid
            off_v = df_raw["off"].values[mask].astype(float)
            on_v  = df_raw["on"].values[mask].astype(float)
            both  = ~np.isnan(off_v) & ~np.isnan(on_v)
            if both.any():
                sum_d2_per_patient[pid] = np.nansum(d_oof[mask][both] ** 2)
        self._k_shrink_ = (
            np.median(list(sum_d2_per_patient.values()))
            if sum_d2_per_patient
            else 1.0
        )

        # Compute r_i on training data using OOF h
        r_oof = self._compute_r_i(df_raw, oof_h_off, oof_h_on, pids, self._k_shrink_)

        # OOF corrected measurements
        off_vals = df_raw["off"].values.astype(float)
        on_vals  = df_raw["on"].values.astype(float)
        off_corr_oof = np.where(np.isnan(off_vals), np.nan, off_vals + r_oof * oof_h_off)
        on_corr_oof  = np.where(np.isnan(on_vals),  np.nan, on_vals  + r_oof * oof_h_on)

        # Step 4: estimate residual variance per measurement type on training fold
        # variance of (off_corr - target) and (on_corr - target)
        t_vals = y.values
        resid_off = off_corr_oof - t_vals
        resid_on  = on_corr_oof  - t_vals
        var_off = np.nanvar(resid_off[~np.isnan(resid_off)])
        var_on  = np.nanvar(resid_on[~np.isnan(resid_on)])
        self._w_off_ = 1.0 / var_off if var_off > 0 else 1.0
        self._w_on_  = 1.0 / var_on  if var_on  > 0 else 1.0

        # Store per-patient r_i from training (indexed by patient id)
        self._r_by_pid_: dict = {}
        for pid in np.unique(pids):
            mask = pids == pid
            self._r_by_pid_[pid] = float(r_oof[mask][0])

        # Store OOF arrays keyed by df index for training transform
        self._train_index_ = df_raw.index.copy()
        self._oof_h_off_ = pd.Series(oof_h_off, index=df_raw.index)
        self._oof_h_on_  = pd.Series(oof_h_on,  index=df_raw.index)
        self._oof_r_i_   = pd.Series(r_oof,     index=df_raw.index)

        return self

    def transform(self, df_raw: pd.DataFrame, patient_ids=None) -> pd.DataFrame:
        df = df_raw.copy()
        pids = np.asarray(
            patient_ids if patient_ids is not None else df["patient_id"].values
        )

        n = len(df)
        train_mask = df.index.isin(self._train_index_)

        h_off = np.full(n, np.nan)
        h_on  = np.full(n, np.nan)
        r_i   = np.ones(n)

        # Training rows: use OOF quantities
        if train_mask.any():
            h_off[train_mask] = self._oof_h_off_.reindex(df.index[train_mask]).values
            h_on[train_mask]  = self._oof_h_on_.reindex(df.index[train_mask]).values
            r_i[train_mask]   = self._oof_r_i_.reindex(df.index[train_mask]).values

        # Test / holdout rows: use full-data h models + stored r_i
        if (~train_mask).any():
            test_df = df[~train_mask]
            h_off[~train_mask] = self._model_h_off_.predict(
                test_df[_H_FEATURES_OFF].values.astype(float)
            )
            h_on[~train_mask]  = self._model_h_on_.predict(
                test_df[_H_FEATURES_ON].values.astype(float)
            )
            # Use stored r_i for known patients; unknown -> 1
            test_pids = pids[~train_mask]
            r_i[~train_mask] = np.array([
                self._r_by_pid_.get(pid, 1.0) for pid in test_pids
            ])

        # Step 3: corrected measurements
        off_vals = df["off"].values.astype(float)
        on_vals  = df["on"].values.astype(float)
        off_corr = np.where(np.isnan(off_vals), np.nan, off_vals + r_i * h_off)
        on_corr  = np.where(np.isnan(on_vals),  np.nan, on_vals  + r_i * h_on)

        # Step 4: weighted OLS pooled line per patient
        pool_fitted = self._weighted_pooled_line(
            df, off_corr, on_corr, pids, self._w_off_, self._w_on_
        )

        # Step 5: count of corrected measurements per patient (vectorized)
        per_row_corr = (~np.isnan(off_corr)).astype(float) + (~np.isnan(on_corr)).astype(float)
        n_corr_series = pd.Series(per_row_corr, index=df.index)
        n_corr = n_corr_series.groupby(pd.Series(pids, index=df.index)).transform("sum").values

        # Attach features
        df["pr_r_i"]           = r_i
        df["pr_h_off"]         = h_off
        df["pr_h_on"]          = h_on
        df["pr_ri_h_off"]      = r_i * h_off
        df["pr_ri_h_on"]       = r_i * h_on
        df["pr_off_corr"]      = off_corr
        df["pr_on_corr"]       = on_corr
        df["pr_pool_fitted"]   = pool_fitted
        df["pr_n_corr"]        = n_corr

        return df


# %% [markdown]
# ## Load data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train     = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw  = pd.read_csv(DATA_DIR / "X_test.csv",  index_col="Index")

groups_train = X_train_raw["patient_id"]

# Pre-add time_since_diagnosis (used by both variant A features and PatientResponse)
X_train_raw2 = X_train_raw.copy()
X_train_raw2["time_since_diagnosis"] = X_train_raw2["age"] - X_train_raw2["age_at_diagnosis"]
X_test_raw2 = X_test_raw.copy()
X_test_raw2["time_since_diagnosis"] = X_test_raw2["age"] - X_test_raw2["age_at_diagnosis"]

# %% [markdown]
# ## Variant A: control (variant B of 05, same features + HGB)

# %%
# Variant A uses the PatientCorrectedPooling from 05 (copied inline below as
# the 05 transformer) — we re-implement the same feature pipeline here so
# this script is self-contained and produces an identical A baseline.

_BIAS_FEATURES_OFF_05 = ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]
_BIAS_FEATURES_ON_05  = ["time_since_intake_on",  "cohort", "ledd", "time_since_diagnosis"]


class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Variant B of 05 — visit-level bias correction + patient trajectory."""

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
            model  = self._bias_corrector(
                col_data.iloc[tr_idx].values, residuals[tr_idx], cat_idx
            )
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

    @staticmethod
    def _trajectory_features(df, off_corr, on_corr, pid_col="patient_id", time_col="time_since_diagnosis"):
        n    = len(df)
        pids = df[pid_col].values
        times = df[time_col].values.astype(float)
        feat_off_fitted  = np.full(n, np.nan)
        feat_on_fitted   = np.full(n, np.nan)
        feat_pool_fitted = np.full(n, np.nan)
        feat_pool_mean   = np.full(n, np.nan)
        feat_n_off = np.zeros(n, dtype=float)
        feat_n_on  = np.zeros(n, dtype=float)
        for pid in np.unique(pids):
            mask = pids == pid
            idx  = np.where(mask)[0]
            t = times[idx]; oc = off_corr[idx]; nc = on_corr[idx]
            valid_off = ~np.isnan(oc); n_off = valid_off.sum()
            if n_off >= 3:
                s, b = _linreg_slope_intercept(t, oc)
                feat_off_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_off > 0:
                feat_off_fitted[idx] = np.nanmean(oc[valid_off])
            valid_on = ~np.isnan(nc); n_on = valid_on.sum()
            if n_on >= 3:
                s, b = _linreg_slope_intercept(t, nc)
                feat_on_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_on > 0:
                feat_on_fitted[idx] = np.nanmean(nc[valid_on])
            all_vals = np.concatenate([oc, nc]); all_times = np.concatenate([t, t])
            valid_pool = ~np.isnan(all_vals); n_pool = valid_pool.sum()
            if n_pool >= 3:
                s, b = _linreg_slope_intercept(all_times, all_vals)
                feat_pool_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_pool > 0:
                feat_pool_fitted[idx] = np.nanmean(all_vals[valid_pool])
            feat_pool_mean[idx] = np.nanmean(all_vals) if valid_pool.sum() > 0 else np.nan
            feat_n_off[idx] = n_off; feat_n_on[idx] = n_on
        return {
            "off_corr_pt_fitted": feat_off_fitted, "on_corr_pt_fitted": feat_on_fitted,
            "pool_corr_pt_fitted": feat_pool_fitted, "pool_corr_pt_mean": feat_pool_mean,
            "pt_n_off_corr": feat_n_off, "pt_n_on_corr": feat_n_on,
        }

    def fit(self, df_raw, y, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids required")
        self._cohort_cats_ = df_raw["cohort"].astype("category").cat.categories.tolist()
        oof_off = self._cross_fit_corrections(df_raw, y, "off", _BIAS_FEATURES_OFF_05, patient_ids)
        oof_on  = self._cross_fit_corrections(df_raw, y, "on",  _BIAS_FEATURES_ON_05,  patient_ids)
        self._model_off_, _ = self._fit_full_corrector(df_raw, y, "off", _BIAS_FEATURES_OFF_05)
        self._model_on_,  _ = self._fit_full_corrector(df_raw, y, "on",  _BIAS_FEATURES_ON_05)
        self._train_index_  = df_raw.index.copy()
        self._oof_off_ = pd.Series(oof_off, index=df_raw.index)
        self._oof_on_  = pd.Series(oof_on,  index=df_raw.index)
        return self

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        train_mask = df.index.isin(self._train_index_)
        off_vals = df["off"].values.astype(float); on_vals = df["on"].values.astype(float)
        g_off = np.full(len(df), np.nan); g_on = np.full(len(df), np.nan)
        if train_mask.any():
            g_off[train_mask] = self._oof_off_.reindex(df.index[train_mask]).values
            g_on[train_mask]  = self._oof_on_.reindex(df.index[train_mask]).values
        if (~train_mask).any():
            test_df = df[~train_mask]
            g_off[~train_mask] = self._apply_corrector(
                self._model_off_, test_df, _BIAS_FEATURES_OFF_05, self._cohort_cats_
            )
            g_on[~train_mask] = self._apply_corrector(
                self._model_on_, test_df, _BIAS_FEATURES_ON_05, self._cohort_cats_
            )
        off_corr = np.where(np.isnan(off_vals), np.nan, off_vals + g_off)
        on_corr  = np.where(np.isnan(on_vals),  np.nan, on_vals  + g_on)
        traj = self._trajectory_features(df, off_corr, on_corr)
        for col, vals in traj.items():
            df[col] = vals
        return df


def prepare_features_A(df_raw: pd.DataFrame, pcp: PatientCorrectedPooling, patient_ids) -> pd.DataFrame:
    X = df_raw.copy()
    if "time_since_diagnosis" not in X.columns:
        X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    X = pcp.transform(X)
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"]   = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


# %% [markdown]
# ## Build CV splits and cross-validate all three variants

# %%
from sklearn.metrics import root_mean_squared_error  # noqa: E402

gkf      = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_train_raw2, y_train, groups=groups_train))

rmse_A = []
rmse_B = []
rmse_C = []

for fold_train_idx, fold_val_idx in cv_splits:
    X_tr_raw = X_train_raw2.iloc[fold_train_idx].copy()
    X_vl_raw = X_train_raw2.iloc[fold_val_idx].copy()
    y_tr     = y_train.iloc[fold_train_idx]
    y_vl     = y_train.iloc[fold_val_idx]
    pids_tr  = groups_train.iloc[fold_train_idx].values

    # ---------- variant A ----------
    pcp_a = PatientCorrectedPooling(n_splits=5)
    pcp_a.fit(X_tr_raw, y_tr, patient_ids=pids_tr)
    X_tr_A = prepare_features_A(X_tr_raw, pcp_a, pids_tr)
    X_vl_A = prepare_features_A(X_vl_raw, pcp_a, groups_train.iloc[fold_val_idx].values)
    m_a = HistGradientBoostingRegressor(random_state=0)
    m_a.fit(X_tr_A, y_tr)
    pred_A = m_a.predict(X_vl_A)
    rmse_A.append(root_mean_squared_error(y_vl, pred_A))

    # ---------- variant B: A features + PatientResponse features ----------
    pr = PatientResponse(n_splits=5)
    pr.fit(X_tr_raw, y_tr, patient_ids=pids_tr)

    def _add_pr_features(df_raw, pr_obj, pids_arr):
        df = pr_obj.transform(df_raw, patient_ids=pids_arr)
        return df

    X_tr_pr = _add_pr_features(X_tr_raw, pr, pids_tr)
    X_vl_pr = _add_pr_features(X_vl_raw, pr, groups_train.iloc[fold_val_idx].values)

    # Combine: start from A's pcp-corrected frame, then attach pr_ columns
    pr_cols = [c for c in X_tr_pr.columns if c.startswith("pr_")]
    X_tr_B = X_tr_A.copy()
    for c in pr_cols:
        X_tr_B[c] = X_tr_pr[c].values
    X_vl_B = X_vl_A.copy()
    for c in pr_cols:
        X_vl_B[c] = X_vl_pr[c].values

    m_b = HistGradientBoostingRegressor(random_state=0)
    m_b.fit(X_tr_B, y_tr)
    pred_B = m_b.predict(X_vl_B)
    rmse_B.append(root_mean_squared_error(y_vl, pred_B))

    # ---------- variant C: pooled line direct, NaN filled with B pred ----------
    pool_fitted_vl = X_vl_pr["pr_pool_fitted"].values.astype(float)
    pred_C = np.where(np.isnan(pool_fitted_vl), pred_B, pool_fitted_vl)
    rmse_C.append(root_mean_squared_error(y_vl, pred_C))

mean_A, std_A = np.mean(rmse_A), np.std(rmse_A)
mean_B, std_B = np.mean(rmse_B), np.std(rmse_B)
mean_C, std_C = np.mean(rmse_C), np.std(rmse_C)
print(f"A RMSE: {mean_A:.4f} ± {std_A:.4f}")
print(f"B RMSE: {mean_B:.4f} ± {std_B:.4f}")
print(f"C RMSE: {mean_C:.4f} ± {std_C:.4f}")

# %% [markdown]
# ## Full-data models for skore reports & hub push

# %%
# Fit full-data transformers
pcp_full = PatientCorrectedPooling(n_splits=5)
pcp_full.fit(X_train_raw2, y_train, patient_ids=groups_train.values)
pr_full  = PatientResponse(n_splits=5)
pr_full.fit(X_train_raw2, y_train, patient_ids=groups_train.values)

X_train_A_full = prepare_features_A(X_train_raw2, pcp_full, groups_train.values)
X_test_A_full  = prepare_features_A(X_test_raw2, pcp_full, X_test_raw2["patient_id"].values)

pr_cols_all = [c for c in pr_full.transform(X_train_raw2, patient_ids=groups_train.values).columns
               if c.startswith("pr_")]

X_train_pr = pr_full.transform(X_train_raw2, patient_ids=groups_train.values)
X_test_pr  = pr_full.transform(X_test_raw2,  patient_ids=X_test_raw2["patient_id"].values)

X_train_B_full = X_train_A_full.copy()
for c in pr_cols_all:
    X_train_B_full[c] = X_train_pr[c].values

X_test_B_full = X_test_A_full.copy()
for c in pr_cols_all:
    X_test_B_full[c] = X_test_pr[c].values

# median r_i and spread
r_i_all = X_train_pr["pr_r_i"].values
median_ri = np.median(r_i_all)
iqr_ri    = np.percentile(r_i_all, 75) - np.percentile(r_i_all, 25)
print(f"Median r_i: {median_ri:.3f}  IQR: {iqr_ri:.3f}")

# skore evaluate for A and B (CV splits on full feature matrices)
cv_splits_A = list(gkf.split(X_train_A_full, y_train, groups=groups_train))
cv_splits_B = list(gkf.split(X_train_B_full, y_train, groups=groups_train))

report_A = skore.evaluate(
    HistGradientBoostingRegressor(random_state=0),
    X_train_A_full, y_train, splitter=cv_splits_A,
)
report_B = skore.evaluate(
    HistGradientBoostingRegressor(random_state=0),
    X_train_B_full, y_train, splitter=cv_splits_B,
)

# %% [markdown]
# ## Push to hub

# %%
from skore import ComparisonReport  # noqa: E402

comparison = ComparisonReport({
    "A_05B_control": report_A,
    "B_plus_response": report_B,
})
# Hub only accepts EstimatorReport / CrossValidationReport; push individually
project.put(f"{EXPERIMENT_KEY}_A", report_A)
project.put(f"{EXPERIMENT_KEY}_B", report_B)
print(f"Pushed {EXPERIMENT_KEY}_A and {EXPERIMENT_KEY}_B to hub")

# %% [markdown]
# ## Determine best variant and produce submission if it beats A

# %%
best_mean = min(mean_A, mean_B, mean_C)
best_name = ["A", "B", "C"][[mean_A, mean_B, mean_C].index(best_mean)]
print(f"Best variant: {best_name} ({best_mean:.4f})")

if best_mean < mean_A or best_name != "A":
    from skore import EstimatorReport  # noqa: E402

    # Determine the best feature matrices
    if best_name in ("B", "C"):
        X_train_best = X_train_B_full
        X_test_best  = X_test_B_full
    else:
        X_train_best = X_train_A_full
        X_test_best  = X_test_A_full

    # 80/20 patient-grouped holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_vl_idx = next(gss.split(X_train_raw2, y_train, groups=groups_train))

    X_hold_tr_raw = X_train_raw2.iloc[hold_tr_idx].copy()
    X_hold_vl_raw = X_train_raw2.iloc[hold_vl_idx].copy()
    y_hold_tr = y_train.iloc[hold_tr_idx]
    y_hold_vl = y_train.iloc[hold_vl_idx]
    pids_hold_tr = groups_train.iloc[hold_tr_idx].values

    pcp_hold = PatientCorrectedPooling(n_splits=5)
    pcp_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)
    pr_hold  = PatientResponse(n_splits=5)
    pr_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)

    X_hold_tr_A = prepare_features_A(X_hold_tr_raw, pcp_hold, pids_hold_tr)
    X_hold_vl_A = prepare_features_A(X_hold_vl_raw, pcp_hold,
                                     groups_train.iloc[hold_vl_idx].values)

    X_hold_tr_pr = pr_hold.transform(X_hold_tr_raw, patient_ids=pids_hold_tr)
    X_hold_vl_pr = pr_hold.transform(X_hold_vl_raw,
                                     patient_ids=groups_train.iloc[hold_vl_idx].values)

    X_hold_tr_B = X_hold_tr_A.copy()
    X_hold_vl_B = X_hold_vl_A.copy()
    for c in pr_cols_all:
        X_hold_tr_B[c] = X_hold_tr_pr[c].values
        X_hold_vl_B[c] = X_hold_vl_pr[c].values

    if best_name in ("B", "C"):
        X_hold_tr_best = X_hold_tr_B
        X_hold_vl_best = X_hold_vl_B
    else:
        X_hold_tr_best = X_hold_tr_A
        X_hold_vl_best = X_hold_vl_A

    report_final = EstimatorReport(
        HistGradientBoostingRegressor(random_state=0),
        X_train=X_hold_tr_best,
        y_train=y_hold_tr,
        X_test=X_hold_vl_best,
        y_test=y_hold_vl,
    )
    project.put(f"{EXPERIMENT_KEY}_final", report_final)
    print(f"Pushed {EXPERIMENT_KEY}_final to hub")

    # Fit on all train, predict X_test
    model_final = HistGradientBoostingRegressor(random_state=0)
    model_final.fit(X_train_best, y_train)
    preds_test = model_final.predict(X_test_best)

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    csv_path = submissions_dir / f"{EXPERIMENT_KEY}.csv"
    submission.to_csv(csv_path, index=False)
    print(f"Submission written: {len(submission)} rows -> {csv_path}")
else:
    print("No variant beats A — no submission produced")
