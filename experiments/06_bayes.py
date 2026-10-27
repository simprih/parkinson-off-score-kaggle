# %% [markdown]
# # Experiment 06 — bayes
#
# **Key:** 06_bayes  |  **Date:** 2026-10-08  |  **Kept:** partly
# **Idea:** Shrunk per-patient OLS line via empirical Bayes (EmpiricalBayesLine).
# **CV (GroupKFold 5):** not scored
# The posterior line feature was partially absorbed into MixedLine (experiment 07).
# **Skore Hub:** report keys `06_bayes_A`, `06_bayes_B`, `06_bayes_final` (ids not listed in the report).
# **Run:** `python experiments/06_bayes.py`
# (Windows: `.venv\Scripts\python.exe -X utf8 -u experiments/06_bayes.py`).
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
EXPERIMENT_KEY = "06_bayes"

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
# ## Helpers shared with 05

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
# ## PatientCorrectedPooling transformer (from 05)

# %%
_BIAS_FEATURES_OFF = ["time_since_intake_off", "cohort", "ledd", "time_since_diagnosis"]
_BIAS_FEATURES_ON = ["time_since_intake_on", "cohort", "ledd", "time_since_diagnosis"]


class PatientCorrectedPooling(BaseEstimator, TransformerMixin):
    """Visit-level bias correction followed by patient trajectory features (from 05)."""

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
        has_score = ~np.isnan(score_vals)
        residuals = y.values - score_vals
        oof = np.full(len(df), np.nan)
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, val_idx in gkf.split(col_data, residuals, groups=patient_ids):
            tr_mask = has_score[train_idx]
            if tr_mask.sum() < 10:
                continue
            tr_idx = train_idx[tr_mask]
            X_tr = col_data.iloc[tr_idx].values
            y_tr = residuals[tr_idx]
            model = self._bias_corrector(X_tr, y_tr, cat_idx)
            X_val = col_data.iloc[val_idx].values
            oof[val_idx] = model.predict(X_val)
        return oof

    def _fit_full_corrector(self, df, y, score_col, bias_features):
        col_data = df[bias_features].copy()
        col_data["cohort"] = col_data["cohort"].astype("category").cat.codes.astype(float)
        cat_idx = [bias_features.index("cohort")]
        score_vals = df[score_col].values.astype(float)
        has_score = ~np.isnan(score_vals)
        residuals = y.values - score_vals
        X_tr = col_data[has_score].values
        y_tr = residuals[has_score]
        return self._bias_corrector(X_tr, y_tr, cat_idx), col_data.columns.tolist()

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
        n = len(df)
        pids = df[pid_col].values
        times = df[time_col].values.astype(float)
        feat_off_fitted = np.full(n, np.nan)
        feat_on_fitted = np.full(n, np.nan)
        feat_pool_fitted = np.full(n, np.nan)
        feat_pool_mean = np.full(n, np.nan)
        feat_n_off = np.zeros(n, dtype=float)
        feat_n_on = np.zeros(n, dtype=float)
        for pid in np.unique(pids):
            mask = pids == pid
            idx = np.where(mask)[0]
            t = times[idx]
            oc = off_corr[idx]
            nc = on_corr[idx]
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
            all_vals = np.concatenate([oc, nc])
            all_times = np.concatenate([t, t])
            valid_pool = ~np.isnan(all_vals)
            n_pool = valid_pool.sum()
            if n_pool >= 3:
                s, b = _linreg_slope_intercept(all_times, all_vals)
                feat_pool_fitted[idx] = np.where(np.isnan(s), np.nan, s * t + b)
            elif n_pool > 0:
                feat_pool_fitted[idx] = np.nanmean(all_vals[valid_pool])
            pool_mean = np.nanmean(all_vals) if valid_pool.sum() > 0 else np.nan
            feat_pool_mean[idx] = pool_mean
            feat_n_off[idx] = n_off
            feat_n_on[idx] = n_on
        return {
            "off_corr_pt_fitted": feat_off_fitted,
            "on_corr_pt_fitted": feat_on_fitted,
            "pool_corr_pt_fitted": feat_pool_fitted,
            "pool_corr_pt_mean": feat_pool_mean,
            "pt_n_off_corr": feat_n_off,
            "pt_n_on_corr": feat_n_on,
        }

    def fit(self, df_raw, y, patient_ids=None):
        if patient_ids is None:
            raise ValueError("patient_ids is required")
        self._cohort_cats_ = df_raw["cohort"].astype("category").cat.categories.tolist()
        oof_off = self._cross_fit_corrections(
            df_raw, y, "off", _BIAS_FEATURES_OFF, patient_ids
        )
        oof_on = self._cross_fit_corrections(
            df_raw, y, "on", _BIAS_FEATURES_ON, patient_ids
        )
        self._model_off_, _ = self._fit_full_corrector(df_raw, y, "off", _BIAS_FEATURES_OFF)
        self._model_on_, _ = self._fit_full_corrector(df_raw, y, "on", _BIAS_FEATURES_ON)
        self._train_index_ = df_raw.index.copy()
        self._oof_off_ = pd.Series(oof_off, index=df_raw.index)
        self._oof_on_ = pd.Series(oof_on, index=df_raw.index)
        return self

    def transform(self, df_raw, patient_ids=None):
        df = df_raw.copy()
        train_mask = df.index.isin(self._train_index_)
        off_vals = df["off"].values.astype(float)
        on_vals = df["on"].values.astype(float)
        g_off = np.full(len(df), np.nan)
        g_on = np.full(len(df), np.nan)
        if train_mask.any():
            g_off[train_mask] = self._oof_off_.reindex(df.index[train_mask]).values
            g_on[train_mask] = self._oof_on_.reindex(df.index[train_mask]).values
        if (~train_mask).any():
            test_df = df[~train_mask]
            g_off[~train_mask] = self._apply_corrector(
                self._model_off_, test_df, _BIAS_FEATURES_OFF, self._cohort_cats_
            )
            g_on[~train_mask] = self._apply_corrector(
                self._model_on_, test_df, _BIAS_FEATURES_ON, self._cohort_cats_
            )
        off_corr = np.where(np.isnan(off_vals), np.nan, off_vals + g_off)
        on_corr = np.where(np.isnan(on_vals), np.nan, on_vals + g_on)
        # Expose corrected measurements for EmpiricalBayesLine
        df["off_corr"] = off_corr
        df["on_corr"] = on_corr
        traj = self._trajectory_features(df, off_corr, on_corr)
        for col, vals in traj.items():
            df[col] = vals
        return df


# %% [markdown]
# ## EmpiricalBayesLine transformer

# %%
_PT_COV_FEATURES = ["cohort_code", "sexM", "gene_known", "age_at_diagnosis",
                    "mean_disease_dur", "any_ledd"]


class EmpiricalBayesLine(BaseEstimator, TransformerMixin):
    """Shrink per-patient OLS lines toward a population prior via conjugate update.

    Fit (training fold only)
    ------------------------
    1. Center time: t = time_since_diagnosis - patient_mean(time_since_diagnosis).
    2. Pool off_corr and on_corr into one set of (t, y_meas) per patient;
       fit each patient's OLS line on target vs t.
    3. Prior mean: ridge regression of (intercept, slope) on patient covariates.
    4. Prior covariance: 2×2 sample covariance of (intercept - prior_mean_intercept,
       slope - prior_mean_slope) across patients.
    5. Noise variance: residual variance of off_corr and on_corr around the
       patient's target line (estimated separately for off and on).

    Transform (train and test)
    --------------------------
    For every patient, gather its corrected measurements, compute the conjugate
    posterior mean and covariance, and add four features per visit:
      eb_line_val  : posterior line evaluated at the visit's t
      eb_intercept : posterior intercept
      eb_slope     : posterior slope
      eb_intercept_std : posterior std of intercept
      eb_slope_std     : posterior std of slope
    """

    def __init__(self, ridge_alpha: float = 1.0):
        self.ridge_alpha = ridge_alpha

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _patient_t(df: pd.DataFrame) -> np.ndarray:
        """time_since_diagnosis centered per patient."""
        t_raw = df["time_since_diagnosis"].values.astype(float)
        pids = df["patient_id"].values
        t_centered = t_raw.copy()
        for pid in np.unique(pids):
            mask = pids == pid
            t_centered[mask] = t_raw[mask] - np.nanmean(t_raw[mask])
        return t_centered

    @staticmethod
    def _patient_covariates(df: pd.DataFrame) -> np.ndarray:
        """Build the patient-level covariate matrix Z (one row per unique patient)."""
        pt = df.groupby("patient_id").first().reset_index()
        Z = np.column_stack([
            np.ones(len(pt)),                                        # intercept
            pd.Categorical(pt["cohort"]).codes.astype(float),       # cohort_code
            (pt["sex"] == "M").astype(float).values if "sex" in pt.columns
                else np.zeros(len(pt)),                              # sexM
            (~pd.isna(pt["gene"]) & (pt["gene"] != "unknown")).astype(float).values
                if "gene" in pt.columns else np.zeros(len(pt)),     # gene_known
            pt["age_at_diagnosis"].values.astype(float),            # age_at_diagnosis
            pt.groupby("patient_id")["time_since_diagnosis"].transform("mean").values
                if False else
                df.groupby("patient_id")["time_since_diagnosis"].mean().reindex(
                    pt["patient_id"]).values.astype(float),         # mean_disease_dur
            (pt["ledd"].notna() & (pt["ledd"] > 0)).astype(float).values
                if "ledd" in pt.columns else np.zeros(len(pt)),     # any_ledd
        ])
        return Z, pt["patient_id"].values

    @staticmethod
    def _ridge_fit(Z: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
        """Ridge regression: returns weight vector (Z.T Z + alpha I)^{-1} Z.T y."""
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        b = Z.T @ targets
        return np.linalg.solve(A, b)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame, y: pd.Series, patient_ids=None):
        """Fit prior from the training fold.

        df must already contain off_corr, on_corr, time_since_diagnosis.
        """
        pids = df["patient_id"].values
        t_centered = self._patient_t(df)
        y_vals = y.values if hasattr(y, "values") else np.array(y)

        # ---- per-patient OLS lines of target vs centered t ----
        intercepts, slopes = {}, {}
        noise_off_sq, noise_on_sq = [], []

        for pid in np.unique(pids):
            mask = pids == pid
            t_p = t_centered[mask]
            y_p = y_vals[np.where(mask)[0]]

            s, b = _linreg_slope_intercept(t_p, y_p)
            if np.isnan(s):
                s, b = 0.0, float(np.nanmean(y_p)) if np.any(~np.isnan(y_p)) else 0.0
            intercepts[pid] = b
            slopes[pid] = s

            # noise variance: residual of off_corr / on_corr around target line
            df_p = df.iloc[np.where(mask)[0]]
            fitted_line = s * t_p + b
            for col, store in [("off_corr", noise_off_sq), ("on_corr", noise_on_sq)]:
                meas = df_p[col].values.astype(float)
                valid = ~np.isnan(meas)
                if valid.sum() >= 1:
                    store.append(np.mean((meas[valid] - fitted_line[valid]) ** 2))

        self._intercepts_train_ = intercepts
        self._slopes_train_ = slopes

        # noise variances (scalar, pooled across patients)
        self._sigma2_off_ = float(np.mean(noise_off_sq)) if noise_off_sq else 1.0
        self._sigma2_on_ = float(np.mean(noise_on_sq)) if noise_on_sq else 1.0

        # ---- prior mean via ridge regression on patient covariates ----
        Z, pid_order = self._patient_covariates(df)
        int_arr = np.array([intercepts[p] for p in pid_order])
        sl_arr = np.array([slopes[p] for p in pid_order])

        self._w_int_ = self._ridge_fit(Z, int_arr, self.ridge_alpha)
        self._w_sl_ = self._ridge_fit(Z, sl_arr, self.ridge_alpha)

        prior_int = Z @ self._w_int_
        prior_sl = Z @ self._w_sl_

        # ---- prior covariance of (intercept, slope) residuals ----
        resid_int = int_arr - prior_int
        resid_sl = sl_arr - prior_sl
        mat = np.column_stack([resid_int, resid_sl])
        cov = mat.T @ mat / max(len(mat) - 1, 1)
        # regularise
        cov += 1e-6 * np.eye(2)
        self._prior_cov_ = cov
        self._pid_order_train_ = pid_order

        # store cohort categories for test transform
        self._cohort_cats_ = df["cohort"].astype("category").cat.categories.tolist()

        return self

    # ------------------------------------------------------------------ #
    # Prior mean for a single patient row
    # ------------------------------------------------------------------ #

    def _prior_mean(self, row_series: pd.Series) -> np.ndarray:
        """Return [prior_intercept, prior_slope] for a single patient's first row."""
        cohort_code = float(
            pd.Categorical([row_series.get("cohort", "")],
                           categories=self._cohort_cats_).codes[0]
        )
        sex_m = float(row_series.get("sex", "") == "M")
        gene = row_series.get("gene", None)
        gene_known = float(gene is not None and gene not in [None, "unknown", ""])
        age_dx = float(row_series.get("age_at_diagnosis", 0) or 0)
        mean_dur = float(row_series.get("_pt_mean_t", 0) or 0)
        any_ledd = float((row_series.get("ledd", None) or 0) > 0)
        z = np.array([1.0, cohort_code, sex_m, gene_known, age_dx, mean_dur, any_ledd])
        return np.array([float(self._w_int_ @ z), float(self._w_sl_ @ z)])

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame, patient_ids=None) -> pd.DataFrame:
        """Add posterior EB features to df.

        df must contain off_corr, on_corr, time_since_diagnosis, patient_id.
        """
        out = df.copy()
        pids = out["patient_id"].values
        t_centered = self._patient_t(out)

        n = len(out)
        eb_line_val = np.full(n, np.nan)
        eb_intercept = np.full(n, np.nan)
        eb_slope = np.full(n, np.nan)
        eb_intercept_std = np.full(n, np.nan)
        eb_slope_std = np.full(n, np.nan)

        # precompute per-patient mean of centered t (used as mean_disease_dur proxy)
        pt_mean_t = {}
        for pid in np.unique(pids):
            mask = pids == pid
            pt_mean_t[pid] = float(np.nanmean(t_centered[mask]))

        for pid in np.unique(pids):
            mask = pids == pid
            idx = np.where(mask)[0]
            t_p = t_centered[idx]

            df_p = out.iloc[idx]
            first_row = df_p.iloc[0].copy()
            first_row["_pt_mean_t"] = pt_mean_t[pid]

            mu0 = self._prior_mean(first_row)       # [mu_int, mu_sl]
            Sigma0 = self._prior_cov_.copy()        # 2×2

            # gather corrected measurements (off_corr and on_corr)
            meas_vals = []  # list of (t_i, y_i, sigma2_i)
            for col, s2 in [("off_corr", self._sigma2_off_), ("on_corr", self._sigma2_on_)]:
                m = df_p[col].values.astype(float)
                valid = ~np.isnan(m)
                for j in np.where(valid)[0]:
                    meas_vals.append((t_p[j], m[j], s2))

            # conjugate Gaussian update: posterior = N(mu_post, Sigma_post)
            # observation model: y_i = [1, t_i] @ theta + noise(sigma2_i)
            Sigma_post = np.linalg.inv(np.linalg.inv(Sigma0))
            mu_post = mu0.copy()

            if meas_vals:
                # Woodbury / sequential update
                Sigma_n = Sigma0.copy()
                info = np.linalg.inv(Sigma_n)
                info_mu = info @ mu_post

                for t_i, y_i, s2_i in meas_vals:
                    h = np.array([1.0, t_i])
                    info += np.outer(h, h) / s2_i
                    info_mu += h * y_i / s2_i

                Sigma_post = np.linalg.inv(info)
                mu_post = Sigma_post @ info_mu

            post_int, post_sl = mu_post[0], mu_post[1]
            post_int_std = float(np.sqrt(max(Sigma_post[0, 0], 0)))
            post_sl_std = float(np.sqrt(max(Sigma_post[1, 1], 0)))

            eb_line_val[idx] = post_int + post_sl * t_p
            eb_intercept[idx] = post_int
            eb_slope[idx] = post_sl
            eb_intercept_std[idx] = post_int_std
            eb_slope_std[idx] = post_sl_std

        out["eb_line_val"] = eb_line_val
        out["eb_intercept"] = eb_intercept
        out["eb_slope"] = eb_slope
        out["eb_intercept_std"] = eb_intercept_std
        out["eb_slope_std"] = eb_slope_std
        return out


# %% [markdown]
# ## Load data

# %%
X_train_raw = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")["target"]
X_test_raw = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")

groups_train = X_train_raw["patient_id"]
gkf = GroupKFold(n_splits=5)

# Stable category dtypes derived from full training data (consistent across folds)
_COHORT_DTYPE = pd.CategoricalDtype(
    categories=X_train_raw["cohort"].astype("category").cat.categories
)
_GENE_DTYPE = pd.CategoricalDtype(
    categories=X_train_raw["gene"].astype("category").cat.categories
)

# %% [markdown]
# ## Variant A: control (05 variant B features, no EmpiricalBayesLine)

# %%
def prepare_features_03(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["time_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    return X


X_train_A = prepare_features_03(X_train_raw)
cv_splits = list(gkf.split(X_train_A, y_train, groups=groups_train))

model_A = HistGradientBoostingRegressor(random_state=0)
report_A = skore.evaluate(model_A, X_train_A, y_train, splitter=cv_splits)

# %% [markdown]
# ## Build B/C features: PCP + EmpiricalBayesLine per fold

# %%
def _add_tsd(df):
    df = df.copy()
    if "time_since_diagnosis" not in df.columns:
        df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]
    return df


def prepare_features_B(df_raw, pcp: PatientCorrectedPooling,
                        ebl: EmpiricalBayesLine) -> pd.DataFrame:
    """Apply PCP + EmpiricalBayesLine, then 03 patient features."""
    X = pcp.transform(df_raw)
    X = ebl.transform(X)
    X = add_patient_features(X)
    X["cohort"] = X["cohort"].astype("category")
    X["gene"] = X["gene"].astype("category")
    X = X.drop(columns=["patient_id"])
    # drop raw corrected cols (already summarised in trajectory features)
    X = X.drop(columns=["off_corr", "on_corr"], errors="ignore")
    return X


rmse_B, rmse_C = [], []

for fold_train_idx, fold_val_idx in cv_splits:
    X_fold_tr_raw = _add_tsd(X_train_raw.iloc[fold_train_idx])
    X_fold_val_raw = _add_tsd(X_train_raw.iloc[fold_val_idx])
    y_fold_tr = y_train.iloc[fold_train_idx]
    y_fold_val = y_train.iloc[fold_val_idx]
    pids_fold_tr = groups_train.iloc[fold_train_idx].values

    # Fit PCP
    pcp = PatientCorrectedPooling(n_splits=5)
    pcp.fit(X_fold_tr_raw, y_fold_tr, patient_ids=pids_fold_tr)

    # Get PCP-transformed (with off_corr/on_corr exposed)
    X_tr_pcp = pcp.transform(X_fold_tr_raw)
    X_val_pcp = pcp.transform(X_fold_val_raw)

    # Fit EBL on training fold
    ebl = EmpiricalBayesLine(ridge_alpha=1.0)
    ebl.fit(X_tr_pcp, y_fold_tr)

    # Build B feature matrices
    X_tr_B = add_patient_features(ebl.transform(X_tr_pcp))
    X_tr_B["cohort"] = X_tr_B["cohort"].astype(_COHORT_DTYPE)
    X_tr_B["gene"] = X_tr_B["gene"].astype(_GENE_DTYPE)
    X_tr_B = X_tr_B.drop(columns=["patient_id", "off_corr", "on_corr"], errors="ignore")

    X_val_B = add_patient_features(ebl.transform(X_val_pcp))
    X_val_B["cohort"] = X_val_B["cohort"].astype(_COHORT_DTYPE)
    X_val_B["gene"] = X_val_B["gene"].astype(_GENE_DTYPE)
    X_val_B = X_val_B.drop(columns=["patient_id", "off_corr", "on_corr"], errors="ignore")

    # Drop constant columns (HGB binning fails on single-unique-value non-cat features)
    non_cat = [c for c in X_tr_B.columns if not pd.api.types.is_categorical_dtype(X_tr_B[c])]
    const_cols = [c for c in non_cat if X_tr_B[c].nunique(dropna=False) <= 1]
    X_tr_B = X_tr_B.drop(columns=const_cols)
    X_val_B = X_val_B.drop(columns=const_cols, errors="ignore")

    # Variant B: HGB on all features including EB features
    m_B = HistGradientBoostingRegressor(random_state=0)
    m_B.fit(X_tr_B, y_fold_tr)
    preds_B = m_B.predict(X_val_B)
    rmse_B.append(root_mean_squared_error(y_fold_val, preds_B))

    # Variant C: posterior line value used directly as prediction
    # Use the eb_line_val from the validation set
    X_val_eb = ebl.transform(X_val_pcp)
    preds_C = X_val_eb["eb_line_val"].values
    # fallback for any NaN: use population mean from training
    pop_mean = float(y_fold_tr.mean())
    preds_C = np.where(np.isnan(preds_C), pop_mean, preds_C)
    rmse_C.append(root_mean_squared_error(y_fold_val, preds_C))

rmse_B_mean, rmse_B_std = np.mean(rmse_B), np.std(rmse_B)
rmse_C_mean, rmse_C_std = np.mean(rmse_C), np.std(rmse_C)
print(f"B CV RMSE: {rmse_B_mean:.4f} ± {rmse_B_std:.4f}")
print(f"C CV RMSE: {rmse_C_mean:.4f} ± {rmse_C_std:.4f}")

# %% [markdown]
# ## Push comparison to Hub

# %%
from skore import ComparisonReport  # noqa: E402

# Build full PCP + EBL on all training data for skore report_B
X_train_tsd = _add_tsd(X_train_raw)
X_test_tsd = _add_tsd(X_test_raw)

pcp_full = PatientCorrectedPooling(n_splits=5)
pcp_full.fit(X_train_tsd, y_train, patient_ids=groups_train.values)

X_train_pcp = pcp_full.transform(X_train_tsd)
X_test_pcp = pcp_full.transform(X_test_tsd)

ebl_full = EmpiricalBayesLine(ridge_alpha=1.0)
ebl_full.fit(X_train_pcp, y_train)


def _finalize(df_pcp, ebl):
    X = add_patient_features(ebl.transform(df_pcp))
    X["cohort"] = X["cohort"].astype(_COHORT_DTYPE)
    X["gene"] = X["gene"].astype(_GENE_DTYPE)
    X = X.drop(columns=["patient_id", "off_corr", "on_corr"], errors="ignore")
    return X

X_train_B = _finalize(X_train_pcp, ebl_full)
X_test_B = _finalize(X_test_pcp, ebl_full)

# Drop any non-categorical column that is constant in at least one CV fold
# (HGB's binning requires ≥2 distinct values for continuous features)
cv_splits_B = list(gkf.split(X_train_B, y_train, groups=groups_train))
_non_cat_B = [c for c in X_train_B.columns
              if not pd.api.types.is_categorical_dtype(X_train_B[c])]
_const_in_fold = set()
for _tr_idx, _ in cv_splits_B:
    _fold_slice = X_train_B.iloc[_tr_idx][_non_cat_B]
    _const_in_fold.update(c for c in _non_cat_B if _fold_slice[c].nunique(dropna=False) <= 1)
if _const_in_fold:
    X_train_B = X_train_B.drop(columns=list(_const_in_fold))
    X_test_B = X_test_B.drop(columns=list(_const_in_fold), errors="ignore")
    cv_splits_B = list(gkf.split(X_train_B, y_train, groups=groups_train))

model_B = HistGradientBoostingRegressor(random_state=0)
report_B = skore.evaluate(model_B, X_train_B, y_train, splitter=cv_splits_B)

# %%
# Extract RMSE for A and B from skore reports
summary_A = report_A.metrics.summarize().frame()
col_mean_A = [c for c in summary_A.columns if c.endswith("_mean")][0]
col_std_A = [c for c in summary_A.columns if c.endswith("_std")][0]
mean_A = float(summary_A.loc["rmse", col_mean_A])
std_A = float(summary_A.loc["rmse", col_std_A])

summary_B = report_B.metrics.summarize().frame()
col_mean_B = [c for c in summary_B.columns if c.endswith("_mean")][0]
col_std_B = [c for c in summary_B.columns if c.endswith("_std")][0]
mean_B = float(summary_B.loc["rmse", col_mean_B])
std_B = float(summary_B.loc["rmse", col_std_B])

print(f"A (control)         RMSE: {mean_A:.4f} ± {std_A:.4f}")
print(f"B (+EBL features)   RMSE: {mean_B:.4f} ± {std_B:.4f}")
print(f"C (posterior line)  RMSE: {rmse_C_mean:.4f} ± {rmse_C_std:.4f}")

# Push individual reports to hub (ComparisonReport is not supported by project.put)
project.put(f"{EXPERIMENT_KEY}_A", report_A)
project.put(f"{EXPERIMENT_KEY}_B", report_B)
comparison = ComparisonReport({"A_control": report_A, "B_ebl_features": report_B})
print(f"Pushed reports to hub keys: {EXPERIMENT_KEY}_A, {EXPERIMENT_KEY}_B")

# %% [markdown]
# ## Conditional: best variant submission

# %%
best_mean = min(mean_A, mean_B, rmse_C_mean)
best_label = (
    "A" if best_mean == mean_A else
    "B" if best_mean == mean_B else "C"
)
print(f"Best variant: {best_label}  RMSE: {best_mean:.4f}")

if best_mean < mean_A:
    print(f"{best_label} beats A — producing holdout report and submission")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    hold_tr_idx, hold_val_idx = next(gss.split(X_train_raw, y_train, groups=groups_train))

    X_hold_tr_raw = _add_tsd(X_train_raw.iloc[hold_tr_idx])
    X_hold_val_raw = _add_tsd(X_train_raw.iloc[hold_val_idx])
    y_hold_tr = y_train.iloc[hold_tr_idx]
    y_hold_val = y_train.iloc[hold_val_idx]
    pids_hold_tr = groups_train.iloc[hold_tr_idx].values

    pcp_hold = PatientCorrectedPooling(n_splits=5)
    pcp_hold.fit(X_hold_tr_raw, y_hold_tr, patient_ids=pids_hold_tr)
    X_hold_tr_pcp = pcp_hold.transform(X_hold_tr_raw)
    X_hold_val_pcp = pcp_hold.transform(X_hold_val_raw)

    ebl_hold = EmpiricalBayesLine(ridge_alpha=1.0)
    ebl_hold.fit(X_hold_tr_pcp, y_hold_tr)

    X_hold_tr_final = _finalize(X_hold_tr_pcp, ebl_hold)
    X_hold_val_final = _finalize(X_hold_val_pcp, ebl_hold)

    # Drop constant columns from holdout (same issue as in CV folds)
    _non_cat_h = [c for c in X_hold_tr_final.columns
                  if not pd.api.types.is_categorical_dtype(X_hold_tr_final[c])]
    _const_hold = [c for c in _non_cat_h
                   if X_hold_tr_final[c].nunique(dropna=False) <= 1]
    if _const_hold:
        X_hold_tr_final = X_hold_tr_final.drop(columns=_const_hold)
        X_hold_val_final = X_hold_val_final.drop(columns=_const_hold, errors="ignore")

    from skore import EstimatorReport  # noqa: E402

    report_best_final = EstimatorReport(
        HistGradientBoostingRegressor(random_state=0),
        X_train=X_hold_tr_final,
        y_train=y_hold_tr,
        X_test=X_hold_val_final,
        y_test=y_hold_val,
    )

    project.put("06_bayes_final", report_best_final)
    print("Pushed 06_bayes_final to hub")

    # Submission: fit best model on all train data, predict X_test
    if best_label == "C":
        preds_test = X_test_pcp.copy()
        preds_test = ebl_full.transform(preds_test)["eb_line_val"].values
        pop_mean_all = float(y_train.mean())
        preds_test = np.where(np.isnan(preds_test), pop_mean_all, preds_test)
    else:
        # B (or A, but A would not have beaten itself)
        model_final = HistGradientBoostingRegressor(random_state=0)
        model_final.fit(X_train_B, y_train)
        preds_test = model_final.predict(X_test_B)

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col=None)
    submission = pd.DataFrame({"Index": X_test_raw.index, "target": preds_test})
    assert list(submission["Index"]) == list(sample_sub["Index"]), \
        "Index mismatch with sample_submission.csv"

    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    out_path = submissions_dir / "06_bayes.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission written: {len(submission)} rows -> submissions/06_bayes.csv")
else:
    print("No variant beats A — no submission produced")
