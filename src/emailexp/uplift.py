"""Heterogeneous treatment effects (who responds to which email) and how well each model ranks customers."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from econml.dml import CausalForestDML
from econml.metalearners import XLearner

ARMS = (1, 2)


def _lgb_reg(n: int, seed: int):
    return lgb.LGBMRegressor(
        n_estimators=n,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )


def _lgb_clf(n: int, seed: int):
    return lgb.LGBMClassifier(
        n_estimators=n, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1
    )


# --------------------------------------------------------------------------- models
def fit_causal_forest(X, T, Y, trees: int, nuisance_trees: int, seed: int) -> CausalForestDML:
    """Causal forest (double machine learning) with LightGBM models for the outcome and treatment."""
    model = CausalForestDML(
        model_y=_lgb_reg(nuisance_trees, seed),
        model_t=_lgb_clf(nuisance_trees // 2, seed),
        discrete_treatment=True,
        categories=[0, 1, 2],
        n_estimators=trees,
        min_samples_leaf=50,
        max_samples=0.45,
        cv=3,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Y, T, X=X)
    return model


def fit_x_learner(X, T, Y, nuisance_trees: int, seed: int) -> XLearner:
    model = XLearner(
        models=_lgb_reg(nuisance_trees, seed), propensity_model=_lgb_clf(50, seed), categories=[0, 1, 2]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Y, T, X=X)
    return model


def predict_uplift(model, X) -> np.ndarray:
    """(n_customers, 2) predicted lift for the men's and women's email vs. no email."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.const_marginal_effect(X)).reshape(len(X), 2)


def rule_scores(test: pd.DataFrame) -> np.ndarray:
    """Business-rule baseline: send each customer the email matching what they bought before."""
    return np.column_stack([test["mens"].to_numpy(float), test["womens"].to_numpy(float)])


# --------------------------------------------------------------------------- Qini evaluation
def qini_curve(score: np.ndarray, T: np.ndarray, Y: np.ndarray, arm: int, points: int = 101) -> pd.DataFrame:
    """Cumulative incremental outcome when targeting customers in order of predicted uplift.

    Uses only customers in `arm` or control. At each cut-off the treated outcome is compared with the
    control outcome rescaled to the treated count (the standard Qini definition)."""
    mask = np.isin(T, [0, arm])
    s, t, y = score[mask], (T[mask] == arm).astype(float), Y[mask].astype(float)
    order = np.argsort(-s, kind="stable")
    t, y = t[order], y[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    idx = np.unique(np.linspace(0, len(q) - 1, points).astype(int))
    frac = (idx + 1) / len(q)
    return pd.DataFrame({"fraction": np.r_[0.0, frac], "qini": np.r_[0.0, q[idx]] / mask.sum()})


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and the random-targeting line (higher = better ranking)."""
    random_line = curve["fraction"] * curve["qini"].iloc[-1]
    return float(np.trapezoid(curve["qini"] - random_line, curve["fraction"]))


# --------------------------------------------------------------------------- explanation
def shap_of_uplift(X: pd.DataFrame, uplift: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict]:
    """SHAP for the causal forest via a LightGBM surrogate fit to its predicted uplift.

    The forest itself only supports slow model-agnostic SHAP; a tree surrogate gives exact TreeSHAP
    values in seconds, and its R² shows how faithfully it mimics the forest."""
    rows, fit_quality = [], {}
    for j, arm in enumerate(ARMS):
        surrogate = _lgb_reg(300, seed).fit(X, uplift[:, j])
        fit_quality[arm] = float(surrogate.score(X, uplift[:, j]))
        values = shap.TreeExplainer(surrogate).shap_values(X)
        for col, v in zip(X.columns, np.abs(values).mean(axis=0), strict=True):
            rows.append({"arm": arm, "feature": col, "mean_abs_shap": float(v)})
    imp = pd.DataFrame(rows)
    order = imp.groupby("feature")["mean_abs_shap"].sum().sort_values(ascending=False).index
    imp["feature"] = pd.Categorical(imp["feature"], categories=order, ordered=True)
    return imp.sort_values(["feature", "arm"]).reset_index(drop=True), fit_quality


PRETTY = {
    "recency": "Months since last purchase",
    "history": "Prior-year spend ($)",
    "mens": "Bought men's before",
    "womens": "Bought women's before",
    "newbie": "New customer",
    "zip_code_Rural": "Area: rural",
    "zip_code_Suburban": "Area: suburban",
    "zip_code_Urban": "Area: urban",
    "channel_Multichannel": "Channel: multichannel",
    "channel_Phone": "Channel: phone",
    "channel_Web": "Channel: web",
}


# --------------------------------------------------------------------------- cross-fitting
def crossfit_predictions(
    X: pd.DataFrame,
    T: np.ndarray,
    outcomes: dict[str, np.ndarray],
    folds: int,
    trees: int,
    nuisance_trees: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Out-of-fold uplift predictions for every customer.

    Each fold's predictions come from models that never saw those customers, so all 64,000 customers can
    be used for honest evaluation instead of a single small hold-out set."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(T), dtype=int)
    for arm in np.unique(T):  # stratify folds by arm
        idx = rng.permutation(np.flatnonzero(T == arm))
        fold[idx] = np.arange(len(idx)) % folds
    preds: dict[str, np.ndarray] = {}
    for k in range(folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        for out_name, y in outcomes.items():
            fitted = {
                f"Causal forest ({out_name})": fit_causal_forest(
                    X.iloc[tr], T[tr], y[tr], trees, nuisance_trees, seed
                ),
                f"X-learner ({out_name})": fit_x_learner(X.iloc[tr], T[tr], y[tr], nuisance_trees, seed),
            }
            for name, model in fitted.items():
                preds.setdefault(name, np.zeros((len(T), 2)))[te] = predict_uplift(model, X.iloc[te])
    return preds


def spend_per_incremental_visit(T: np.ndarray, spend: np.ndarray, visit: np.ndarray) -> np.ndarray:
    """Average extra dollars per extra visit for each email, used to turn visit uplift into $ uplift."""
    ratios = []
    for arm in ARMS:
        d_spend = spend[T == arm].mean() - spend[T == 0].mean()
        d_visit = visit[T == arm].mean() - visit[T == 0].mean()
        ratios.append(d_spend / d_visit if d_visit > 0 else 0.0)
    return np.array(ratios)
