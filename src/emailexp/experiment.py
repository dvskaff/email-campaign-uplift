"""Classical A/B test analysis: health checks, lift with confidence intervals, Holm correction,
CUPED / regression adjustment, bootstrap, and minimum detectable effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import ARM_NAMES, OUTCOMES

BALANCE_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]


# --------------------------------------------------------------------------- health checks
def srm_test(counts: list[int] | np.ndarray, expected_shares: list[float] | None = None) -> float:
    """Sample-ratio mismatch: chi-square test that arms received the planned share of customers."""
    counts = np.asarray(counts, dtype=float)
    shares = np.full(len(counts), 1 / len(counts)) if expected_shares is None else np.asarray(expected_shares)
    return float(stats.chisquare(counts, counts.sum() * shares).pvalue)


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized mean difference of each pre-treatment covariate, email arm vs. no email.
    |SMD| < 0.1 is the usual threshold for 'balanced'."""
    control = df[df["arm"] == 0]
    rows = []
    for arm in (1, 2):
        treated = df[df["arm"] == arm]
        for cov in BALANCE_COVARIATES:
            pooled_sd = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
            smd = (treated[cov].mean() - control[cov].mean()) / pooled_sd if pooled_sd > 0 else 0.0
            rows.append({"arm": ARM_NAMES[arm], "covariate": cov, "smd": float(smd)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- estimators
def diff_in_means(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Difference in means with a Welch (unequal-variance) normal-approximation interval."""
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    est = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(est / se)) if se > 0 else 1.0
    return {
        "control_mean": float(y_c.mean()),
        "treated_mean": float(y_t.mean()),
        "lift": float(est),
        "se": float(se),
        "ci_low": float(est - z * se),
        "ci_high": float(est + z * se),
        "p_value": float(p),
        "relative_lift": float(est / y_c.mean()) if y_c.mean() else np.nan,
    }


def cuped(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: subtract the part of the outcome explained by a pre-experiment covariate.
    theta is pooled across arms, so the adjustment cannot depend on treatment."""
    y, x_pre = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / x_pre.var(ddof=1)
    return y - theta * (x_pre - x_pre.mean()), float(theta)


def regression_adjusted(df: pd.DataFrame, outcome: str, arm: int, covariates: list[str]) -> dict:
    """OLS of the outcome on the treatment flag + centered pre-treatment covariates (Lin 2013),
    with heteroskedasticity-robust (HC2) standard errors. Generalizes CUPED to many covariates."""
    sub = df[df["arm"].isin([0, arm])]
    X = pd.get_dummies(sub[covariates], drop_first=True, dtype=float)
    X = X - X.mean()
    X.insert(0, "treat", (sub["arm"] == arm).astype(float))
    X = sm.add_constant(X)
    fit = sm.OLS(sub[outcome].astype(float), X).fit(cov_type="HC2")
    ci = fit.conf_int().loc["treat"]
    return {
        "lift": float(fit.params["treat"]),
        "se": float(fit.bse["treat"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def bootstrap_ci(y_t: np.ndarray, y_c: np.ndarray, reps: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap interval for the difference in means (robust to heavy-tailed spend)."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    t_means = np.array([rng.choice(y_t, len(y_t)).mean() for _ in range(reps)])
    c_means = np.array([rng.choice(y_c, len(y_c)).mean() for _ in range(reps)])
    diffs = t_means - c_means
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def mde(n_t: int, n_c: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true lift detectable with the given power in a two-sided test."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd * np.sqrt(1 / n_t + 1 / n_c))


# --------------------------------------------------------------------------- headline table
def effects_table(df: pd.DataFrame, alpha: float, power: float, reps: int, seed: int) -> pd.DataFrame:
    """Every email arm vs. no email on every outcome: raw, CUPED and regression-adjusted estimates,
    Holm-adjusted p-values across all 6 tests, bootstrap CI for spend, and the MDE."""
    covariates = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
    rows = []
    for outcome in OUTCOMES:
        y_cuped, theta = cuped(df[outcome].to_numpy(), df["history"].to_numpy())
        control = df["arm"].to_numpy() == 0
        for arm in (1, 2):
            treated = df["arm"].to_numpy() == arm
            raw = diff_in_means(df.loc[treated, outcome], df.loc[control, outcome], alpha)
            adj = diff_in_means(y_cuped[treated], y_cuped[control], alpha)
            reg = regression_adjusted(df, outcome, arm, covariates)
            row = {
                "arm": ARM_NAMES[arm],
                "outcome": outcome,
                **raw,
                "cuped_lift": adj["lift"],
                "cuped_ci_low": adj["ci_low"],
                "cuped_ci_high": adj["ci_high"],
                "cuped_theta": theta,
                "cuped_var_reduction": 1 - (adj["se"] / raw["se"]) ** 2,
                "reg_lift": reg["lift"],
                "reg_ci_low": reg["ci_low"],
                "reg_ci_high": reg["ci_high"],
                "reg_var_reduction": 1 - (reg["se"] / raw["se"]) ** 2,
                "mde": mde(treated.sum(), control.sum(), df.loc[control, outcome].std(), alpha, power),
                "n_treated": int(treated.sum()),
                "n_control": int(control.sum()),
            }
            if outcome == "spend":
                row["boot_ci_low"], row["boot_ci_high"] = bootstrap_ci(
                    df.loc[treated, outcome], df.loc[control, outcome], reps, seed + arm, alpha
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"], alpha=alpha, method="holm")[1]
    out["significant"] = out["p_holm"] < alpha
    return out


def mde_curve(df: pd.DataFrame, outcome: str, alpha: float, power: float) -> pd.DataFrame:
    """MDE (as % of the control mean) for a range of per-arm sample sizes."""
    control = df.loc[df["arm"] == 0, outcome]
    sizes = np.unique(np.geomspace(1_000, 200_000, 40).astype(int))
    return pd.DataFrame(
        {
            "n_per_arm": sizes,
            "mde_relative": [mde(n, n, control.std(), alpha, power) / control.mean() for n in sizes],
        }
    )
