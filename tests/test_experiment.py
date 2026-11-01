import numpy as np
import pandas as pd
import pytest

from emailexp import experiment as ex


def test_diff_in_means_recovers_known_lift():
    rng = np.random.default_rng(0)
    y_c = rng.normal(10, 2, 50_000)
    y_t = rng.normal(10.5, 2, 50_000)
    r = ex.diff_in_means(y_t, y_c)
    assert r["ci_low"] < 0.5 < r["ci_high"]
    assert r["p_value"] < 1e-6
    assert r["relative_lift"] == pytest.approx(0.05, abs=0.01)


def test_diff_in_means_no_effect_is_not_significant():
    rng = np.random.default_rng(1)
    r = ex.diff_in_means(rng.normal(0, 1, 5_000), rng.normal(0, 1, 5_000))
    assert r["p_value"] > 0.01


def test_cuped_reduces_variance_with_a_predictive_covariate():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 20_000)
    y = 3 * x + rng.normal(0, 1, 20_000)
    y_adj, theta = ex.cuped(y, x)
    assert theta == pytest.approx(3, abs=0.1)
    assert y_adj.var() < 0.2 * y.var()
    assert y_adj.mean() == pytest.approx(y.mean())  # CUPED never shifts the mean


def test_srm_flags_broken_randomization():
    assert ex.srm_test([10_000, 10_050, 9_980]) > 0.05
    assert ex.srm_test([10_000, 10_000, 9_000]) < 1e-6


def test_mde_shrinks_with_sample_size():
    small, big = ex.mde(1_000, 1_000, 1.0), ex.mde(100_000, 100_000, 1.0)
    assert big == pytest.approx(small / 10, rel=1e-6)
    assert ex.mde(10_000, 10_000, 1.0) == pytest.approx(2.8016 * np.sqrt(2 / 10_000), rel=1e-3)


def _fake_experiment(n=6_000, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "arm": rng.integers(0, 3, n),
            "recency": rng.integers(1, 13, n),
            "history": rng.gamma(2, 120, n),
            "mens": rng.integers(0, 2, n),
            "newbie": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Suburban", "Rural"], n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        }
    )
    df["womens"] = 1 - df["mens"]
    df["visit"] = (rng.random(n) < 0.10 + 0.08 * (df["arm"] == 1)).astype(int)
    df["conversion"] = (df["visit"] & (rng.random(n) < 0.1)).astype(int)
    df["spend"] = df["conversion"] * rng.gamma(2, 50, n)
    return df


def test_effects_table_shape_and_holm():
    df = _fake_experiment()
    t = ex.effects_table(df, 0.05, 0.8, reps=100, seed=0)
    assert len(t) == 6
    assert (t["p_holm"] >= t["p_value"] - 1e-12).all()  # Holm never makes p-values smaller
    men_visit = t[(t["arm"] == "Men's email") & (t["outcome"] == "visit")].iloc[0]
    assert men_visit["significant"]
    assert men_visit["reg_lift"] == pytest.approx(men_visit["lift"], abs=0.02)


def test_balance_table_is_small_for_randomized_groups():
    bal = ex.balance_table(_fake_experiment())
    assert bal["smd"].abs().max() < 0.1
