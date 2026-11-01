import numpy as np
import pytest

from emailexp import policy as po
from emailexp import uplift as up


def _rct(n=60_000, seed=0):
    """Synthetic 3-arm experiment where only customers with x = 1 respond to email 1."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n)
    T = rng.integers(0, 3, n)
    Y = rng.normal(5, 1, n) + 4 * x * (T == 1)
    return x, T, Y


def test_qini_rewards_a_good_ranking():
    x, T, Y = _rct()
    good = up.qini_coefficient(up.qini_curve(x.astype(float), T, Y, arm=1))
    bad = up.qini_coefficient(up.qini_curve(-x.astype(float), T, Y, arm=1))
    rand = up.qini_coefficient(up.qini_curve(np.random.default_rng(1).random(len(x)), T, Y, arm=1))
    assert good > abs(rand) * 5 and bad < 0


def test_qini_curve_ends_at_average_lift():
    x, T, Y = _rct()
    curve = up.qini_curve(np.zeros(len(x)), T, Y, arm=1)
    mask = np.isin(T, [0, 1])
    true_total = (Y[T == 1].mean() - Y[T == 0].mean()) * (T == 1).sum() / mask.sum()
    assert curve["qini"].iloc[-1] == pytest.approx(true_total, rel=0.05)


def test_ipw_recovers_the_true_policy_value():
    x, T, Y = _rct()
    prop = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
    targeted = np.where(x == 1, 1, 0)
    lift = po.ipw_value(targeted, T, Y, prop, margin=1.0, cost=0.0) - po.ipw_value(
        np.zeros_like(T), T, Y, prop, margin=1.0, cost=0.0
    )
    assert lift == pytest.approx(4 * 0.5, abs=0.15)  # half the customers gain 4


def test_best_arm_policy_skips_unprofitable_customers():
    uplift = np.array([[10.0, 1.0], [0.1, 0.2], [2.0, 5.0]])
    assert po.best_arm_policy(uplift, margin=0.5, cost=0.3).tolist() == [1, 0, 2]


def test_top_share_policy_emails_the_requested_share():
    uplift = np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 0.5], [0.1, 0.2]])
    assert po.top_share_policy(uplift, 0.5).tolist() == [0, 2, 1, 0]
    assert (po.top_share_policy(uplift, 0.0) == 0).all()
    assert (po.top_share_policy(uplift, 1.0) > 0).all()


def test_spend_per_incremental_visit():
    T = np.array([0, 0, 1, 1, 2, 2])
    visit = np.array([0, 0, 1, 1, 1, 0], dtype=float)
    spend = np.array([0, 0, 4, 2, 6, 0], dtype=float)
    assert up.spend_per_incremental_visit(T, spend, visit).tolist() == [3.0, 6.0]
