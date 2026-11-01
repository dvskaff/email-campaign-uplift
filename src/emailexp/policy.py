"""Turn predicted uplift into a targeting decision and value it with inverse-propensity weighting.

Because the experiment assigned emails at random with known probabilities, any targeting policy can be
evaluated without bias on held-out customers: keep the customers whose random assignment happens to
match what the policy would have done, and weight them by 1 / P(that assignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PER = 100_000  # report values per 100,000 customers


def best_arm_policy(uplift_spend: np.ndarray, margin: float, cost: float, budget: float = 1.0) -> np.ndarray:
    """For each customer pick the email with the highest expected incremental profit, or no email if
    neither pays for itself. With budget < 1, email only the top `budget` share by expected profit."""
    profit = margin * uplift_spend - cost  # (n, 2)
    best = profit.argmax(axis=1)
    best_profit = profit.max(axis=1)
    choice = np.where(best_profit > 0, best + 1, 0)
    if budget < 1.0:
        k = int(np.floor(budget * len(choice)))
        keep = np.zeros(len(choice), dtype=bool)
        keep[np.argsort(-best_profit, kind="stable")[:k]] = True
        choice = np.where(keep, choice, 0)
    return choice


def _weights(T: np.ndarray, Y: np.ndarray, propensity: dict[int, float], margin: float, cost: float):
    """Per-customer profit divided by the probability of the arm they were randomly assigned."""
    p = np.array([propensity[int(t)] for t in range(3)])[T]
    return (margin * Y - cost * (T > 0)) / p


def ipw_value(
    policy: np.ndarray, T: np.ndarray, Y: np.ndarray, propensity: dict[int, float], margin: float, cost: float
) -> float:
    """Expected profit per customer under `policy`: margin x spend - email cost."""
    return float(np.mean((policy == T) * _weights(T, Y, propensity, margin, cost)))


def evaluate_policies(
    scores: dict[str, np.ndarray],
    test: pd.DataFrame,
    propensity: dict[int, float],
    margin: float,
    cost: float,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Incremental profit vs. sending nothing, per 100k customers, with bootstrap 90% intervals."""
    T, Y = test["arm"].to_numpy(), test["spend"].to_numpy(float)
    n = len(T)
    w = _weights(T, Y, propensity, margin, cost)
    policies = {
        "Email no one": np.zeros(n, dtype=int),
        "Men's email to everyone": np.ones(n, dtype=int),
        "Women's email to everyone": np.full(n, 2),
        "Email matching past purchases": np.where(test["mens"].to_numpy() == 1, 1, 2),
        **{f"Targeted: {name}": best_arm_policy(s, margin, cost) for name, s in scores.items()},
    }
    boot_idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    baseline = (T == 0) * w
    rows = []
    for name, pol in policies.items():
        contribution = (pol == T) * w - baseline  # incremental profit vs. emailing no one
        boots = contribution[boot_idx].mean(axis=1)
        rows.append(
            {
                "policy": name,
                "incremental_profit": float(contribution.mean()) * PER,
                "ci_low": float(np.quantile(boots, 0.05)) * PER,
                "ci_high": float(np.quantile(boots, 0.95)) * PER,
                "share_emailed": float(np.mean(pol > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("incremental_profit", ascending=False).reset_index(drop=True)


def top_share_policy(uplift_spend: np.ndarray, share: float) -> np.ndarray:
    """Email the `share` of customers with the largest predicted $ uplift, each with their better email."""
    best = uplift_spend.argmax(axis=1) + 1
    k = int(round(share * len(best)))
    choice = np.zeros(len(best), dtype=int)
    top = np.argsort(-uplift_spend.max(axis=1), kind="stable")[:k]
    choice[top] = best[top]
    return choice
