"""Rewrites the results block in README.md after each run."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OUTCOME_LABEL = {"visit": "Visit rate", "conversion": "Purchase rate", "spend": "Spend per customer"}


def _fmt(outcome: str, value: float) -> str:
    return (
        f"{'+' if value >= 0 else '−'}${abs(value):.2f}" if outcome == "spend" else f"{value * 100:+.2f} pts"
    )


def update_readme(
    readme: Path, results: dict, effects: pd.DataFrame, policies: pd.DataFrame, qini: pd.DataFrame
) -> None:
    lines = [
        f"_{results['n_customers']:,} customers · sample-ratio check p = {results['srm_p']:.2f} · "
        f"largest covariate imbalance |SMD| = {results['max_abs_smd']:.3f}_",
        "",
        "| Email vs. no email | Outcome | Lift (95% CI) | Relative lift | Holm-adjusted p |",
        "|---|---|---|---|---|",
    ]
    for r in effects.itertuples(index=False):
        lines.append(
            f"| {r.arm} | {OUTCOME_LABEL[r.outcome]} | {_fmt(r.outcome, r.lift)} "
            f"({_fmt(r.outcome, r.ci_low)} to {_fmt(r.outcome, r.ci_high)}) | {r.relative_lift:+.0%} | "
            f"{'< 0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}'} |"
        )
    best = policies.iloc[0]
    blast = policies.set_index("policy").loc["Men's email to everyone"]
    spend_q = qini[qini["outcome"] == "spend"].pivot_table(index="model", columns="arm", values="qini")
    champ = results["champion"]
    lines += [
        "",
        f"**Best uplift model:** {champ} (out-of-fold Qini on spend: men's email "
        f"{spend_q.loc[champ, 1]:.3f} vs. {spend_q.loc['Random', 1]:.3f} for random targeting).",
        "",
        f"**Best targeting policy** at a {results['margin']:.0%} margin and ${results['email_cost']:.2f} per email: "
        f"{best.policy}, worth **${best.incremental_profit:,.0f}** in incremental profit per 100,000 customers "
        f"(90% CI ${best.ci_low:,.0f} to ${best.ci_high:,.0f}), vs. ${blast.incremental_profit:,.0f} for sending "
        f"the men's email to everyone.",
    ]
    block = "\n".join(lines)
    text = readme.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)"
    readme.write_text(re.sub(pattern, lambda m: m.group(1) + block + m.group(2), text, flags=re.S))
