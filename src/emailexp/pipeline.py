"""End-to-end analysis: data -> experiment checks -> treatment effects -> uplift models -> targeting -> site.

python -m emailexp.pipeline                 # full analysis (~5 minutes)
python -m emailexp.pipeline --quick         # small, fast demo run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import data
from . import experiment as ex
from . import policy as po
from . import uplift as up
from .config import Config
from .readme import update_readme
from .report import build_site

log = logging.getLogger("emailexp")


def run(cfg: Config) -> dict:
    t0 = time.time()
    path = data.ensure_data(cfg.data_path)
    con = data.connect(path, cfg.sample_rows, cfg.seed)
    arm_summary = data.run_sql(con, cfg.sql_dir / "arm_summary.sql")
    segment_lift = data.run_sql(con, cfg.sql_dir / "segment_lift.sql")
    df = data.customers(con)
    T = df["arm"].to_numpy()
    spend, visit = df["spend"].to_numpy(float), df["visit"].to_numpy(float)
    log.info("Loaded %s customers", f"{len(df):,}")

    # 1. Is the experiment trustworthy?
    counts = df["arm"].value_counts().sort_index().to_numpy()
    srm_p = ex.srm_test(counts)
    balance = ex.balance_table(df)

    # 2. Average effects of each email.
    effects = ex.effects_table(df, cfg.alpha, cfg.power, cfg.bootstrap_reps, cfg.seed)
    mde = {o: ex.mde_curve(df, o, cfg.alpha, cfg.power) for o in ("visit", "spend")}
    log.info("Effects:\n%s", effects[["arm", "outcome", "lift", "ci_low", "ci_high", "p_holm"]].round(4))

    # 3. Who responds? Out-of-fold uplift predictions for every customer.
    X = data.model_matrix(df)
    preds = up.crossfit_predictions(
        X,
        T,
        {"visits": visit, "spend": spend},
        cfg.crossfit_folds,
        cfg.forest_trees,
        cfg.nuisance_trees,
        cfg.seed,
    )
    ratio = up.spend_per_incremental_visit(T, spend, visit)
    # Visit models predict extra visits; convert to extra dollars so every model is scored on spend.
    dollar_scores = {name: (p * ratio if "(visits)" in name else p) for name, p in preds.items()}

    rng = np.random.default_rng(cfg.seed)
    ranking = {
        **dollar_scores,
        "Rule: matches past purchases": up.rule_scores(df),
        "Random": rng.random((len(df), 2)),
    }
    qini_rows, curves = [], {}
    for arm in up.ARMS:
        for name, s in ranking.items():
            for outcome, y in (("spend", spend), ("visit", visit)):
                curve = up.qini_curve(s[:, arm - 1], T, y, arm)
                curves[(arm, outcome, name)] = curve
                qini_rows.append(
                    {"arm": arm, "model": name, "outcome": outcome, "qini": up.qini_coefficient(curve)}
                )
    qini = pd.DataFrame(qini_rows)
    model_rank = (
        qini[(qini["outcome"] == "spend") & ~qini["model"].isin(["Random", "Rule: matches past purchases"])]
        .groupby("model")["qini"]
        .mean()
        .sort_values(ascending=False)
    )
    champion = model_rank.index[0]
    log.info("Qini (spend, mean over emails):\n%s\nChampion: %s", model_rank.round(5), champion)

    # 4. Turn uplift into a targeting policy and value it.
    propensity = {int(k): float(v) for k, v in df["arm"].value_counts(normalize=True).items()}
    policies = po.evaluate_policies(
        dollar_scores, df, propensity, cfg.margin, cfg.email_cost, cfg.policy_bootstrap_reps, cfg.seed
    )
    budget_curve = pd.DataFrame(
        [
            {
                "budget": b,
                "incremental_profit": (
                    po.ipw_value(
                        po.top_share_policy(dollar_scores[champion], b),
                        T,
                        spend,
                        propensity,
                        cfg.margin,
                        cfg.email_cost,
                    )
                    - po.ipw_value(
                        np.zeros(len(T), dtype=int), T, spend, propensity, cfg.margin, cfg.email_cost
                    )
                )
                * po.PER,
            }
            for b in np.linspace(0, 1, 21)
        ]
    )

    # 5. What drives the predicted uplift?
    shap_importance, surrogate_r2 = up.shap_of_uplift(X, dollar_scores[champion], cfg.seed)

    results = {
        "n_customers": int(len(df)),
        "arm_counts": {int(k): int(v) for k, v in zip(range(3), counts, strict=True)},
        "srm_p": srm_p,
        "max_abs_smd": float(balance["smd"].abs().max()),
        "champion": champion,
        "spend_per_visit": ratio.tolist(),
        "surrogate_r2": surrogate_r2,
        "margin": cfg.margin,
        "email_cost": cfg.email_cost,
        "crossfit_folds": cfg.crossfit_folds,
    }
    simulator = {
        "T": T.tolist(),
        "Y": np.round(spend, 2).tolist(),
        "s1": np.round(dollar_scores[champion][:, 0], 4).tolist(),
        "s2": np.round(dollar_scores[champion][:, 1], 4).tolist(),
        "p": [propensity[0], propensity[1], propensity[2]],
        "per": po.PER,
    }
    build_site(
        cfg.site_dir,
        results=results,
        arm_summary=arm_summary,
        segment_lift=segment_lift,
        balance=balance,
        effects=effects,
        mde=mde,
        qini=qini,
        curves=curves,
        policies=policies,
        budget_curve=budget_curve,
        shap_importance=shap_importance,
        simulator=simulator,
    )
    summary = {
        **results,
        "effects": effects.round(6).to_dict(orient="records"),
        "qini": qini.round(6).to_dict(orient="records"),
        "policies": policies.round(2).to_dict(orient="records"),
        "runtime_sec": round(time.time() - t0, 1),
    }
    (cfg.site_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    if cfg.readme_path.exists() and not cfg.sample_rows:
        update_readme(cfg.readme_path, results, effects, policies, qini)
    log.info("Done in %.0fs. Site: %s", time.time() - t0, cfg.site_dir / "index.html")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="fast demo on 8,000 customers with small models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    if args.quick:
        cfg = replace(
            cfg,
            sample_rows=8000,
            crossfit_folds=2,
            forest_trees=100,
            nuisance_trees=60,
            bootstrap_reps=200,
            policy_bootstrap_reps=100,
        )
    run(cfg)


if __name__ == "__main__":
    main()
