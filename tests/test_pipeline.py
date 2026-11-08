import json
from dataclasses import replace

from emailexp import data
from emailexp.config import Config
from emailexp.pipeline import run


def test_dataset_loads_and_is_cleaned():
    cfg = Config()
    con = data.connect(cfg.data_path)
    df = data.customers(con)
    assert len(df) == 64_000
    assert set(df["zip_code"]) == {"Urban", "Suburban", "Rural"}  # 'Surburban' typo fixed
    assert df["arm"].value_counts().min() > 21_000
    X = data.model_matrix(df)
    assert not {"visit", "conversion", "spend", "segment"} & set(X.columns)  # no outcome leakage


def test_end_to_end_quick_run(tmp_path):
    cfg = replace(
        Config(),
        site_dir=tmp_path / "site",
        readme_path=tmp_path / "README.md",
        sample_rows=4_000,
        crossfit_folds=2,
        forest_trees=40,
        nuisance_trees=30,
        bootstrap_reps=50,
        policy_bootstrap_reps=30,
    )
    summary = run(cfg)
    site = cfg.site_dir
    assert (site / "index.html").stat().st_size > 50_000
    sim = json.loads((site / "simulator-data.json").read_text())
    assert len(sim["T"]) == len(sim["Y"]) == len(sim["s1"]) == 4_000
    assert summary["n_customers"] == 4_000
    assert {p["policy"] for p in summary["policies"]} >= {"Email no one", "Men's email to everyone"}
    # customer-level download: one row per customer, scores match the simulator's, recommendations are valid
    import pandas as pd

    scores = pd.read_csv(site / "data" / "customer_uplift_scores.csv")
    assert len(scores) == 4_000 and scores["customer_id"].is_unique
    assert (scores["uplift_mens_email"].round(4) == pd.Series(sim["s1"]).round(4)).all()
    assert set(scores["email_sent"]) == {"No email", "Men's email", "Women's email"}
    assert set(scores["recommended_email"]) <= {"No email", "Men's email", "Women's email"}
    assert "customer_uplift_scores.csv" in (site / "index.html").read_text()
