"""Project settings. Override from the command line (see pipeline.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARMS = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
ARM_NAMES = {0: "No email", 1: "Men's email", 2: "Women's email"}
OUTCOMES = {"visit": "Website visit", "conversion": "Purchase", "spend": "Spend ($)"}


@dataclass
class Config:
    data_path: Path = field(default_factory=lambda: ROOT / "data" / "hillstrom.csv")
    sql_dir: Path = field(default_factory=lambda: ROOT / "sql")
    site_dir: Path = field(default_factory=lambda: ROOT / "site")
    readme_path: Path = field(default_factory=lambda: ROOT / "README.md")

    alpha: float = 0.05  # family-wise error rate for the 6 headline tests (Holm)
    power: float = 0.80
    bootstrap_reps: int = 2000
    policy_bootstrap_reps: int = 400
    crossfit_folds: int = (
        5  # out-of-fold predictions so every customer is scored by a model that never saw them
    )
    seed: int = 42

    # uplift models
    forest_trees: int = 400
    nuisance_trees: int = 200

    # default economics for the targeting policy (adjustable in the website simulator)
    margin: float = 0.40  # gross margin on incremental spend
    email_cost: float = 0.10  # cost per email sent ($)

    sample_rows: int | None = None  # run on a random subset (tests / quick demo)
