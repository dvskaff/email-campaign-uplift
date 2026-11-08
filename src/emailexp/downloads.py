"""Customer-level download: every customer with the email they were randomly sent, what they did, the champion
model's out-of-fold predicted extra spend from each email, and the email the targeting policy would send them.
Written to site/data/ on every run and linked from the report."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import best_arm_policy

log = logging.getLogger(__name__)
EMAILS = {0: "No email", 1: "Men's email", 2: "Women's email"}
COLUMNS = {
    "customer_id": "Row number in the original file",
    "recency": "Months since last purchase", "history_segment": "Past-year spend band",
    "history": "Dollars spent in the past year", "mens": "Bought men's merchandise in the past year (1/0)",
    "womens": "Bought women's merchandise in the past year (1/0)", "zip_code": "Urban, Suburban or Rural",
    "newbie": "New customer in the past twelve months (1/0)", "channel": "Purchase channels in the past year",
    "email_sent": "The email the customer was randomly assigned", "visit": "Visited the website within two weeks (1/0)",
    "conversion": "Bought within two weeks (1/0)", "spend": "Dollars spent within two weeks",
    "uplift_mens_email": "Predicted extra spend ($) from the men's email (out-of-fold: from a model that never saw "
                         "this customer)",
    "uplift_womens_email": "Predicted extra spend ($) from the women's email (out-of-fold)",
    "recommended_email": "What the targeting policy would send, at the report's margin and email cost",
}  # fmt: skip


def write(
    site_dir: Path, df: pd.DataFrame, uplift: np.ndarray, margin: float, cost: float, model: str
) -> dict:
    out = df[["customer_id", "recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie",
              "channel"]].copy()  # fmt: skip
    out["email_sent"] = df["arm"].map(EMAILS)
    out[["visit", "conversion", "spend"]] = df[["visit", "conversion", "spend"]]
    out["uplift_mens_email"] = np.round(uplift[:, 0], 4)
    out["uplift_womens_email"] = np.round(uplift[:, 1], 4)
    out["recommended_email"] = pd.Series(best_arm_policy(uplift, margin, cost)).map(EMAILS).to_numpy()
    path = site_dir / "data" / "customer_uplift_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    meta = {"file": "data/customer_uplift_scores.csv", "rows": len(out), "kb": round(path.stat().st_size / 1024),
            "model": model, "columns": COLUMNS, "recommended": out["recommended_email"].value_counts().to_dict()}  # fmt: skip
    log.info("Customer download: %s rows (%s KB)", f"{len(out):,}", meta["kb"])
    return meta
