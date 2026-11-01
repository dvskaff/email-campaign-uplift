"""Load the experiment with DuckDB and run the SQL summaries in sql/."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from .config import ARMS

log = logging.getLogger(__name__)

SOURCE_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
SHA256 = "27bab8c5d3669f26ec08ebb50a0a78317542f29501156f2e2af6781fab4cd7e2"
FEATURES = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]


def ensure_data(path: Path) -> Path:
    """The CSV ships with the repo; if it is missing, download the original and verify its checksum."""
    if not path.exists():
        log.info("Downloading the Hillstrom dataset from %s", SOURCE_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)  # noqa: S310 (fixed, known URL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        log.warning("Dataset checksum differs from the published file (%s)", digest[:12])
    return path


def connect(path: Path, sample_rows: int | None = None, seed: int = 42) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with a cleaned `customers` view (one row per customer)."""
    con = duckdb.connect()
    src = str(path).replace("'", "''")
    sample = f"USING SAMPLE {int(sample_rows)} ROWS (reservoir, {int(seed)})" if sample_rows else ""
    con.execute(f"""
        CREATE TABLE customers AS
        SELECT row_number() OVER () AS customer_id,
               recency, history_segment, history, mens, womens,
               replace(zip_code, 'Surburban', 'Suburban') AS zip_code,  -- fix a typo in the source file
               newbie, channel, segment, visit, conversion, spend
        FROM read_csv('{src}', header = true)
        {sample}
    """)
    return con


def run_sql(con: duckdb.DuckDBPyConnection, sql_file: Path) -> pd.DataFrame:
    return con.execute(sql_file.read_text()).df()


def customers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM customers ORDER BY customer_id").df()
    df["arm"] = df["segment"].map(ARMS).astype(int)
    return df


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment covariates only (never outcomes), one-hot encoded for the models."""
    X = pd.get_dummies(df[FEATURES], columns=["zip_code", "channel"], dtype=float)
    return X.astype(float)
