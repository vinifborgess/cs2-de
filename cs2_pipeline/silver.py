from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import (
    KILLS_SCHEMA,
    ROUNDS_RENAME,
    ROUNDS_SCHEMA,
    SILVER_DIR,
    get_logger,
)
from .utils import coerce_schema, missing_columns

log = get_logger()


def _silver_path(table: str, match_id: str) -> Path:
    directory = SILVER_DIR / table
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"match_{match_id}.parquet"


def clean_kills(df_raw: pd.DataFrame, match_id: str) -> pd.DataFrame:
    df = df_raw.copy()
    df["match_id"] = match_id

    if "is_warmup" in df.columns:
        df = df[df["is_warmup"] == False]

    if "attacker_name" in df.columns:
        df = df.dropna(subset=["attacker_name"])

    if "weapon" in df.columns:
        df = df[df["weapon"].astype("string").str.lower() != "world"]

    if {"attacker_side", "victim_side"}.issubset(df.columns):
        df = df[df["attacker_side"] != df["victim_side"]]
    elif {"attacker_name", "victim_name"}.issubset(df.columns):
        df = df[df["attacker_name"] != df["victim_name"]]

    missing = missing_columns(df, KILLS_SCHEMA)
    if missing:
        log.info("kills[%s]: missing columns ignored: %s", match_id, missing)

    return coerce_schema(df, KILLS_SCHEMA)


def clean_rounds(df_raw: pd.DataFrame, match_id: str) -> pd.DataFrame:
    df = df_raw.copy()
    df["match_id"] = match_id
    df = df.rename(columns={k: v for k, v in ROUNDS_RENAME.items() if k in df.columns})

    if "start_tick" in df.columns:
        df = df.dropna(subset=["start_tick"])
        df = df[df["start_tick"] > 0]

    if "winner" in df.columns:
        df = df[df["winner"].astype("string").str.strip().str.upper().isin(["CT", "T"])]

    missing = missing_columns(df, ROUNDS_SCHEMA)
    if missing:
        log.info("rounds[%s]: missing columns ignored: %s", match_id, missing)

    return coerce_schema(df, ROUNDS_SCHEMA)


def bronze_to_silver(
    bronze: dict[str, pd.DataFrame],
    match_id: str,
    *,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Apply cleaning rules and persist to the Silver layer. Idempotent per match."""
    targets = {
        "kills": _silver_path("kills", match_id),
        "rounds": _silver_path("rounds", match_id),
    }

    if not force and all(p.exists() for p in targets.values()):
        log.info("Silver already exists for '%s' — skipping cleaning.", match_id)
        return {t: pd.read_parquet(p) for t, p in targets.items()}

    cleaned_kills = clean_kills(bronze["kills"], match_id)
    cleaned_rounds = clean_rounds(bronze["rounds"], match_id)

    cleaned_kills.to_parquet(targets["kills"], engine="pyarrow", index=False)
    cleaned_rounds.to_parquet(targets["rounds"], engine="pyarrow", index=False)
    log.info("Silver[%s]: %d kills, %d rounds.", match_id, len(cleaned_kills), len(cleaned_rounds))

    return {"kills": cleaned_kills, "rounds": cleaned_rounds}


def load_silver_lake() -> dict[str, pd.DataFrame]:
    """Read and concatenate all Silver data across matches."""
    def _read_all(table: str) -> pd.DataFrame:
        files = sorted((SILVER_DIR / table).glob("match_*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    return {"kills": _read_all("kills"), "rounds": _read_all("rounds")}