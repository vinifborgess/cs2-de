from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import BRONZE_DIR, get_logger
from .utils import to_pandas

log = get_logger()
BRONZE_TABLES = ("kills", "rounds")


def _bronze_path(match_id: str, table: str) -> Path:
    path = BRONZE_DIR / match_id / f"{table}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_demo_to_bronze(demo_path: str | Path, match_id: str, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Parse a CS2 demo file and persist raw tables to the Bronze layer.

    Returns {table_name: DataFrame}. Skips parsing if Parquet outputs
    already exist, unless force=True. Raises FileNotFoundError if the
    demo file is missing.
    """
    parquet_paths = [_bronze_path(match_id, t) for t in BRONZE_TABLES]
    if not force and all(p.exists() for p in parquet_paths):
        log.info("Bronze layer already exists for '%s' — skipping parsing.", match_id)
        return {t: pd.read_parquet(p) for t, p in zip(BRONZE_TABLES, parquet_paths)}

    from awpy import Demo

    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    log.info("Parsing '%s' (%s)...", match_id, demo_path.name)
    demo = Demo(str(demo_path))
    demo.parse()

    result: dict[str, pd.DataFrame] = {}
    for table in BRONZE_TABLES:
        df = to_pandas(getattr(demo, table, None))
        if df.empty:
            log.warning("Table '%s' is empty for '%s'.", table, match_id)
        df.to_parquet(_bronze_path(match_id, table), engine="pyarrow", index=False)
        result[table] = df
        log.info("Bronze[%s]: %d rows.", table, len(df))

    return result