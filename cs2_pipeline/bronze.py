"""Bronze layer: .dem -> raw Parquet tables, one directory per match.

Bronze keeps everything awpy extracts, untouched (engine noise included), so
later layers can be rebuilt without parsing the demo again. Every match gets a
``_manifest.json`` with the demo hash and parser settings; it is what makes
re-runs idempotent and every table traceable to its source file.

Layout::

    data/bronze/<match_id>/
        _manifest.json
        header.parquet, players.parquet, rounds.parquet, kills.parquet, ...
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import polars as pl

from .config import BRONZE_DIR, BRONZE_TABLES, SNAPSHOT_HZ, get_logger

log = get_logger()

MANIFEST_NAME = "_manifest.json"
MANIFEST_VERSION = 1

# Datasets decoded by awpy's lazy properties (everything except header/snapshots).
_AWPY_DATASETS = tuple(t for t in BRONZE_TABLES if t not in ("header", "snapshots"))


def _check_match_id(match_id: str) -> str:
    if not match_id or any(sep in match_id for sep in ("/", "\\")) or match_id.startswith("."):
        raise ValueError(f"Invalid match_id: {match_id!r}")
    return match_id


def bronze_dir(match_id: str) -> Path:
    return BRONZE_DIR / _check_match_id(match_id)


def bronze_path(match_id: str, table: str) -> Path:
    return bronze_dir(match_id) / f"{table}.parquet"


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest(match_id: str) -> dict | None:
    path = bronze_dir(match_id) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _is_fresh(match_id: str, sha256: str, snapshot_hz: float) -> bool:
    """True when Bronze already holds this exact demo, parsed with these settings."""
    m = read_manifest(match_id)
    if not m or m.get("manifest_version") != MANIFEST_VERSION:
        return False
    if m.get("demo_sha256") != sha256 or m.get("snapshot_hz") != snapshot_hz:
        return False
    if m.get("awpy_version", "").split(".")[0] != _awpy_version().split(".")[0]:
        return False
    return all(bronze_path(match_id, t).exists() for t in BRONZE_TABLES)


def _awpy_version() -> str:
    try:
        return metadata.version("awpy")
    except metadata.PackageNotFoundError:
        return "unknown"


def write_bronze_tables(match_id: str, tables: dict[str, pl.DataFrame], meta: dict) -> dict[str, Path]:
    """Persist tables plus manifest atomically (temp dir, then rename).

    A crash mid-write never leaves a half-populated match directory behind.
    """
    final = bronze_dir(match_id)
    tmp = BRONZE_DIR / f".{match_id}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        rows: dict[str, int] = {}
        for name, df in tables.items():
            df.write_parquet(tmp / f"{name}.parquet", compression="zstd")
            rows[name] = df.height
        manifest = {"manifest_version": MANIFEST_VERSION, "match_id": match_id, **meta, "rows": rows}
        (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {name: bronze_path(match_id, name) for name in tables}


def read_bronze(match_id: str, table: str) -> pd.DataFrame:
    """Load one Bronze table as a Pandas DataFrame (``*steamid`` columns as strings)."""
    path = bronze_path(match_id, table)
    if not path.exists():
        raise FileNotFoundError(f"Bronze table '{table}' not found for '{match_id}': {path}")
    df = pl.read_parquet(path)
    # Steam IDs are 17-digit uint64s. Pandas would turn a nullable one into float64
    # and silently lose precision (> 2**53), so IDs are handed over as strings.
    ids = [c for c in df.columns if c.endswith("steamid")]
    if ids:
        df = df.with_columns(pl.col(c).cast(pl.String) for c in ids)
    return df.to_pandas()


def parse_demo_to_bronze(
    demo_path: str | Path,
    match_id: str,
    *,
    force: bool = False,
    snapshot_hz: float = SNAPSHOT_HZ,
) -> dict[str, Path]:
    """Parse a CS2 demo and persist every raw table to the Bronze layer.

    Returns ``{table_name: parquet_path}``. Parsing is skipped when Bronze already
    holds this exact demo (same SHA-256) parsed with the same settings, unless
    ``force=True``. Raises FileNotFoundError if the demo does not exist.
    """
    _check_match_id(match_id)
    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")
    if snapshot_hz <= 0:
        raise ValueError("snapshot_hz must be positive")

    sha = _sha256(demo_path)
    if not force and _is_fresh(match_id, sha, snapshot_hz):
        log.info("Bronze already up to date for '%s' - skipping parsing.", match_id)
        return {t: bronze_path(match_id, t) for t in BRONZE_TABLES}

    from awpy import Demo

    t0 = time.time()
    log.info("Parsing '%s' (%s)...", match_id, demo_path.name)
    demo = Demo(str(demo_path))
    demo.load(*_AWPY_DATASETS)  # one shared decoding pass for compatible datasets

    header = dict(demo.header)
    header["tick_rate"] = float(demo.tick_rate)

    tables: dict[str, pl.DataFrame] = {"header": pl.DataFrame([header])}
    for name in _AWPY_DATASETS:
        df = getattr(demo, name)
        if df is None:
            log.warning("Table '%s' is unavailable for '%s'.", name, match_id)
            df = pl.DataFrame()
        tables[name] = df
    tables["snapshots"] = demo.snapshots(seconds=1.0 / snapshot_hz)

    for name, df in tables.items():
        if df.is_empty():
            log.warning("Bronze[%s] is empty for '%s'.", name, match_id)

    meta = {
        "demo_file": demo_path.name,
        "demo_bytes": demo_path.stat().st_size,
        "demo_sha256": sha,
        "awpy_version": _awpy_version(),
        "snapshot_hz": snapshot_hz,
        "tick_rate": header["tick_rate"],
        "map_name": header.get("map_name"),
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parse_seconds": round(time.time() - t0, 1),
    }
    paths = write_bronze_tables(match_id, tables, meta)
    log.info("Bronze[%s]: %s in %.1fs.", match_id,
             ", ".join(f"{n}={tables[n].height}" for n in ("rounds", "kills", "damages", "snapshots")),
             time.time() - t0)
    return paths
