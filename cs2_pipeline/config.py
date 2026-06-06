"""
Central configuration of the CS2 pipeline.

(raw -> bronze -> silver -> gold) is anchored here.

Changing a path or schema happens in ONE place only.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CS2_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

for _dir in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

KILLS_SCHEMA: dict[str, str] = {
    "match_id": "string",
    "round_num": "int64",
    "tick": "int64",
    "attacker_name": "string",
    "attacker_side": "string",
    "victim_name": "string",
    "victim_side": "string",
    "weapon": "string",
    "headshot": "boolean",
    "attacker_X": "float32",
    "attacker_Y": "float32",
    "victim_X": "float32",
    "victim_Y": "float32",
}

ROUNDS_SCHEMA: dict[str, str] = {
    "match_id": "string",
    "round_num": "int64",
    "start_tick": "int64",
    "freeze_end": "int64",
    "end_tick": "int64",
    "winner": "string",
    "reason": "string",
    "bomb_site": "string",
}

ROUNDS_RENAME: dict[str, str] = {"start": "start_tick", "end": "end_tick"}

def get_logger(name: str = "cs2") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger