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

# --- Bronze -----------------------------------------------------------------
# Raw tables persisted per match, exactly as awpy 3 returns them.
BRONZE_TABLES: tuple[str, ...] = (
    "header",        # 1 row: map, server, tick rate, playback length
    "players",       # roster: steamid, name, side, clan name
    "rounds",
    "kills",
    "damages",
    "bomb",          # pickup / drop / plant / defuse events
    "smokes",
    "grenades",      # grenade trajectories, one row per tick
    "shots",
    "blinds",
    "item_events",   # buys, pickups, drops
    "round_economy",
    "snapshots",     # player state sampled at SNAPSHOT_HZ (positions, hp, weapon...)
)

# Player-state sampling rate for the snapshots table, in samples per second.
# 4 Hz keeps a full match around 2 MB of Parquet and is enough for a 2D replay.
SNAPSHOT_HZ: float = float(os.environ.get("CS2_SNAPSHOT_HZ", "4"))

# --- Silver data contracts ----------------------------------------------------
# Column -> pandas dtype, per table. Silver keeps ONLY these columns, in this order.
# Conventions: sides are "CT" / "T"; teams are the clan names from the demo;
# steamids are strings (17-digit IDs do not survive float64); t_s is seconds since
# the end of freeze time, so t_s = 0 is the moment the round goes live.
_ID = {"match_id": "string", "round_num": "int64"}

SILVER_SCHEMAS: dict[str, dict[str, str]] = {
    "matches": {
        "match_id": "string", "map_name": "string", "server_name": "string",
        "tick_rate": "float64", "n_rounds": "int64", "team_a": "string", "team_b": "string",
        "score_a": "int64", "score_b": "int64", "demo_sha256": "string",
    },
    "players": {
        "match_id": "string", "steamid": "string", "name": "string", "team": "string",
    },
    "rounds": {
        **_ID, "start_tick": "int64", "freeze_end": "int64", "end_tick": "int64",
        "duration_s": "float64", "winner": "string", "winner_team": "string",
        "ct_team": "string", "t_team": "string", "reason": "string", "bomb_site": "string",
    },
    "kills": {
        **_ID, "tick": "int64", "t_s": "float64", "kill_order": "int64", "is_first_kill": "boolean",
        "attacker_steamid": "string", "attacker_name": "string", "attacker_team": "string",
        "attacker_side": "string", "victim_steamid": "string", "victim_name": "string",
        "victim_team": "string", "victim_side": "string", "assister_name": "string",
        "weapon": "string", "headshot": "boolean", "is_trade": "boolean", "victim_traded": "boolean",
        "thrusmoke": "boolean", "noscope": "boolean", "penetrated": "boolean", "assist_flash": "boolean",
        "hitgroup_name": "string",
        "attacker_x": "float32", "attacker_y": "float32", "attacker_z": "float32",
        "victim_x": "float32", "victim_y": "float32", "victim_z": "float32",
    },
    "damages": {
        **_ID, "tick": "int64", "t_s": "float64",
        "attacker_steamid": "string", "attacker_name": "string", "attacker_team": "string",
        "attacker_side": "string", "victim_steamid": "string", "victim_name": "string",
        "victim_team": "string", "victim_side": "string", "weapon": "string",
        "dmg_health": "int64", "dmg_health_real": "int64", "dmg_armor": "int64",
        "hitgroup_name": "string", "is_team_damage": "boolean",
    },
    "bomb": {
        **_ID, "tick": "int64", "t_s": "float64", "event": "string", "bombsite": "string",
        "steamid": "string", "name": "string",
        "x": "float32", "y": "float32", "z": "float32",
    },
    "smokes": {
        **_ID, "start_tick": "int64", "end_tick": "int64", "t_start_s": "float64", "t_end_s": "float64",
        "thrower_steamid": "string", "thrower_name": "string", "thrower_team": "string",
        "thrower_side": "string", "x": "float32", "y": "float32", "z": "float32",
    },
    "economy": {
        **_ID, "side": "string", "team": "string", "equipment_value": "int64",
        "buy_type": "string", "n_players": "int64",
    },
    "positions": {
        **_ID, "tick": "int64", "t_s": "float64", "steamid": "string", "name": "string",
        "team": "string", "side": "string", "x": "float32", "y": "float32", "z": "float32",
        "yaw": "float32", "pitch": "float32", "health": "int64", "armor": "int64",
        "is_alive": "boolean", "active_weapon": "string", "has_bomb": "boolean",
        "is_blinded": "boolean", "is_defusing": "boolean", "money": "int64", "equipment_value": "int64",
    },
}

SILVER_TABLES: tuple[str, ...] = tuple(SILVER_SCHEMAS)

# Bump whenever Silver logic or contracts change in a way that alters its output:
# every cached Silver match built with an older version is rebuilt on the next run.
SILVER_VERSION: int = 2

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