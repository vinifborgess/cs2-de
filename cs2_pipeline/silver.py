"""Silver layer: typed, clean, analysis-ready tables with enforced data contracts.

Reads the Bronze tables of one match and applies, in order:

* side normalisation (``terrorist`` -> ``T``, ``counter-terrorist`` -> ``CT``);
* roster cleanup (the CSTV/GOTV pseudo-player is dropped) and team lookup;
* round assignment from tick ranges (awpy 3 events carry no ``round_num``),
  which also removes warm-up, knife rounds and the time between rounds;
* a per-round clock, ``t_s`` = seconds since freeze time ended;
* engine-noise removal (world kills, team kills, suicides);
* schema enforcement from ``config.SILVER_SCHEMAS``.

Layout: ``data/silver/<table>/match_<match_id>.parquet``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .bronze import read_bronze, read_manifest
from .config import SILVER_DIR, SILVER_SCHEMAS, SILVER_TABLES, SILVER_VERSION, get_logger
from .utils import coerce_schema, missing_columns

log = get_logger()

_SIDE_MAP = {"terrorist": "T", "t": "T", "counter-terrorist": "CT", "ct": "CT"}
_PSEUDO_PLAYER = "0"  # SourceTV / CSTV


def _silver_path(table: str, match_id: str) -> Path:
    directory = SILVER_DIR / table
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"match_{match_id}.parquet"


def _manifest_path(match_id: str) -> Path:
    directory = SILVER_DIR / "_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{match_id}.json"


def _bronze_fingerprint(match_id: str) -> dict:
    """What this match's Bronze currently is: demo hash, sampling rate and write stamp."""
    m = read_manifest(match_id) or {}
    return {k: m.get(k) for k in ("demo_sha256", "snapshot_hz", "written_at_ns")}


def _is_fresh(match_id: str, targets: dict[str, Path]) -> bool:
    """True when Silver was built from the current Bronze with the current contract."""
    path = _manifest_path(match_id)
    if not path.exists() or not all(p.exists() for p in targets.values()):
        return False
    try:
        built = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (built.get("silver_version") == SILVER_VERSION
            and built.get("bronze") == _bronze_fingerprint(match_id))


def _quarantine_path(match_id: str) -> Path:
    directory = SILVER_DIR / "_quarantine"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{match_id}.json"


def quarantine(match_id: str, reasons: list[str]) -> None:
    """Keep a match out of the lake (and so out of Gold) and record why."""
    _quarantine_path(match_id).write_text(json.dumps({
        "match_id": match_id, "reasons": reasons,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")


def release(match_id: str) -> bool:
    """Lift a quarantine. Returns True if the match was quarantined."""
    path = _quarantine_path(match_id)
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def quarantined() -> dict[str, list[str]]:
    """{match_id: reasons} for every quarantined match."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")).get("reasons", [])
            for p in sorted((SILVER_DIR / "_quarantine").glob("*.json"))}


# --- helpers ------------------------------------------------------------------

def normalize_side(s: pd.Series) -> pd.Series:
    """Map awpy side labels to the canonical "CT" / "T" (anything else -> NA)."""
    return s.astype("string").str.strip().str.lower().map(_SIDE_MAP).astype("string")


def assign_round(df: pd.DataFrame, rounds: pd.DataFrame, tick_col: str = "tick") -> pd.DataFrame:
    """Attach ``round_num`` to events by tick range; drop events outside any round.

    ``rounds`` must hold only the rounds that are kept (no knife rounds). Events that
    happen in warm-up or between rounds fall outside every [start_tick, end_tick] and
    are dropped. That includes post-round events: once the round is decided (``end_tick``),
    a kill before ``official_end_tick`` no longer counts toward its outcome.
    """
    if df.empty:
        return df.assign(round_num=pd.Series(dtype="int64"))
    left = df.copy()
    left["_tick"] = left[tick_col].astype("int64")
    left = left.sort_values("_tick", kind="stable")
    right = (rounds[["round_num", "start_tick", "end_tick"]]
             .rename(columns={"start_tick": "_start", "end_tick": "_end"})
             .astype("int64").sort_values("_start"))
    merged = pd.merge_asof(left, right, left_on="_tick", right_on="_start", direction="backward")
    merged = merged[merged["_tick"] <= merged["_end"]]
    return merged.drop(columns=["_tick", "_start", "_end"]).reset_index(drop=True)


def add_clock(df: pd.DataFrame, rounds: pd.DataFrame, tick_rate: float,
              tick_col: str = "tick", out: str = "t_s") -> pd.DataFrame:
    """Add ``out`` = seconds since freeze time ended, using the event's round."""
    freeze = rounds.set_index("round_num")["freeze_end"]
    df = df.copy()
    df[out] = ((df[tick_col].astype("int64") - df["round_num"].map(freeze)) / tick_rate).round(3)
    return df


def _team(series: pd.Series, team_of: dict[str, str]) -> pd.Series:
    return series.map(team_of).astype("string")


def _finish(df: pd.DataFrame, table: str, match_id: str) -> pd.DataFrame:
    df = df.copy()
    df["match_id"] = match_id
    miss = missing_columns(df, SILVER_SCHEMAS[table])
    if miss:
        log.warning("silver.%s[%s]: missing columns %s", table, match_id, miss)
    return coerce_schema(df, SILVER_SCHEMAS[table]).reset_index(drop=True)


# --- table builders -----------------------------------------------------------

def infer_start_sides(snap: pd.DataFrame, base: pd.DataFrame) -> dict[str, str]:
    """Side (CT/T) each player started the first valid round on, from the snapshots."""
    if snap.empty or base.empty:
        return {}
    first = base.iloc[0]
    df = snap[(snap["tick"] >= first["start_tick"]) & (snap["tick"] <= first["end_tick"])].copy()
    df["side"] = normalize_side(df["side"])
    df = df[df["side"].isin(["CT", "T"])].sort_values("tick")
    df["steamid"] = df["steamid"].astype("string")
    return df.drop_duplicates("steamid").set_index("steamid")["side"].to_dict()


def _fill_team_names(df: pd.DataFrame, start_side: dict[str, str]) -> pd.DataFrame:
    """Name players without a clan name from who they started the match with.

    Players that start on the same side are one team, so a missing name takes
    the team name its side-mates carry. With no names at all, the two halves
    become 'Team A' (started CT) and 'Team B' (started T). A side holding more
    than 5 players is not a team, so nothing is inferred for it.
    """
    df = df.copy()
    df["_start"] = df["steamid"].astype("string").map(start_side)
    fallback = {"CT": "Team A", "T": "Team B"}
    for side, grp in df.dropna(subset=["_start"]).groupby("_start"):
        if len(grp) > 5:
            continue
        known = grp["team_clan_name"].dropna()
        name = known.mode().iat[0] if len(known) else fallback[side]
        idx = grp.index[grp["team_clan_name"].isna()]
        df.loc[idx, "team_clan_name"] = name
    return df.drop(columns="_start")


def clean_players(raw: pd.DataFrame, match_id: str,
                  start_side: dict[str, str] | None = None) -> pd.DataFrame:
    df = raw[raw["steamid"].astype("string") != _PSEUDO_PLAYER].copy()
    # A blank clan name is a missing one: otherwise "" would be treated as a team.
    df["team_clan_name"] = df["team_clan_name"].astype("string").str.strip().replace("", pd.NA)
    if start_side:
        df = _fill_team_names(df, start_side)
    df = df.dropna(subset=["team_clan_name"])
    df = df.rename(columns={"team_clan_name": "team"})
    return _finish(df.drop_duplicates("steamid"), "players", match_id)


def clean_rounds_base(raw: pd.DataFrame) -> pd.DataFrame:
    """Round boundaries and winner, before team/bomb enrichment."""
    # awpy ships both a numeric `reason` code and `reason_name`; the name is what we keep.
    df = raw.drop(columns=["reason"], errors="ignore").rename(
        columns={"freeze_end_tick": "freeze_end", "reason_name": "reason"})
    df["winner"] = normalize_side(df["winner_side"])
    keep = (~df["is_knife_round"].fillna(False)) & (df["start_tick"] > 0) & df["winner"].isin(["CT", "T"])
    df = df[keep].sort_values("start_tick")
    # A restart (mp_restartgame, a replayed round) leaves two rows for the same round
    # number. The last one is the round that was actually played out.
    dup = df["round_num"].duplicated(keep="last")
    if dup.any():
        log.warning("silver.rounds: dropped %d superseded round(s) %s (restarted rounds)",
                    int(dup.sum()), sorted(df.loc[dup, "round_num"].tolist()))
    return df[~dup].sort_values("round_num").reset_index(drop=True)


def clean_positions(snap: pd.DataFrame, rounds: pd.DataFrame, team_of: dict[str, str],
                    tick_rate: float, match_id: str) -> pd.DataFrame:
    df = snap[snap["steamid"].astype("string").isin(team_of)].copy()
    df["side"] = normalize_side(df["side"])
    df = assign_round(df, rounds)
    df = add_clock(df, rounds, tick_rate)
    df = df[df["t_s"] >= 0]  # live phase only: freeze time is not part of the replay
    df["team"] = _team(df["steamid"], team_of)
    df["is_alive"] = df["health"] > 0
    return _finish(df, "positions", match_id)


def _round_teams(positions: pd.DataFrame) -> pd.DataFrame:
    """Which team played CT / T in each round (sides swap at halftime)."""
    if positions.empty:
        return pd.DataFrame(columns=["round_num", "ct_team", "t_team"])
    mode = (positions.dropna(subset=["side", "team"])
            .groupby(["round_num", "side"])["team"]
            .agg(lambda s: s.mode().iat[0]).unstack("side"))
    mode = mode.rename(columns={"CT": "ct_team", "T": "t_team"}).reset_index()
    for c in ("ct_team", "t_team"):
        if c not in mode:
            mode[c] = pd.NA
    return mode[["round_num", "ct_team", "t_team"]]


def clean_rounds(base: pd.DataFrame, positions: pd.DataFrame, bomb: pd.DataFrame,
                 tick_rate: float, match_id: str) -> pd.DataFrame:
    df = base.merge(_round_teams(positions), on="round_num", how="left")
    df["winner_team"] = np.where(df["winner"] == "CT", df["ct_team"], df["t_team"])
    df["duration_s"] = ((df["end_tick"] - df["freeze_end"]) / tick_rate).round(3)
    sites = (bomb[bomb["event"] == "finish_plant"].drop_duplicates("round_num")
             .set_index("round_num")["bombsite"])
    df["bomb_site"] = df["round_num"].map(sites).fillna("none")
    return _finish(df, "rounds", match_id)


def clean_kills(raw: pd.DataFrame, rounds: pd.DataFrame, team_of: dict[str, str],
                tick_rate: float, match_id: str) -> pd.DataFrame:
    df = assign_round(raw, rounds)
    df = add_clock(df, rounds, tick_rate)
    df = df[df["t_s"] >= 0]
    for side in ("attacker_side", "victim_side"):
        df[side] = normalize_side(df[side])
    df["attacker_team"] = _team(df["attacker_steamid"], team_of)
    df["victim_team"] = _team(df["victim_steamid"], team_of)
    # Engine noise: world deaths (fall, bomb...), suicides and team kills.
    df = df.dropna(subset=["attacker_steamid", "attacker_team", "victim_team"])
    df = df[(df["weapon"].astype("string").str.lower() != "world")
            & (df["attacker_team"] != df["victim_team"])]
    df = df.sort_values(["round_num", "tick"], kind="stable")
    df["kill_order"] = df.groupby("round_num").cumcount() + 1
    df["is_first_kill"] = df["kill_order"] == 1
    return _finish(df, "kills", match_id)


def clean_damages(raw: pd.DataFrame, rounds: pd.DataFrame, team_of: dict[str, str],
                  tick_rate: float, match_id: str) -> pd.DataFrame:
    df = assign_round(raw, rounds)
    df = add_clock(df, rounds, tick_rate)
    df = df[df["t_s"] >= 0]
    for side in ("attacker_side", "victim_side"):
        df[side] = normalize_side(df[side])
    df["attacker_team"] = _team(df["attacker_steamid"], team_of)
    df["victim_team"] = _team(df["victim_steamid"], team_of)
    df = df.dropna(subset=["attacker_steamid", "attacker_team", "victim_team"])
    df["is_team_damage"] = df["attacker_team"] == df["victim_team"]
    # dmg_health is raw weapon damage and can exceed what the victim had left.
    df["dmg_health_real"] = (df["health_pre"] - df["health_post"]).clip(lower=0)
    return _finish(df, "damages", match_id)


def clean_bomb(raw: pd.DataFrame, rounds: pd.DataFrame, team_of: dict[str, str],
               tick_rate: float, match_id: str) -> pd.DataFrame:
    df = assign_round(raw, rounds)
    df = add_clock(df, rounds, tick_rate)
    return _finish(df, "bomb", match_id)


def clean_smokes(raw: pd.DataFrame, rounds: pd.DataFrame, team_of: dict[str, str],
                 tick_rate: float, match_id: str) -> pd.DataFrame:
    df = assign_round(raw, rounds, tick_col="start_tick")
    df = add_clock(df, rounds, tick_rate, tick_col="start_tick", out="t_start_s")
    df = add_clock(df, rounds, tick_rate, tick_col="end_tick", out="t_end_s")
    df["thrower_side"] = normalize_side(df["thrower_side"])
    df["thrower_team"] = _team(df["thrower_steamid"], team_of)
    return _finish(df, "smokes", match_id)


def clean_economy(raw: pd.DataFrame, rounds: pd.DataFrame, match_id: str) -> pd.DataFrame:
    df = raw[raw["round_num"].isin(rounds["round_num"])].copy()
    df["side"] = normalize_side(df["side"])
    teams = rounds.set_index("round_num")[["ct_team", "t_team"]]
    df["team"] = np.where(df["side"] == "CT", df["round_num"].map(teams["ct_team"]),
                          df["round_num"].map(teams["t_team"]))
    return _finish(df, "economy", match_id)


def build_matches(rounds: pd.DataFrame, header: pd.DataFrame, manifest: dict | None,
                  match_id: str) -> pd.DataFrame:
    teams = sorted(set(rounds["ct_team"].dropna()) | set(rounds["t_team"].dropna()))
    a, b = (teams + [None, None])[:2]
    wins = rounds["winner_team"].value_counts()
    h = header.iloc[0]
    row = {
        "map_name": h.get("map_name"), "server_name": h.get("server_name"),
        "tick_rate": float(h["tick_rate"]), "n_rounds": len(rounds), "team_a": a, "team_b": b,
        "score_a": int(wins.get(a, 0)), "score_b": int(wins.get(b, 0)),
        "demo_sha256": (manifest or {}).get("demo_sha256"),
    }
    return _finish(pd.DataFrame([row]), "matches", match_id)


# --- orchestration ------------------------------------------------------------

def bronze_to_silver(match_id: str, *, force: bool = False) -> dict[str, pd.DataFrame]:
    """Build the Silver tables for one match from its Bronze tables. Idempotent."""
    targets = {t: _silver_path(t, match_id) for t in SILVER_TABLES}
    if not force and _is_fresh(match_id, targets):
        log.info("Silver is up to date for '%s' - skipping cleaning.", match_id)
        return {t: pd.read_parquet(p) for t, p in targets.items()}

    header = read_bronze(match_id, "header")
    tick_rate = float(header["tick_rate"].iloc[0])

    base = clean_rounds_base(read_bronze(match_id, "rounds"))
    snapshots = read_bronze(match_id, "snapshots")
    players = clean_players(read_bronze(match_id, "players"), match_id,
                            start_side=infer_start_sides(snapshots, base))
    team_of = dict(zip(players["steamid"], players["team"]))
    positions = clean_positions(snapshots, base, team_of, tick_rate, match_id)
    bomb = clean_bomb(read_bronze(match_id, "bomb"), base, team_of, tick_rate, match_id)
    rounds = clean_rounds(base, positions, bomb, tick_rate, match_id)

    tables = {
        "matches": build_matches(rounds, header, read_manifest(match_id), match_id),
        "players": players,
        "rounds": rounds,
        "kills": clean_kills(read_bronze(match_id, "kills"), rounds, team_of, tick_rate, match_id),
        "damages": clean_damages(read_bronze(match_id, "damages"), rounds, team_of, tick_rate, match_id),
        "bomb": bomb,
        "smokes": clean_smokes(read_bronze(match_id, "smokes"), rounds, team_of, tick_rate, match_id),
        "economy": clean_economy(read_bronze(match_id, "round_economy"), rounds, match_id),
        "positions": positions,
    }
    for name, df in tables.items():
        df.to_parquet(targets[name], engine="pyarrow", index=False)
    _manifest_path(match_id).write_text(json.dumps({
        "silver_version": SILVER_VERSION,
        "bronze": _bronze_fingerprint(match_id),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {n: len(d) for n, d in tables.items()},
    }, indent=2), encoding="utf-8")
    log.info("Silver[%s]: %s", match_id, ", ".join(f"{n}={len(d)}" for n, d in tables.items()))
    return tables


def load_silver_lake(tables: tuple[str, ...] | None = None, *,
                     include_quarantined: bool = False) -> dict[str, pd.DataFrame]:
    """Read and concatenate Silver data across matches (all tables by default).

    Quarantined matches (failed data-quality errors) are left out unless asked for.
    """
    skip = set() if include_quarantined else set(quarantined())

    def _read_all(table: str) -> pd.DataFrame:
        files = [f for f in sorted((SILVER_DIR / table).glob("match_*.parquet"))
                 if f.stem.removeprefix("match_") not in skip]
        if not files:
            return pd.DataFrame()
        return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    return {t: _read_all(t) for t in (tables or SILVER_TABLES)}


def list_matches(include_quarantined: bool = False) -> list[str]:
    """Match ids that have Silver data, quarantined ones excluded unless asked for."""
    skip = set() if include_quarantined else set(quarantined())
    ids = (f.stem.removeprefix("match_") for f in (SILVER_DIR / "matches").glob("match_*.parquet"))
    return sorted(i for i in ids if i not in skip)


def read_match(match_id: str, tables: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Read the Silver tables of ONE match (all by default)."""
    return {t: pd.read_parquet(_silver_path(t, match_id)) for t in (tables or SILVER_TABLES)}
