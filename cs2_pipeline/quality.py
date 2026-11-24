"""Data-quality checks for the Silver layer.

Each check encodes an invariant that every valid CS2 match must satisfy, so it can be
run on any demo without an expected answer to compare against. Two levels:

* ``error``   - the data is wrong or unusable; downstream numbers cannot be trusted.
* ``warning`` - unusual but possible (a substitute player, a disconnect, a pause).

``check_match`` works on in-memory Silver tables; ``run_checks`` reads one match from
the lake. Nothing here modifies data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SILVER_DIR, SILVER_TABLES


@dataclass(frozen=True)
class Check:
    name: str
    level: str  # "error" | "warning"
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.level == "error" else "warn")
        return f"[{mark}] {self.name}" + (f": {self.detail}" if self.detail and not self.ok else "")


def _c(name: str, level: str, ok: bool, detail: str = "") -> Check:
    return Check(name, level, bool(ok), detail)


def _durations(rounds: pd.DataFrame) -> pd.Series:
    """round_num -> duration, tolerant of duplicated round numbers (reported elsewhere)."""
    return rounds.drop_duplicates("round_num").set_index("round_num")["duration_s"]


def _rounds_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    r, m, p = t["rounds"], t["matches"], t["players"]
    out: list[Check] = []
    if r.empty:
        return [_c("rounds.not_empty", "error", False, "no rounds survived cleaning")]
    dup = r["round_num"].duplicated().sum()
    out.append(_c("rounds.unique_round_num", "error", dup == 0, f"{dup} duplicated round numbers"))
    nums = sorted(r["round_num"])
    out.append(_c("rounds.starts_at_round_1", "error", nums[0] == 1,
                  f"first round is {nums[0]}: the demo starts mid-match, so scores and sides would be wrong"))
    gaps = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    out.append(_c("rounds.contiguous", "warning", not gaps, f"missing rounds {gaps[:10]}"))
    bad_winner = (~r["winner"].isin(["CT", "T"]) | r["winner_team"].isna()).sum()
    out.append(_c("rounds.winner_known", "error", bad_winner == 0, f"{bad_winner} rounds without a winner team"))
    same = (r["ct_team"].isna() | r["t_team"].isna() | (r["ct_team"] == r["t_team"])).sum()
    out.append(_c("rounds.teams_distinct", "error", same == 0, f"{same} rounds with missing or equal CT/T team"))
    odd = (~r["duration_s"].between(0, 200, inclusive="neither")).sum()
    out.append(_c("rounds.duration_plausible", "warning", odd == 0, f"{odd} rounds outside (0, 200] s"))
    # A round won by bombing or defusing needs a plant event; if the Bomb table lost
    # its events the site would silently read "none" for every such round.
    needs_plant = r["reason"].isin(["target_bombed", "bomb_defused"])
    unplanted = (needs_plant & (r["bomb_site"] == "none")).sum()
    out.append(_c("rounds.bomb_reason_has_plant", "error", unplanted == 0,
                  f"{unplanted} rounds ended by bomb explosion/defuse without a recorded plant"))
    row = m.iloc[0]
    out.append(_c("matches.has_winner", "warning", max(row["score_a"], row["score_b"]) >= 13,
                  f"score {row['score_a']}-{row['score_b']}: nobody reached 13, the demo may be cut short"))
    out.append(_c("matches.score_consistent", "error",
                  row["score_a"] + row["score_b"] == row["n_rounds"] == len(r),
                  f"score {row['score_a']}+{row['score_b']} vs {row['n_rounds']} / {len(r)} rounds"))
    out.append(_c("matches.two_teams", "error", pd.notna(row["team_a"]) and pd.notna(row["team_b"]) and row["team_a"] != row["team_b"],
                  f"teams: {row['team_a']!r}, {row['team_b']!r}"))
    out.append(_c("players.roster_size", "warning", len(p) == 10, f"{len(p)} players in the roster"))
    return out


def _kills_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    k, r = t["kills"], t["rounds"]
    if k.empty:
        return [_c("kills.not_empty", "error", False, "no kills")]
    out: list[Check] = []
    firsts = k.groupby("round_num")["is_first_kill"].sum()
    out.append(_c("kills.one_first_kill_per_round", "error", (firsts == 1).all(), f"rounds with != 1 first kill: {list(firsts[firsts != 1].index)[:10]}"))
    contiguous = k.groupby("round_num")["kill_order"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    out.append(_c("kills.kill_order_contiguous", "error", contiguous.all(), f"rounds: {list(contiguous[~contiguous].index)[:10]}"))
    twice = k.duplicated(["round_num", "victim_steamid"]).sum()
    out.append(_c("kills.victim_dies_once_per_round", "error", twice == 0, f"{twice} players killed twice in one round"))
    per_round = k.groupby("round_num").size()
    out.append(_c("kills.at_most_10_per_round", "error", (per_round <= 10).all(), f"max {per_round.max()}"))
    dur = k["round_num"].map(_durations(r))
    off = (~k["t_s"].between(0, dur + 1)).sum()
    out.append(_c("kills.inside_round", "error", off == 0, f"{off} kills outside [0, duration]"))
    mixed = (k["attacker_team"].isna() | k["victim_team"].isna() | (k["attacker_team"] == k["victim_team"])).sum()
    out.append(_c("kills.opposing_teams", "error", mixed == 0, f"{mixed} kills without two opposing teams"))
    # Cross-table: a killed player must show 0 health in the next position sample.
    pos = t["positions"]
    if not pos.empty:
        left = k[["round_num", "tick", "victim_steamid"]].rename(columns={"victim_steamid": "steamid"}).sort_values("tick")
        right = pos[["round_num", "tick", "steamid", "health"]].rename(columns={"tick": "ptick"}).sort_values("ptick")
        merged = pd.merge_asof(left, right, left_on="tick", right_on="ptick", by=["steamid", "round_num"], direction="forward")
        seen = merged.dropna(subset=["health"])
        share = (seen["health"] <= 0).mean() if len(seen) else 1.0
        out.append(_c("kills.victim_dead_in_positions", "warning", share >= 0.97, f"only {share:.0%} of victims are dead in the next snapshot"))
    return out


def _positions_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    p, r = t["positions"], t["rounds"]
    if p.empty:
        return [_c("positions.not_empty", "error", False, "no player positions")]
    out: list[Check] = []
    dup = p.duplicated(["round_num", "tick", "steamid"]).sum()
    out.append(_c("positions.no_duplicates", "error", dup == 0, f"{dup} duplicated (round, tick, player)"))
    sizes = p.groupby(["round_num", "tick"]).size()
    full = (sizes == 10).mean()
    out.append(_c("positions.full_lineup", "warning", full >= 0.99, f"only {full:.1%} of snapshots have 10 players"))
    sides = p.groupby(["round_num", "tick", "side"]).size().unstack(fill_value=0)
    balanced = ((sides.get("CT", 0) == 5) & (sides.get("T", 0) == 5)).mean()
    out.append(_c("positions.sides_balanced", "warning", balanced >= 0.99, f"only {balanced:.1%} of snapshots are 5 CT vs 5 T"))
    out.append(_c("positions.health_range", "error", p["health"].between(0, 100).all(), "health outside 0-100"))
    out.append(_c("positions.finite_coords", "error", np.isfinite(p[["x", "y", "z"]].to_numpy(dtype="float64")).all(), "NaN or infinite coordinates"))
    dur = p["round_num"].map(_durations(r))
    off = (~p["t_s"].between(0, dur + 1)).sum()
    out.append(_c("positions.inside_round", "error", off == 0, f"{off} samples outside [0, duration]"))
    return out


def _events_checks(t: dict[str, pd.DataFrame]) -> list[Check]:
    d, b, e, r = t["damages"], t["bomb"], t["economy"], t["rounds"]
    out: list[Check] = []
    if not d.empty:
        ok = d["dmg_health_real"].between(0, 100).all() and (d["dmg_armor"] >= 0).all()
        out.append(_c("damages.values_sane", "error", ok, "damage outside expected ranges"))
    plants = b[b["event"] == "finish_plant"]
    multi = plants.groupby("round_num").size().gt(1).sum()
    out.append(_c("bomb.one_plant_per_round", "error", multi == 0, f"{multi} rounds with more than one plant"))
    defuses = b[b["event"] == "defuse"]
    first_plant = plants.groupby("round_num")["t_s"].min()
    orphan = sum(1 for _, x in defuses.iterrows() if x["round_num"] not in first_plant.index or x["t_s"] < first_plant[x["round_num"]])
    out.append(_c("bomb.defuse_after_plant", "error", orphan == 0, f"{orphan} defuses without a prior plant"))
    out.append(_c("economy.available", "warning", not e.empty, "no economy rows: buy-type insights will be missing"))
    if not e.empty:  # absent is a warning above; present but inconsistent is an error
        per_round = e.groupby("round_num").size()
        out.append(_c("economy.two_sides_per_round", "error",
                      set(per_round.index) == set(r["round_num"]) and (per_round == 2).all(),
                      "economy must have exactly one CT and one T row per round"))
    return out


def check_match(tables: dict[str, pd.DataFrame]) -> list[Check]:
    """Run every invariant on the Silver tables of ONE match."""
    missing = [n for n in SILVER_TABLES if n not in tables]
    if missing:
        return [_c("silver.tables_present", "error", False, f"missing tables: {missing}")]
    out: list[Check] = []
    for section in (_rounds_checks, _kills_checks, _positions_checks, _events_checks):
        try:
            out += section(tables)
        except Exception as exc:  # noqa: BLE001 - a checker must report bad data, never crash on it
            out.append(_c(f"{section.__name__.strip('_').replace('_checks', '')}.checks_crashed", "error", False,
                          f"{type(exc).__name__}: {exc}"))
    return out


def run_checks(match_id: str) -> list[Check]:
    """Read one match from the Silver lake and check it."""
    tables = {}
    for name in SILVER_TABLES:
        path = SILVER_DIR / name / f"match_{match_id}.parquet"
        if not path.exists():
            return [_c("silver.tables_present", "error", False, f"{path} not found")]
        tables[name] = pd.read_parquet(path)
    return check_match(tables)


def list_matches() -> list[str]:
    return sorted(p.stem.removeprefix("match_") for p in (SILVER_DIR / "matches").glob("match_*.parquet"))


def summarize(checks: list[Check]) -> tuple[int, int]:
    """(failed errors, failed warnings)."""
    return (sum(1 for c in checks if not c.ok and c.level == "error"),
            sum(1 for c in checks if not c.ok and c.level == "warning"))
