"""Each data-quality check must pass on a clean match and fail when its invariant is broken."""
from tests.test_pipeline import make_match  # noqa: F401  (also points CS2_DATA_DIR at a scratch dir)

import copy

import numpy as np
import pandas as pd

from cs2_pipeline.quality import check_match, run_checks, summarize
from cs2_pipeline.silver import bronze_to_silver


def _clean():
    make_match("q-base", seed=11)
    return bronze_to_silver("q-base", force=True)


def _failed(tables):
    return {c.name for c in check_match(tables) if not c.ok}


def test_clean_match_passes_every_check():
    tables = _clean()
    checks = check_match(tables)
    assert len(checks) >= 40
    assert summarize(checks) == (0, 0), [str(c) for c in checks if not c.ok]
    assert summarize(run_checks("q-base")) == (0, 0)


def _mut(fn):
    def apply(t):
        t = {k: v.copy() for k, v in t.items()}
        fn(t)
        return t
    return apply


def _dup_round(t): t["rounds"].loc[1, "round_num"] = t["rounds"].loc[0, "round_num"]
def _drop_round(t): t["rounds"].drop(index=4, inplace=True)
def _no_winner(t): t["rounds"].loc[0, "winner_team"] = pd.NA
def _same_teams(t): t["rounds"].loc[0, "t_team"] = t["rounds"].loc[0, "ct_team"]
def _long_round(t): t["rounds"].loc[0, "duration_s"] = 999.0
def _bad_score(t): t["matches"].loc[0, "score_a"] += 1
def _one_team(t): t["matches"].loc[0, "team_b"] = t["matches"].loc[0, "team_a"]
def _small_roster(t): t["players"].drop(index=0, inplace=True)
def _no_first(t): t["kills"].loc[t["kills"]["is_first_kill"].idxmax(), "is_first_kill"] = False
def _bad_order(t): t["kills"].loc[0, "kill_order"] = 99
def _victim_twice(t):
    k = t["kills"]; rnd = k["round_num"].iloc[0]; idx = k.index[k["round_num"] == rnd]
    k.loc[idx[1], "victim_steamid"] = k.loc[idx[0], "victim_steamid"]
def _too_many_kills(t):
    k = t["kills"]; rnd = k["round_num"].iloc[0]
    t["kills"] = pd.concat([k] + [k[k["round_num"] == rnd]] * 4, ignore_index=True)
def _kill_outside(t): t["kills"].loc[0, "t_s"] = 9999.0
def _team_kill(t): t["kills"].loc[0, "attacker_team"] = t["kills"].loc[0, "victim_team"]
def _victim_alive(t): t["positions"]["health"] = 100
def _dup_position(t): t["positions"] = pd.concat([t["positions"], t["positions"].iloc[:1]], ignore_index=True)
def _lose_player(t):
    p = t["positions"]; t["positions"] = p[p["steamid"] != p["steamid"].iloc[0]].reset_index(drop=True)
def _unbalanced_sides(t):
    p = t["positions"]; sid = p["steamid"].iloc[0]; cur = p.loc[p["steamid"] == sid, "side"].iloc[0]
    p.loc[p["steamid"] == sid, "side"] = "T" if cur == "CT" else "CT"
def _bad_health(t): t["positions"].loc[0, "health"] = 150
def _nan_coord(t): t["positions"].loc[0, "x"] = np.nan
def _pos_outside(t): t["positions"].loc[0, "t_s"] = 9999.0
def _bad_damage(t): t["damages"].loc[0, "dmg_health_real"] = 500
def _two_plants(t):
    b = t["bomb"]; row = b.iloc[[0]].copy(); row["event"] = "finish_plant"
    t["bomb"] = pd.concat([b[b["event"] != "finish_plant"], row, row], ignore_index=True)
def _orphan_defuse(t):
    b = t["bomb"]; row = b.iloc[[0]].copy(); row["event"] = "defuse"; row["round_num"] = 4; row["t_s"] = 50.0
    t["bomb"] = pd.concat([b, row], ignore_index=True)
def _lose_economy(t): t["economy"].drop(index=0, inplace=True)
def _blind_outside_rounds(t): t["blinds"].loc[0, "round_num"] = 999
def _grenade_outside_rounds(t): t["grenades"].loc[0, "round_num"] = 999
def _shot_outside_rounds(t): t["shots"].loc[0, "round_num"] = 999
def _item_outside_rounds(t): t["item_events"].loc[0, "round_num"] = 999
def _long_flash(t): t["blinds"].loc[0, "duration"] = 9.0
def _flash_both_kinds(t):
    b = t["blinds"]; b.loc[0, ["is_self_flash", "is_team_flash"]] = [True, True]
def _duplicate_throw(t):
    g = t["grenades"]; t["grenades"] = pd.concat([g, g.iloc[[0]]], ignore_index=True)
def _grenade_lands_first(t): t["grenades"].loc[0, "end_tick"] = t["grenades"].loc[0, "start_tick"] - 10
def _unknown_grenade(t): t["grenades"].loc[0, "grenade_type"] = "plasma"
def _prefixed_weapon(t): t["shots"].loc[0, "weapon"] = "weapon_ak47"
def _shooter_without_team(t): t["shots"].loc[0, "team"] = pd.NA
def _unknown_item_action(t): t["item_events"].loc[0, "action"] = "teleport"
def _freeze_flag_lies(t):
    ie = t["item_events"]; ie.loc[0, "is_freeze_time"] = not bool(ie.loc[0, "is_freeze_time"])
def _negative_cost(t): t["item_events"].loc[0, "cost"] = -5
def _ordered_post_round_kill(t):
    k = t["kills"]; idx = k.index[k["is_post_round"].fillna(False).astype(bool)][0]
    k.loc[idx, "kill_order"] = 99
def _mid_match(t):
    r = t["rounds"]; t["rounds"] = r[r["round_num"] > 3].reset_index(drop=True)
def _bomb_without_plant(t): t["rounds"].loc[0, "reason"] = "target_bombed"; t["rounds"].loc[0, "bomb_site"] = "none"
def _no_winner_score(t): t["matches"].loc[0, "score_a"] = 6; t["matches"].loc[0, "score_b"] = 6
def _empty_economy(t): t["economy"] = t["economy"].iloc[0:0]

MUTATIONS = {
    "rounds.unique_round_num": _dup_round, "rounds.contiguous": _drop_round,
    "rounds.winner_known": _no_winner, "rounds.teams_distinct": _same_teams,
    "rounds.duration_plausible": _long_round, "matches.score_consistent": _bad_score,
    "matches.two_teams": _one_team, "players.roster_size": _small_roster,
    "kills.one_first_kill_per_round": _no_first, "kills.kill_order_contiguous": _bad_order,
    "kills.victim_dies_once_per_round": _victim_twice, "kills.at_most_10_per_round": _too_many_kills, "kills.inside_round": _kill_outside,
    "kills.opposing_teams": _team_kill, "kills.victim_dead_in_positions": _victim_alive,
    "positions.no_duplicates": _dup_position, "positions.full_lineup": _lose_player,
    "positions.sides_balanced": _unbalanced_sides, "positions.health_range": _bad_health, "positions.finite_coords": _nan_coord,
    "positions.inside_round": _pos_outside, "damages.values_sane": _bad_damage,
    "bomb.one_plant_per_round": _two_plants, "bomb.defuse_after_plant": _orphan_defuse,
    "economy.two_sides_per_round": _lose_economy, "rounds.starts_at_round_1": _mid_match,
    "rounds.bomb_reason_has_plant": _bomb_without_plant, "matches.has_winner": _no_winner_score,
    "economy.available": _empty_economy, "kills.post_round_has_no_order": _ordered_post_round_kill,
    "blinds.inside_known_rounds": _blind_outside_rounds,
    "grenades.inside_known_rounds": _grenade_outside_rounds,
    "shots.inside_known_rounds": _shot_outside_rounds,
    "item_events.inside_known_rounds": _item_outside_rounds,
    "blinds.duration_plausible": _long_flash, "blinds.self_and_team_exclusive": _flash_both_kinds,
    "grenades.one_row_per_throw": _duplicate_throw, "grenades.flight_forwards": _grenade_lands_first,
    "grenades.known_types": _unknown_grenade, "shots.weapon_prefix_stripped": _prefixed_weapon,
    "shots.shooter_known": _shooter_without_team, "item_events.known_actions": _unknown_item_action,
    "item_events.freeze_flag_matches_clock": _freeze_flag_lies,
    "item_events.costs_not_negative": _negative_cost,
}


def test_every_check_fails_when_its_invariant_is_broken():
    base = _clean()
    all_names = {c.name for c in check_match(base)}
    uncovered = all_names - set(MUTATIONS)
    assert not uncovered, f"checks without a mutation test: {sorted(uncovered)}"
    misses = [name for name, fn in MUTATIONS.items() if name not in _failed(_mut(fn)(base))]
    assert not misses, f"corruption not detected by: {misses}"


def test_empty_economy_is_a_warning_not_an_error():
    t = _mut(_empty_economy)(_clean())
    errors, warnings = summarize(check_match(t))
    assert errors == 0 and warnings == 1


def test_corrupt_data_is_reported_not_raised():
    base = _clean()
    for name, fn in MUTATIONS.items():
        checks = check_match(_mut(fn)(base))  # must not raise
        crashed = [c for c in checks if c.name.endswith("checks_crashed")]
        assert not crashed, f"{name}: {[str(c) for c in crashed]}"


def test_missing_tables_are_an_error():
    tables = _clean(); tables.pop("kills")
    assert summarize(check_match(tables))[0] == 1


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
