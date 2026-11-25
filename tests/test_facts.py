"""fact_round: grain, joins and edge cases, checked against what the synthetic match is known to contain."""
from tests.test_pipeline import make_match

import pandas as pd

from cs2_pipeline.facts import FACT_PLAYER_COLUMNS, FACT_ROUND_COLUMNS, build_fact_player, build_fact_round, half_of
from cs2_pipeline.silver import bronze_to_silver


def _tables(mid="f-base", seed=21):
    exp = make_match(mid, seed=seed)
    return exp, bronze_to_silver(mid, force=True)


def test_fact_round_has_one_row_per_round_with_a_stable_schema():
    exp, t = _tables()
    f = build_fact_round(t)
    assert list(f.columns) == FACT_ROUND_COLUMNS
    assert len(f) == exp["n_rounds"] and f["round_num"].is_unique
    assert (f["map_name"] == "de_test").all() and f["match_id"].eq("f-base").all()


def test_opening_duel_matches_the_known_first_kills():
    exp, t = _tables()
    f = build_fact_round(t).set_index("round_num")
    assert f["fb_player"].to_dict() == exp["first"]
    assert (f["fb_team"] == f["fb_player"].str.split("_").str[0]).all()   # players are named <Team>_pN
    won = f["fb_team"] == f["winner_team"]
    assert (f["fb_won"] == won).all()
    assert (f["fb_t_s"] >= 0).all() and f["n_kills"].gt(0).all()


def test_sides_halves_and_bomb_follow_the_match_structure():
    _, t = _tables()
    f = build_fact_round(t)
    assert f.loc[f["round_num"] <= 12, "half"].eq(1).all() and f.loc[f["round_num"] > 12, "half"].eq(2).all()
    assert f.loc[f["round_num"] <= 12, "ct_team"].eq("Alpha").all()      # Alpha starts CT, swaps at half
    assert f.loc[f["round_num"] > 12, "ct_team"].eq("Bravo").all()
    assert f["planted"].sum() == 1 and f.loc[f["planted"], "round_num"].iloc[0] == 3
    assert f.loc[f["planted"], "plant_t_s"].notna().all() and f.loc[~f["planted"], "plant_t_s"].isna().all()
    assert f.loc[f["round_num"] == 3, "bomb_site"].iloc[0] == "B"
    assert f["ct_buy"].eq("full").all() and f["ct_equip"].eq(20000).all()


def test_missing_economy_and_bomb_leave_nulls_not_errors():
    _, t = _tables("f-sparse", seed=22)
    t["economy"] = t["economy"].iloc[0:0]
    t["bomb"] = t["bomb"].iloc[0:0]
    f = build_fact_round(t)
    assert len(f) == 24 and f["ct_buy"].isna().all() and f["ct_equip"].isna().all()
    assert not f["planted"].any() and f["plant_t_s"].isna().all()


def test_round_without_kills_has_no_opening_duel():
    _, t = _tables("f-nokill", seed=23)
    t["kills"] = t["kills"][t["kills"]["round_num"] != 5]
    f = build_fact_round(t).set_index("round_num")
    assert pd.isna(f.loc[5, "fb_team"]) and pd.isna(f.loc[5, "fb_won"]) and f.loc[5, "n_kills"] == 0
    assert f["fb_won"].dropna().size == 23


def test_half_boundaries():
    assert half_of(pd.Series([1, 12, 13, 24, 25, 30])).tolist() == [1, 1, 2, 2, 3, 3]


# --- fact_player: a tiny hand-made match whose every stat is worked out by hand ----------------
A = {f"a{i}": f"7656119800000000{i}" for i in range(1, 6)}
B = {f"b{i}": f"7656119800000001{i}" for i in range(1, 6)}
SID = {**A, **B}


def _hand_match():
    players = pd.DataFrame([{"match_id": "h", "steamid": SID[n], "name": n, "team": "A" if n[0] == "a" else "B"}
                            for n in SID])
    rounds = pd.DataFrame({"round_num": [1, 2, 3], "winner_team": ["A", "A", "B"]})

    def k(rnd, order, att, vic, **kw):
        row = {"round_num": rnd, "kill_order": order, "attacker_steamid": SID[att], "victim_steamid": SID[vic],
               "attacker_team": att[0].upper(), "victim_team": vic[0].upper(), "assister_name": None,
               "headshot": False, "is_trade": False, "victim_traded": False, "is_first_kill": order == 1,
               "is_post_round": False}
        row.update(kw)
        return row

    kills = pd.DataFrame([
        k(1, 1, "a1", "b1", headshot=True, assister_name="a2"),
        k(1, 2, "b2", "a1", victim_traded=True),
        k(1, 3, "a3", "b2", is_trade=True),
        k(1, 4, "a3", "b3"),
        k(1, 5, "a3", "b4"),
        k(2, 1, "b1", "a1"), k(2, 2, "b2", "a2"), k(2, 3, "b3", "a3"), k(2, 4, "b4", "a4"),
        k(2, 5, "a5", "b1"), k(2, 6, "a5", "b2"), k(2, 7, "a5", "b3"), k(2, 8, "a5", "b4"), k(2, 9, "a5", "b5"),
        # Round 3: a4 dies but is traded, and that trade is the only reason a4 is KAST that round.
        k(3, 1, "b5", "a4", victim_traded=True), k(3, 2, "a2", "b5", is_trade=True),
    ])
    dmg = pd.DataFrame([
        {"attacker_steamid": SID["a1"], "dmg_health_real": 100, "is_team_damage": False, "is_post_round": False},
        {"attacker_steamid": SID["a1"], "dmg_health_real": 40, "is_team_damage": True, "is_post_round": False},
        *[{"attacker_steamid": SID["a3"], "dmg_health_real": 100, "is_team_damage": False, "is_post_round": False}] * 3,
    ])
    pos = pd.DataFrame([{"steamid": SID[n], "round_num": r} for n in SID for r in (1, 2, 3)])
    return {"players": players, "kills": kills, "damages": dmg, "rounds": rounds, "positions": pos}


def _row(f, name):
    return f[f["name"] == name].iloc[0]


def test_post_round_events_are_counted_apart_from_every_other_metric():
    t = _hand_match()
    # a1 kills b5 and does 100 damage after round 1 was already decided.
    post_k = t["kills"].iloc[[0]].copy()
    post_k[["round_num", "kill_order", "is_post_round", "is_first_kill"]] = [1, pd.NA, True, False]
    post_k["attacker_steamid"], post_k["victim_steamid"] = SID["a1"], SID["b5"]
    post_k["attacker_team"], post_k["victim_team"] = "A", "B"
    t["kills"] = pd.concat([t["kills"], post_k], ignore_index=True)
    t["damages"] = pd.concat([t["damages"], pd.DataFrame([
        {"attacker_steamid": SID["a1"], "dmg_health_real": 100, "is_team_damage": False, "is_post_round": True}])],
        ignore_index=True)
    f = build_fact_player(t)
    base = build_fact_player(_hand_match())
    a1, b5 = _row(f, "a1"), _row(f, "b5")
    assert (a1["post_round_kills"], a1["post_round_damage"]) == (1, 100.0)
    assert (b5["post_round_deaths"], b5["post_round_kills"]) == (1, 0)
    # The published K-D is kills + post_round_kills; every live metric is untouched.
    for col in ("kills", "deaths", "damage", "adr", "kast_rounds", "kast_pct", "clutch_attempts", "k2", "k3"):
        assert f.set_index("name")[col].equals(base.set_index("name")[col]), col


def test_fact_player_counts_match_the_hand_worked_answer():
    f = build_fact_player(_hand_match())
    assert list(f.columns) == FACT_PLAYER_COLUMNS and len(f) == 10
    a1, a3, a5, b2 = (_row(f, n) for n in ("a1", "a3", "a5", "b2"))
    assert (a1["kills"], a1["deaths"], a1["headshots"], a1["opening_kills"], a1["opening_deaths"]) == (1, 2, 1, 1, 1)
    assert (a3["kills"], a3["trade_kills"], a3["k3"]) == (3, 1, 1)
    assert (a5["kills"], a5["k5"], a5["deaths"]) == (5, 1, 0)
    assert (b2["kills"], b2["deaths"], b2["k2"]) == (2, 2, 0)
    assert _row(f, "a2")["assists"] == 1 and a1["traded_deaths"] == 1 and _row(f, "a4")["traded_deaths"] == 1
    assert (_row(f, "a2")["kills"], _row(f, "a2")["trade_kills"]) == (1, 1)
    assert f["kills"].sum() == 16 and f["deaths"].sum() == 16


def test_adr_ignores_friendly_fire_and_divides_by_rounds_played():
    f = build_fact_player(_hand_match())
    assert _row(f, "a1")["damage"] == 100 and _row(f, "a1")["adr"] == 33.33
    assert _row(f, "a3")["adr"] == 100.0 and _row(f, "b1")["adr"] == 0.0


def test_kast_counts_kill_assist_survive_and_trade():
    f = build_fact_player(_hand_match()).set_index("name")
    # a4 dies in rounds 2 and 3 but round 3's death is traded, so a4 keeps that round (T in KAST).
    expected = {"a1": 2, "a2": 2, "a3": 2, "a4": 2, "a5": 3, "b1": 2, "b2": 3, "b3": 2, "b4": 2, "b5": 2}
    assert f["kast_rounds"].to_dict() == expected
    assert f.loc["a5", "kast_pct"] == 1.0 and f.loc["a1", "kast_pct"] == 0.6667


def test_clutches_start_when_one_player_is_left_against_enemies():
    f = build_fact_player(_hand_match()).set_index("name")
    assert (f.loc["b5", "clutch_attempts"], f.loc["b5", "clutch_wins"]) == (1, 0)   # round 1: 1v4, lost
    assert (f.loc["a5", "clutch_attempts"], f.loc["a5", "clutch_wins"]) == (1, 1)   # round 2: 1v5, won
    assert f["clutch_attempts"].sum() == 2


def test_player_without_events_still_has_a_row_and_a_perfect_survival_round():
    t = _hand_match()
    t["kills"] = t["kills"].iloc[0:0]
    f = build_fact_player(t)
    assert len(f) == 10 and f["kills"].sum() == 0 and f["clutch_attempts"].sum() == 0
    assert f["kast_pct"].eq(1.0).all()          # everyone survives every round


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
