"""fact_round: grain, joins and edge cases, checked against what the synthetic match is known to contain."""
from tests.test_pipeline import make_match

import pandas as pd

from cs2_pipeline.facts import FACT_ROUND_COLUMNS, build_fact_round, half_of
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


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
