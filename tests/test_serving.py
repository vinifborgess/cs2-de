"""The static JSON a browser fetches: shape, filterability, size and what it must not invent."""
from tests.test_pipeline import make_match

import json
import math

import pandas as pd

from cs2_pipeline.serving import (ROUND_FILTER_FIELDS, Normaliser, export, map_payload,
                                  round_payload)
from cs2_pipeline.silver import bronze_to_silver
from cs2_pipeline.zones import MapZones

ZONES = MapZones("de_test", {"A site": [100.0, 0.0], "B site": [-100.0, 0.0]},
                 {"A site": 5, "B site": 1}, [-200.0, -200.0, 200.0, 200.0])


def test_normaliser_maps_the_extent_onto_the_unit_square():
    n = Normaliser([-200.0, -200.0, 200.0, 200.0])
    assert n.xy(-200.0, -200.0) == (0.0, 0.0)
    assert n.xy(200.0, 200.0) == (1.0, 1.0)
    assert n.xy(0.0, 0.0) == (0.5, 0.5)
    assert n.xy(None, 1.0) == (None, None) and n.xy(float("nan"), 1.0) == (None, None)
    # A degenerate extent must not divide by zero.
    assert all(math.isfinite(v) for v in Normaliser([5.0, 5.0, 5.0, 5.0]).xy(5.0, 5.0))


def test_normalised_numbers_are_short_enough_to_ship():
    df = pd.DataFrame({"x": pd.Series([123.456], dtype="float32"), "y": pd.Series([-77.1], dtype="float32")})
    out = Normaliser([-200.0, -200.0, 200.0, 200.0]).frame(df)
    # Rounding a float32 would leave 0.8086000084877014 in the JSON; this must be 4 decimals.
    for value in (out["nx"].iloc[0], out["ny"].iloc[0]):
        assert len(json.dumps(float(value))) <= 7, value


def test_map_payload_carries_the_extent_and_flags_weak_anchors():
    p = map_payload(ZONES)
    assert p["map_name"] == "de_test" and p["extent"] == ZONES.extent
    a = {z["zone"]: z for z in p["zones"]}
    assert a["A site"]["x"] == 0.75 and a["A site"]["y"] == 0.5 and a["A site"]["weak"] is False
    assert a["B site"]["weak"] is True and a["B site"]["world_x"] == -100.0
    assert "derived from the demos themselves" in p["note"]


def test_map_payload_carries_the_derived_silhouette():
    from cs2_pipeline.zones import OCCUPANCY_GRID, fit
    from tests.test_zones import _bomb, _positions

    p = map_payload(fit("de_shape", _positions(), _bomb(), ["m"]))
    assert len(p["occupancy"]) == OCCUPANCY_GRID
    assert all(set(row) <= {"0", "1"} for row in p["occupancy"])
    assert any("1" in row for row in p["occupancy"])


def _round(mid="srv-base", seed=81, rnd=3, **kw):
    make_match(mid, seed=seed)
    tables = bronze_to_silver(mid, force=True)
    return tables, round_payload(rnd, tables, Normaliser(ZONES.extent), **kw)


def test_round_payload_has_every_layer_the_replay_draws():
    tables, p = _round()
    assert set(p) == {"round_num", "players", "kills", "bomb", "smokes", "blinds", "grenades"}
    assert p["round_num"] == 3 and len(p["players"]) == 10
    one = p["players"][0]
    assert set(one) == {"steamid", "name", "team", "side", "track"}
    assert len(one["steamid"]) == 17                       # not mangled into a float
    for t, x, y, hp in one["track"]:
        assert isinstance(t, float) and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= hp <= 100
    assert p["bomb"] and p["bomb"][0]["event"] == "finish_plant"        # round 3 is the planted one
    assert p["smokes"] and p["blinds"] and p["grenades"]
    assert all("nx" in e and "ny" in e for e in p["blinds"])


def test_the_opening_duel_is_identifiable_in_the_payload():
    tables, p = _round()
    firsts = [k for k in p["kills"] if k["is_first_kill"]]
    assert len(firsts) == 1
    k = firsts[0]
    assert k["kill_order"] == 1 and k["attacker_team"] != k["victim_team"]
    assert 0 <= k["victim_nx"] <= 1 and 0 <= k["attacker_ny"] <= 1
    # Post-round kills travel with the round but are marked, so the page can hide them.
    assert any(k["is_post_round"] for k in p["kills"])
    assert not any(k["is_post_round"] and k["is_first_kill"] for k in p["kills"])


def test_thinning_the_tracks_cuts_samples_without_losing_players():
    tables, full = _round("srv-hz", seed=82, hz=None)
    thinned = round_payload(3, tables, Normaliser(ZONES.extent), hz=0.5)
    assert len(thinned["players"]) == len(full["players"]) == 10
    assert sum(len(p["track"]) for p in thinned["players"]) <= sum(len(p["track"]) for p in full["players"])
    assert all(p["track"] for p in thinned["players"]), "every player must keep at least one sample"


def test_export_writes_an_index_that_can_be_filtered_without_fetching_rounds():
    import tempfile
    from pathlib import Path

    out = Path(tempfile.mkdtemp(prefix="cs2-serve-"))
    for i, mid in enumerate(("srv-x-vs-y-m1-test", "srv-x-vs-y-m2-test")):
        make_match(mid, seed=90 + i)
        bronze_to_silver(mid, force=True)
    res = export(["srv-x-vs-y-m1-test", "srv-x-vs-y-m2-test"], out_dir=out)
    assert res["matches"] == 2 and res["rounds"] == 48 and not res["skipped"]

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert sorted(m["match_id"] for m in index["matches"]) == ["srv-x-vs-y-m1-test", "srv-x-vs-y-m2-test"]
    m = index["matches"][0]
    assert m["map_name"] == "de_test" and m["n_rounds"] == 24 and len(m["rounds"]) == 24
    assert set(ROUND_FILTER_FIELDS) <= set(m["rounds"][0])
    # The filter the product is for: rounds where one team won the opening duel.
    won_opening = [r for r in m["rounds"] if r["fb_team"] == "Alpha"]
    assert won_opening and all(r["fb_player"].startswith("Alpha") for r in won_opening)
    assert all((out / m["match_id"] / f"r{r['round_num']}.json").exists() for r in m["rounds"])
    assert (out / "maps" / "de_test.json").exists()
    assert index["sample_hz"] == 2.0 and index["maps"] == ["de_test"]


def test_export_leaves_out_a_match_that_fails_quality_and_says_which():
    import tempfile
    from pathlib import Path

    out = Path(tempfile.mkdtemp(prefix="cs2-serve-"))
    make_match("srv-ok", seed=95)
    make_match("srv-bad", seed=96)
    for mid in ("srv-ok", "srv-bad"):
        bronze_to_silver(mid, force=True)
    from cs2_pipeline.silver import _silver_path
    path = _silver_path("rounds", "srv-bad")
    r = pd.read_parquet(path)
    r.loc[1, "round_num"] = r.loc[0, "round_num"]
    r.to_parquet(path, index=False)
    res = export(["srv-ok", "srv-bad"], out_dir=out)
    assert res["matches"] == 1 and "srv-bad" in res["skipped"]
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [m["match_id"] for m in index["matches"]] == ["srv-ok"]
    assert "rounds.unique_round_num" in index["skipped"]["srv-bad"]
    assert not (out / "srv-bad").exists()


#: What web/index.html reads. Changing a name here breaks the page, so the page and this list
#: have to move together - that is the point of the test below.
PAGE_CONTRACT = {
    "index": ["matches", "sample_hz", "skipped", "maps"],
    "match": ["match_id", "map_name", "team_a", "team_b", "score_a", "score_b", "n_rounds",
              "rounds", "event"],
    "round_row": ["round_num", "half", "ct_team", "t_team", "winner", "winner_team", "reason",
                  "duration_s", "bomb_site", "planted", "plant_t_s", "fb_team", "fb_player",
                  "fb_victim", "fb_weapon", "fb_t_s", "fb_won", "fb_zone", "ct_buy", "t_buy"],
    "map": ["map_name", "extent", "zones", "occupancy", "note"],
    "zone": ["zone", "x", "y", "support", "weak"],
    "payload": ["round_num", "players", "kills", "bomb", "smokes", "blinds", "grenades"],
    "player": ["steamid", "name", "team", "side", "track"],
    "kill": ["t_s", "kill_order", "is_first_kill", "is_post_round", "attacker_name",
             "attacker_side", "victim_name", "victim_side", "weapon", "headshot",
             "attacker_nx", "attacker_ny", "victim_nx", "victim_ny"],
}


def test_the_exported_shape_is_the_one_the_page_reads():
    import tempfile
    from pathlib import Path

    out = Path(tempfile.mkdtemp(prefix="cs2-contract-"))
    make_match("srv-contract-vs-b-m1-test", seed=99)
    bronze_to_silver("srv-contract-vs-b-m1-test", force=True)
    export(["srv-contract-vs-b-m1-test"], out_dir=out)

    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    match = index["matches"][0]
    row = match["rounds"][0]
    mp = json.loads((out / "maps" / "de_test.json").read_text(encoding="utf-8"))
    payload = json.loads((out / match["match_id"] / f"r{row['round_num']}.json").read_text(encoding="utf-8"))
    sample = {"index": index, "match": match, "round_row": row, "map": mp, "zone": mp["zones"][0],
              "payload": payload, "player": payload["players"][0], "kill": payload["kills"][0]}
    for part, fields in PAGE_CONTRACT.items():
        missing = [f for f in fields if f not in sample[part]]
        assert not missing, f"web/index.html reads {part}.{missing} and the export no longer has it"

    # Tracks are [t_s, x, y, health] with the coordinates already normalised.
    t, x, y, hp = payload["players"][0]["track"][0]
    assert isinstance(t, float) and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= hp <= 100
    # Every round the index promises has a file.
    for r in match["rounds"]:
        assert (out / match["match_id"] / f"r{r['round_num']}.json").exists()


def test_the_replay_agrees_with_the_kill_feed():
    """The victim is alive in the last sample before the kill and dead in the first one after.

    Stated against the samples themselves rather than a fixed number of seconds, so it holds
    whatever rate the tracks were thinned to.
    """
    tables, p = _round("srv-agree", seed=98)
    first = next(k for k in p["kills"] if k["is_first_kill"])
    victim = next(pl for pl in p["players"] if pl["name"] == first["victim_name"])
    before = [s for s in victim["track"] if s[0] <= first["t_s"]]
    after = [s for s in victim["track"] if s[0] > first["t_s"]]
    assert before and before[-1][3] > 0, "the victim should be alive going into the kill"
    assert after and after[0][3] == 0, "and dead in the next sample"


def test_json_holds_no_nan_and_every_value_is_serialisable():
    tables, p = _round("srv-nan", seed=97)
    text = json.dumps(p)                                   # would raise on a pandas NA
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["round_num"] == 3


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
