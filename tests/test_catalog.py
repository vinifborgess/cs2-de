"""Match catalog: id parsing, file handling, and what it must never invent."""
from tests.test_pipeline import make_match  # noqa: F401  (points CS2_DATA_DIR at a scratch dir)

import json

import pandas as pd

import cs2_pipeline.catalog as cat
from cs2_pipeline.catalog import CATALOG_FIELDS, derive_from_match_id, entry_for, incomplete, load_catalog, upsert
from cs2_pipeline.silver import bronze_to_silver


def _write(payload) -> None:
    cat.CATALOG_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _clear() -> None:
    cat.CATALOG_PATH.unlink(missing_ok=True)


def test_series_and_map_order_are_read_off_the_match_id():
    assert derive_from_match_id("spirit-vs-furia-m2-dust2") == {"series_id": "furia-vs-spirit", "map_order": 2}
    # Teams are sorted, so both halves of a series land in the same series_id.
    assert (derive_from_match_id("furia-vs-spirit-m3-mirage")["series_id"]
            == derive_from_match_id("spirit-vs-furia-m1-ancient")["series_id"])
    assert derive_from_match_id("team-a-vs-team-b-m1-mirage")["series_id"] == "team-a-vs-team-b"
    for bad in ("random-name", "furia-spirit-mirage", ""):
        assert derive_from_match_id(bad) == {}


def test_an_event_in_the_file_name_needs_the_real_team_names_to_be_read():
    mid = "blast-rivals-fall-2026-furia-vs-spirit-m1-ancient"
    # Without them the prefix is left alone: it could be part of a team's name.
    assert "event" not in derive_from_match_id(mid)
    got = derive_from_match_id(mid, ["FURIA", "Spirit"])
    assert got["event"] == "blast-rivals-fall-2026"
    assert got["series_id"] == "blast-rivals-fall-2026-furia-vs-spirit"


def test_a_hyphenated_team_name_is_not_mistaken_for_an_event():
    teams = ["Natus Vincere", "FURIA"]
    plain = derive_from_match_id("natus-vincere-vs-furia-m1-nuke", teams)
    assert "event" not in plain and plain["series_id"] == "furia-vs-natus-vincere"
    with_event = derive_from_match_id("iem-cologne-2026-natus-vincere-vs-furia-m1-nuke", teams)
    assert with_event["event"] == "iem-cologne-2026"
    assert with_event["series_id"] == "iem-cologne-2026-furia-vs-natus-vincere"
    # A name the demo does not report cannot strip anything off the front.
    assert "event" not in derive_from_match_id("iem-cologne-2026-natus-vincere-vs-furia-m1-nuke", ["G2", "NAVI"])


def test_a_team_name_only_matches_on_a_hyphen_boundary():
    # "myalpha" is not Alpha with an event called "my": there is no boundary to cut on.
    got = derive_from_match_id("myalpha-vs-bravo-m1-test", ["Alpha", "Bravo"])
    assert "event" not in got and got["series_id"] == "bravo-vs-myalpha"
    # With the boundary there, the same name is read as the team.
    got = derive_from_match_id("my-alpha-vs-bravo-m1-test", ["Alpha", "Bravo"])
    assert got["event"] == "my" and got["series_id"] == "my-alpha-vs-bravo"


def test_a_date_is_never_derived_and_neither_is_a_stage():
    _clear()
    e = entry_for("spirit-vs-furia-m2-dust2", teams=["FURIA", "Spirit"])
    assert e["event"] is None and e["stage"] is None and e["played_on"] is None
    assert (e["series_id"], e["map_order"], e["meta_source"]) == ("furia-vs-spirit", 2, "derived")
    assert entry_for("no-convention-here")["meta_source"] == "none"


def test_catalog_values_win_over_derived_ones_and_the_source_says_so():
    _write({"matches": {"spirit-vs-furia-m2-dust2": {
        "event": "BLAST Rivals Fall 2026", "stage": "Grand Final", "played_on": "2026-11-15",
        "map_order": 9, "best_of": 5}}})
    e = entry_for("spirit-vs-furia-m2-dust2")
    assert e["event"] == "BLAST Rivals Fall 2026" and e["map_order"] == 9      # not the 2 from the id
    assert e["series_id"] == "furia-vs-spirit" and e["meta_source"] == "catalog"
    assert set(e) == set(CATALOG_FIELDS) | {"meta_source"}
    _clear()


def test_unusable_values_are_dropped_not_guessed():
    _write({"matches": {"m": {"played_on": "15/11/2026", "map_order": "second", "best_of": ""}}})
    e = entry_for("m")
    assert e["played_on"] is None and e["map_order"] is None and e["best_of"] is None
    _write({"matches": {"m": {"played_on": "2026-11-15T20:30:00Z"}}})
    assert entry_for("m")["played_on"] == "2026-11-15"
    _clear()


def test_a_broken_or_missing_catalog_file_does_not_stop_the_pipeline():
    _clear()
    assert load_catalog() == {}
    cat.CATALOG_PATH.write_text("{not json", encoding="utf-8")
    assert load_catalog() == {}                      # logged as an error, treated as empty
    _write({"matches": [1, 2, 3]})
    assert load_catalog() == {}
    _clear()


def test_upsert_keeps_other_matches_and_other_fields():
    _clear()
    upsert("a", event="E1", played_on="2026-01-02")
    upsert("b", event="E2")
    upsert("a", stage="Final")                        # must not wipe a's event
    data = load_catalog()
    assert data["a"] == {"event": "E1", "played_on": "2026-01-02", "stage": "Final"}
    assert data["b"] == {"event": "E2"}
    try:
        upsert("a", trophy="yes")
    except ValueError as exc:
        assert "trophy" in str(exc)
    else:
        raise AssertionError("unknown fields must be refused")
    _clear()


def test_incomplete_lists_matches_without_event_or_date():
    _write({"matches": {"has-all": {"event": "E", "played_on": "2026-01-02"},
                        "no-date": {"event": "E"}}})
    out = incomplete(["has-all", "no-date", "never-seen"])
    assert "has-all" not in out
    assert out["no-date"] == ["played_on"] and out["never-seen"] == ["event", "played_on"]
    _clear()


def test_a_derived_event_still_counts_as_missing_because_it_is_only_a_slug():
    # The generator's teams are Alpha and Bravo, which is what resolves the prefix.
    mid = "slugcup-alpha-vs-bravo-m1-test"
    make_match(mid, seed=62, n_rounds=12)
    m = bronze_to_silver(mid, force=True)["matches"].iloc[0]
    assert m["event"] == "slugcup" and m["meta_source"] == "derived"
    assert m["series_id"] == "slugcup-alpha-vs-bravo" and m["map_order"] == 1
    # A slug groups matches but is not the event's name, so it still needs filling in.
    assert incomplete([mid])[mid] == ["event", "played_on"]


def test_a_prefix_that_is_not_a_team_boundary_is_left_in_the_series_id():
    mid = "mystery-cat-vs-dog-m1-test"
    make_match(mid, seed=63, n_rounds=12)
    m = bronze_to_silver(mid, force=True)["matches"].iloc[0]
    assert pd.isna(m["event"]), "nothing in the id matches a team the demo reports"
    assert m["series_id"] == "dog-vs-mystery-cat"


def test_silver_matches_carries_the_event_context():
    make_match("cat-vs-dog-m2-test", seed=61, n_rounds=12)
    _write({"matches": {"cat-vs-dog-m2-test": {"event": "Test Cup", "stage": "Semi", "played_on": "2026-05-01"}}})
    m = bronze_to_silver("cat-vs-dog-m2-test", force=True)["matches"].iloc[0]
    assert m["event"] == "Test Cup" and m["stage"] == "Semi" and str(m["played_on"]) == "2026-05-01"
    assert m["series_id"] == "cat-vs-dog" and m["map_order"] == 2 and m["meta_source"] == "catalog"
    _clear()
    m = bronze_to_silver("cat-vs-dog-m2-test", force=True)["matches"].iloc[0]
    import pandas as pd
    assert pd.isna(m["event"]) and m["meta_source"] == "derived"


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
