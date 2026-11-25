"""Match catalog: id parsing, file handling, and what it must never invent."""
from tests.test_pipeline import make_match  # noqa: F401  (points CS2_DATA_DIR at a scratch dir)

import json

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


def test_event_and_date_are_never_derived_only_the_catalog_sets_them():
    _clear()
    e = entry_for("spirit-vs-furia-m2-dust2")
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
