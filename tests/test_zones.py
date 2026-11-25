"""Map zones: fitted from the demos, stable once stored, and never named beyond the evidence."""
from tests.test_pipeline import make_match  # noqa: F401  (points the lake at a scratch dir)

import json

import numpy as np
import pandas as pd

from cs2_pipeline import zones as Z


def _positions(n_rounds=4):
    """Players starting at two spawns, then walking towards the middle."""
    rows = []
    for rnd in range(1, n_rounds + 1):
        for side, (sx, sy) in (("CT", (1000.0, 1000.0)), ("T", (-1000.0, -1000.0))):
            for p in range(5):
                for i, t in enumerate((0, 100, 200)):
                    rows.append({"match_id": "m", "round_num": rnd, "steamid": f"{side}{p}",
                                 "side": side, "tick": rnd * 1000 + t,
                                 "x": sx + i * 10 + p, "y": sy + i * 10 + p})
    return pd.DataFrame(rows)


def _bomb():
    rows = []
    for i in range(4):                                  # A site near (2000, -100), B near (-2000, 100)
        rows.append({"event": "finish_plant", "bombsite": "A", "x": 2000.0 + i, "y": -100.0 + i})
        rows.append({"event": "finish_plant", "bombsite": "B", "x": -2000.0 + i, "y": 100.0 + i})
    rows.append({"event": "defuse", "bombsite": "A", "x": 9999.0, "y": 9999.0})   # not a plant
    return pd.DataFrame(rows)


def test_sites_come_from_plants_and_spawns_from_the_first_live_sample():
    z = Z.fit("de_fixture", _positions(), _bomb(), ["m"])
    # No Mid here: every position in this fixture sits on a spawn, so there is no ground
    # between the anchors to name, and inventing one would be worse than leaving it out.
    assert set(z.names) == {"A site", "B site", "CT spawn", "T spawn"}
    assert z.anchors["A site"] == [2001.5, -98.5] and z.anchors["B site"] == [-1998.5, 101.5]
    # The defuse row must not pull the A site anchor towards (9999, 9999).
    assert z.support["A site"] == 4 and z.support["B site"] == 4
    assert z.anchors["CT spawn"][0] == 1002.0 and z.anchors["T spawn"][0] == -998.0
    assert z.support["CT spawn"] == 4 and z.support["T spawn"] == 4
    assert z.extent == [-1000.0, -1000.0, 1024.0, 1024.0]


def test_every_plant_is_labelled_with_its_own_site():
    """The ground-truth property: the sites are where the bomb was planted."""
    z = Z.fit("de_fixture", _positions(), _bomb(), ["m"])
    plants = _bomb().query("event == 'finish_plant'").reset_index(drop=True)
    labelled = z.label(plants["x"], plants["y"])
    assert (labelled == plants["bombsite"] + " site").all()


def test_labels_are_the_nearest_anchor_and_missing_coordinates_stay_missing():
    z = Z.fit("de_fixture", _positions(), _bomb(), ["m"])
    pts = pd.DataFrame({"x": [2000.0, -2000.0, 1000.0, -1000.0, np.nan, 2000.0],
                        "y": [-100.0, 100.0, 1000.0, -1000.0, 5.0, np.nan]})
    got = z.label(pts["x"], pts["y"])
    assert got.tolist()[:4] == ["A site", "B site", "CT spawn", "T spawn"]
    assert pd.isna(got.iloc[4]) and pd.isna(got.iloc[5])
    assert Z.MapZones("m", {}, {}, [0, 0, 1, 1]).label([1.0], [1.0]).isna().all()


def test_cells_are_a_fixed_size_grid_from_the_lower_left_corner():
    z = Z.MapZones("m", {"A site": [0.0, 0.0]}, {"A site": 9}, [-1000.0, -1000.0, 1000.0, 1000.0])
    got = z.cell([-1000.0, -999.0, -744.0, 0.0, np.nan], [-1000.0, -1000.0, -1000.0, 0.0, 0.0])
    assert got.tolist()[:4] == ["x0y0", "x0y0", "x1y0", "x3y3"]
    assert pd.isna(got.iloc[4])
    assert z.cell([], []).empty


def test_thin_evidence_is_reported_not_hidden():
    bomb = pd.DataFrame([{"event": "finish_plant", "bombsite": "A", "x": 2000.0, "y": 0.0},
                         {"event": "finish_plant", "bombsite": "B", "x": -2000.0, "y": 0.0},
                         {"event": "finish_plant", "bombsite": "B", "x": -2010.0, "y": 0.0},
                         {"event": "finish_plant", "bombsite": "B", "x": -1990.0, "y": 0.0}])
    z = Z.fit("de_thin", _positions(), bomb, ["m"])
    assert z.weak() == ["A site"] and z.support["A site"] == 1 and "B site" not in z.weak()


def test_mid_is_the_ground_that_is_far_from_every_other_anchor():
    pos = _positions()
    crossing = pd.DataFrame([{"match_id": "m", "round_num": 1, "steamid": "x", "side": "CT",
                              "tick": 50_000 + i, "x": 0.0, "y": 0.0} for i in range(60)])
    z = Z.fit("de_mid", pd.concat([pos, crossing], ignore_index=True), _bomb(), ["m"])
    assert "Mid" in z.names and z.anchors["Mid"] == [0.0, 0.0] and z.support["Mid"] == 60
    assert z.label([0.0], [0.0]).iloc[0] == "Mid"
    # Too few samples in the middle is not a zone.
    thin = pd.concat([pos, crossing.head(10)], ignore_index=True)
    assert "Mid" not in Z.fit("de_thinmid", thin, _bomb(), ["m"]).names


def test_a_map_without_plants_still_gets_its_spawns():
    z = Z.fit("de_noplant", _positions(), pd.DataFrame(columns=["event", "bombsite", "x", "y"]), ["m"])
    assert set(z.names) == {"CT spawn", "T spawn"}
    assert z.label([1000.0], [1000.0]).iloc[0] == "CT spawn"


def test_fitting_without_positions_is_an_error_not_an_empty_map():
    try:
        Z.fit("de_empty", pd.DataFrame(columns=["x", "y", "tick", "side", "round_num", "steamid", "match_id"]),
              _bomb(), [])
    except ValueError as exc:
        assert "no positions" in str(exc)
    else:
        raise AssertionError("fitting a map with no positions must raise")


def test_zones_are_stored_once_and_not_refitted_behind_your_back():
    name = "de_stable"
    Z._path(name).unlink(missing_ok=True)
    assert Z.load(name) is None
    first = Z.ensure(name, _positions(), _bomb(), ["m"])
    moved = _positions().assign(x=lambda d: d["x"] + 5000)
    again = Z.ensure(name, moved, _bomb(), ["m", "m2"])
    assert again.anchors == first.anchors, "a second run must reuse the stored anchors"
    refitted = Z.ensure(name, moved, _bomb(), ["m", "m2"], refit=True)
    assert refitted.anchors["CT spawn"][0] == first.anchors["CT spawn"][0] + 5000
    assert refitted.fitted_from == ["m", "m2"]
    assert Z.load(name).anchors == refitted.anchors        # the refit was stored
    Z._path(name).unlink(missing_ok=True)


def test_an_unreadable_zone_file_reads_as_missing():
    name = "de_broken"
    Z.ZONES_DIR.mkdir(parents=True, exist_ok=True)
    Z._path(name).write_text("{not json", encoding="utf-8")
    assert Z.load(name) is None
    Z._path(name).write_text(json.dumps({"map_name": name}), encoding="utf-8")   # missing fields
    assert Z.load(name) is None
    Z._path(name).unlink(missing_ok=True)


def test_as_frame_publishes_one_row_per_anchor():
    z = Z.fit("de_fixture", _positions(), _bomb(), ["m1", "m2"])
    f = Z.as_frame([z])
    assert len(f) == len(z.names) and set(f["zone"]) == set(z.names)
    assert (f["map_name"] == "de_fixture").all() and (f["fitted_from"] == "m1,m2").all()
    assert f.loc[f["zone"] == "A site", "support"].iloc[0] == 4
    assert Z.as_frame([]).empty


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
