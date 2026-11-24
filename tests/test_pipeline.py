"""Synthetic tests for Bronze -> Silver -> Gold on the awpy 3 schema. No .dem needed.

Run with ``python -m tests.test_pipeline`` (or ``pytest``). Set CS2_TEST_DEMO to a
real .dem path to also run an end-to-end check on it.
"""
import os
import tempfile
from pathlib import Path

# The data lake location is read at import time: point it at a scratch directory first.
os.environ["CS2_DATA_DIR"] = tempfile.mkdtemp(prefix="cs2-test-")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from cs2_pipeline.bronze import read_bronze, write_bronze_tables  # noqa: E402
from cs2_pipeline.config import SILVER_SCHEMAS  # noqa: E402
from cs2_pipeline.gold import compute_round_outcomes, run_gold  # noqa: E402
from cs2_pipeline.silver import (  # noqa: E402
    assign_round, bronze_to_silver, load_silver_lake, normalize_side,
)

TICK_RATE = 64.0
STEAM_BASE = 76561198000000000  # 17 digits: beyond what float64 can hold exactly
SIDE = {"CT": "counter-terrorist", "T": "terrorist"}
FREEZE, LIVE = 1280, 6000


def make_match(match_id: str, seed: int, n_rounds: int = 24, fb_win_prob: float = 0.7) -> dict:
    """Write a synthetic Bronze match and return what Silver is expected to produce."""
    rng = np.random.default_rng(seed)
    teams = {"Alpha": [], "Bravo": []}
    roster = []
    for t_i, team in enumerate(teams):
        for p in range(5):
            sid = STEAM_BASE + t_i * 10 + p
            teams[team].append(sid)
            roster.append((sid, f"{team}_p{p + 1}", team))
    team_of = {sid: team for sid, _, team in roster}
    name_of = {sid: name for sid, name, _ in roster}

    def side_of(team: str, rnd: int) -> str:  # Alpha is CT in the first half, T after the swap
        first_half = rnd <= n_rounds // 2
        return ("CT" if first_half else "T") if team == "Alpha" else ("T" if first_half else "CT")

    players = pl.DataFrame({
        "steamid": [0] + [r[0] for r in roster], "name": ["CSTV"] + [r[1] for r in roster],
        "side": [None] + [SIDE["CT"]] * 10, "team_clan_name": [None] + [r[2] for r in roster],
    }, schema_overrides={"steamid": pl.UInt64})

    rounds, kills, damages, snaps, smokes, econ, bomb = [], [], [], [], [], [], []
    expected_first, expected_kills = {}, 0

    def kill_row(tick, a, v, weapon="ak47", a_side=None, v_side=None):
        return {"tick": tick, "attacker_steamid": a, "attacker_name": name_of.get(a),
                "attacker_side": a_side, "attacker_x": 1.0, "attacker_y": 2.0, "attacker_z": 3.0,
                "victim_steamid": v, "victim_name": name_of.get(v), "victim_side": v_side,
                "victim_x": 4.0, "victim_y": 5.0, "victim_z": 6.0, "assister_name": None,
                "weapon": weapon, "headshot": False, "is_trade": False, "victim_traded": False,
                "thrusmoke": False, "noscope": False, "penetrated": False, "assist_flash": False,
                "hitgroup_name": "chest"}

    # A knife round at the start (must be dropped) and a warm-up kill before it.
    tick = 1000
    kills.append(kill_row(50, teams["Alpha"][0], teams["Bravo"][0], a_side=SIDE["CT"], v_side=SIDE["T"]))
    rounds.append({"round_num": 0, "start_tick": tick, "freeze_end_tick": tick + FREEZE,
                   "end_tick": tick + FREEZE + 2000, "official_end_tick": tick + FREEZE + 2400,
                   "winner_side": SIDE["CT"], "reason": 7, "reason_name": "bomb_defused", "is_knife_round": True})
    tick += FREEZE + 2400

    for rnd in range(1, n_rounds + 1):
        start, fe = tick, tick + FREEZE
        end = fe + LIVE
        fb_team = str(rng.choice(["Alpha", "Bravo"]))
        other = "Bravo" if fb_team == "Alpha" else "Alpha"
        fb_side = side_of(fb_team, rnd)
        win_side = fb_side if rng.random() < fb_win_prob else ("T" if fb_side == "CT" else "CT")
        rounds.append({"round_num": rnd, "start_tick": start, "freeze_end_tick": fe, "end_tick": end,
                       "official_end_tick": end + 448, "winner_side": SIDE[win_side], "reason": 9,
                       "reason_name": "terrorists_win" if win_side == "T" else "ct_win", "is_knife_round": False})
        n_k = int(rng.integers(3, 7))
        dead: dict[int, int] = {}  # steamid -> tick of death; nobody dies twice or acts after dying
        for i in range(n_k):
            a_team = fb_team if i == 0 else str(rng.choice(["Alpha", "Bravo"]))
            v_team = "Bravo" if a_team == "Alpha" else "Alpha"
            attackers = [x for x in teams[a_team] if x not in dead]
            victims = [x for x in teams[v_team] if x not in dead]
            if not attackers or not victims:
                continue
            a, v = int(rng.choice(attackers)), int(rng.choice(victims))
            k_tick = fe + 400 + i * 200
            dead[v] = k_tick
            kills.append(kill_row(k_tick, a, v, a_side=SIDE[side_of(a_team, rnd)], v_side=SIDE[side_of(v_team, rnd)]))
            damages.append({"tick": k_tick, "attacker_steamid": a, "attacker_name": name_of[a],
                            "attacker_side": SIDE[side_of(a_team, rnd)], "victim_steamid": v, "victim_name": name_of[v],
                            "victim_side": SIDE[side_of(v_team, rnd)], "weapon": "ak47", "dmg_health": 120, "dmg_armor": 0,
                            "hitgroup_name": "head", "health_pre": 100, "health_post": 0})
            expected_kills += 1
            if i == 0:
                expected_first[rnd] = name_of[a]
        # Engine noise the Silver layer must remove.
        kills.append(kill_row(fe + 100, None, int(teams[other][0]), weapon="world", v_side=SIDE[side_of(other, rnd)]))
        kills.append(kill_row(fe + 150, teams[fb_team][0], teams[fb_team][1], a_side=SIDE[fb_side], v_side=SIDE[fb_side]))
        kills.append(kill_row(end + 100, teams["Alpha"][0], teams["Bravo"][0], a_side=SIDE["CT"], v_side=SIDE["T"]))  # post-round
        if rnd == 3:
            bomb.append({"tick": fe + 3000, "event": "finish_plant", "entity_id": 1, "entity_serial": 1,
                         "steamid": teams[fb_team][0], "name": name_of[teams[fb_team][0]], "bombsite": "B",
                         "x": 1.0, "y": 1.0, "z": 1.0})
        smokes.append({"start_tick": fe + 500, "end_tick": fe + 1900, "thrower_name": name_of[teams["Alpha"][0]],
                       "thrower_steamid": teams["Alpha"][0], "thrower_side": SIDE[side_of("Alpha", rnd)],
                       "entity_id": 2, "entity_serial": 2, "x": 7.0, "y": 8.0, "z": 9.0})
        for side in ("CT", "T"):
            econ.append({"round_num": rnd, "side": SIDE[side], "equipment_value": 20000, "buy_type": "full", "n_players": 5})
        for t in (fe - 64, fe, fe + 1000, end):  # first one is freeze time: must be dropped
            for sid, name, team in roster:
                hp = 0 if (sid in dead and t >= dead[sid]) else 100
                snaps.append({"tick": t, "steamid": sid, "name": name, "side": SIDE[side_of(team, rnd)],
                              "x": 10.0, "y": 20.0, "z": 30.0, "yaw": 90.0, "pitch": 0.0, "health": hp,
                              "armor": 100, "active_weapon": "ak47", "has_bomb": False, "is_blinded": False,
                              "is_defusing": False, "money": 4000, "equipment_value": 4000})
        tick = end + 448
    for rnd in (0,):
        for side in ("CT", "T"):
            econ.append({"round_num": rnd, "side": SIDE[side], "equipment_value": 0, "buy_type": "eco", "n_players": 5})

    tables = {
        "header": pl.DataFrame([{"map_name": "de_test", "server_name": "synthetic", "tick_rate": TICK_RATE}]),
        "players": players,
        "rounds": pl.DataFrame(rounds),
        "kills": pl.DataFrame(kills, schema_overrides={"attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64}, infer_schema_length=None),
        "damages": pl.DataFrame(damages),
        "bomb": pl.DataFrame(bomb) if bomb else pl.DataFrame(schema={"tick": pl.Int64, "event": pl.String, "bombsite": pl.String}),
        "smokes": pl.DataFrame(smokes),
        "round_economy": pl.DataFrame(econ),
        "snapshots": pl.DataFrame(snaps),
    }
    write_bronze_tables(match_id, tables, {"demo_sha256": "synthetic"})
    return {"n_rounds": n_rounds, "first": expected_first, "n_kills": expected_kills}


# --- tests ----------------------------------------------------------------------

def test_read_bronze_keeps_steamid_precision():
    sid = 76561198134401925
    write_bronze_tables("precision", {"kills": pl.DataFrame(
        {"attacker_steamid": [sid, None], "tick": [1, 2]}, schema_overrides={"attacker_steamid": pl.UInt64})}, {})
    got = read_bronze("precision", "kills")["attacker_steamid"]
    assert got.iloc[0] == str(sid), got.iloc[0]  # float64 would give ...1920 or similar
    assert pd.isna(got.iloc[1])


def test_normalize_side():
    out = normalize_side(pd.Series(["terrorist", "counter-terrorist", " CT ", "t", None, "spectator"]))
    assert out.tolist()[:4] == ["T", "CT", "CT", "T"] and out.isna().tolist()[4:] == [True, True]


def test_blank_clan_names_are_missing_not_a_team():
    from cs2_pipeline.silver import clean_players
    raw = pd.DataFrame({"steamid": ["0", "1", "2", "3", "4"], "name": ["CSTV", "a", "b", "c", "d"],
                        "team_clan_name": [None, "  ", "", "Alpha ", None]})
    out = clean_players(raw, "m")
    assert out["steamid"].tolist() == ["3"] and out["team"].tolist() == ["Alpha"]


def test_assign_round_drops_warmup_and_gaps():
    rounds = pd.DataFrame({"round_num": [1, 2], "start_tick": [100, 500], "end_tick": [400, 900]})
    ev = pd.DataFrame({"tick": [50, 100, 400, 450, 500, 901]})
    out = assign_round(ev, rounds)
    assert out["tick"].tolist() == [100, 400, 500] and out["round_num"].tolist() == [1, 1, 2]


def test_silver_contracts_noise_and_sides():
    exp = make_match("syn-a", seed=1)
    s = bronze_to_silver("syn-a", force=True)
    assert set(s) == set(SILVER_SCHEMAS)
    for name, df in s.items():
        assert list(df.columns) == [c for c in SILVER_SCHEMAS[name] if c in df.columns], name

    r, k = s["rounds"], s["kills"]
    assert len(r) == exp["n_rounds"] and 0 not in r["round_num"].tolist()  # knife round dropped
    assert len(k) == exp["n_kills"]                                        # world, team and post-round kills removed
    assert set(k["attacker_side"]) <= {"CT", "T"} and (k["attacker_team"] != k["victim_team"]).all()
    assert (k["t_s"] >= 0).all() and k["is_first_kill"].sum() == exp["n_rounds"]
    first = k[k["is_first_kill"]].set_index("round_num")["attacker_name"].to_dict()
    assert first == exp["first"]

    half = exp["n_rounds"] // 2
    assert (r.loc[r["round_num"] <= half, "ct_team"] == "Alpha").all()     # sides swap at halftime
    assert (r.loc[r["round_num"] > half, "ct_team"] == "Bravo").all()
    assert r.loc[r["round_num"] == 3, "bomb_site"].iloc[0] == "B" and (r["bomb_site"] != "B").sum() == exp["n_rounds"] - 1
    m = s["matches"].iloc[0]
    assert m["score_a"] + m["score_b"] == exp["n_rounds"]
    # Freeze-time snapshots are dropped: 3 live samples x 10 players per round.
    assert len(s["positions"]) == exp["n_rounds"] * 3 * 10 and (s["positions"]["t_s"] >= 0).all()
    assert s["positions"]["steamid"].str.len().eq(17).all()


def test_silver_is_idempotent():
    make_match("syn-idem", seed=2)
    bronze_to_silver("syn-idem", force=True)
    from cs2_pipeline.silver import _silver_path
    path = _silver_path("kills", "syn-idem")
    before = path.stat().st_mtime_ns
    bronze_to_silver("syn-idem")
    assert path.stat().st_mtime_ns == before


def test_silver_rebuilds_when_bronze_or_contract_changes():
    import cs2_pipeline.silver as silver_mod
    from cs2_pipeline.silver import _silver_path
    make_match("syn-stale", seed=5, n_rounds=12)
    bronze_to_silver("syn-stale", force=True)
    path = _silver_path("kills", "syn-stale")

    t0 = path.stat().st_mtime_ns
    bronze_to_silver("syn-stale")
    assert path.stat().st_mtime_ns == t0, "unchanged Bronze must not trigger a rebuild"

    make_match("syn-stale", seed=6, n_rounds=12)  # Bronze rewritten (e.g. re-parsed)
    bronze_to_silver("syn-stale")
    t1 = path.stat().st_mtime_ns
    assert t1 != t0, "a re-written Bronze must invalidate Silver"

    old = silver_mod.SILVER_VERSION
    silver_mod.SILVER_VERSION = old + 1  # contract bump
    try:
        bronze_to_silver("syn-stale")
    finally:
        silver_mod.SILVER_VERSION = old
    assert path.stat().st_mtime_ns != t1, "a new contract version must invalidate Silver"


def test_batch_exit_codes_and_bad_demos():
    import run_pipeline
    from cs2_pipeline.config import RAW_DIR

    def reset(*names_and_bytes):
        for f in RAW_DIR.glob("*.dem"):
            f.unlink()
        for name, data in names_and_bytes:
            (RAW_DIR / name).write_bytes(data)

    real_process, real_gold, real_lake = run_pipeline.process_one, run_pipeline.run_gold, run_pipeline.load_silver_lake
    run_pipeline.run_gold, run_pipeline.load_silver_lake = (lambda s: None), (lambda t=None: {})
    try:
        # Invalid files go through the real parser: skipped, never raised, nothing ingested -> 1.
        reset(("empty.dem", b""), ("fake.dem", b"not a demo"))
        assert run_pipeline.main(["--all"]) == 1

        # Only "ok.dem" succeeds: the batch completes but reports the failures -> 2.
        run_pipeline.process_one = lambda path, match_id, force: path.name == "ok.dem"
        reset(("empty.dem", b""), ("fake.dem", b"not a demo"), ("ok.dem", b"x"))
        assert run_pipeline.main(["--all"]) == 2

        # Everything succeeds -> 0.
        reset(("ok.dem", b"x"))
        assert run_pipeline.main(["--all"]) == 0
    finally:
        run_pipeline.process_one, run_pipeline.run_gold, run_pipeline.load_silver_lake = real_process, real_gold, real_lake
        reset()


def test_check_command_exit_codes():
    import run_pipeline
    from cs2_pipeline.silver import _silver_path
    make_match("chk", seed=7)
    bronze_to_silver("chk", force=True)
    assert run_pipeline.main(["--check", "--match-id", "chk"]) == 0

    players = _silver_path("players", "chk")  # a 9-player roster is a warning, not an error
    df = pd.read_parquet(players)
    df.iloc[:-1].to_parquet(players, index=False)
    assert run_pipeline.main(["--check", "--match-id", "chk"]) == 0
    assert run_pipeline.main(["--check", "--match-id", "chk", "--strict"]) == 1

    rounds = _silver_path("rounds", "chk")  # duplicated round numbers are an error
    df = pd.read_parquet(rounds)
    df.loc[1, "round_num"] = df.loc[0, "round_num"]
    df.to_parquet(rounds, index=False)
    assert run_pipeline.main(["--check", "--match-id", "chk"]) == 1
    assert run_pipeline.main(["--check", "--match-id", "does-not-exist"]) == 1


def test_failed_quality_checks_quarantine_the_match():
    import run_pipeline
    from cs2_pipeline.quality import Check
    from cs2_pipeline.silver import quarantined

    make_match("qa-good", seed=8, n_rounds=12)
    make_match("qa-bad", seed=9, n_rounds=12)
    real_parse, real_checks = run_pipeline.parse_demo_to_bronze, run_pipeline.run_checks
    run_pipeline.parse_demo_to_bronze = lambda path, mid, force=False: None  # Bronze already written above
    fail = Check("kills.inside_round", "error", False, "3 kills outside [0, duration]")
    try:
        run_pipeline.run_checks = lambda mid: [fail] if mid == "qa-bad" else real_checks(mid)
        assert run_pipeline.process_one(Path("qa-good.dem"), "qa-good", True) is True
        assert run_pipeline.process_one(Path("qa-bad.dem"), "qa-bad", True) is False
        assert "qa-bad" in quarantined() and "qa-good" not in quarantined()
        assert "kills.inside_round" in quarantined()["qa-bad"][0]

        lake = load_silver_lake(("rounds",))
        assert "qa-bad" not in set(lake["rounds"]["match_id"]) and "qa-good" in set(lake["rounds"]["match_id"])
        assert "qa-bad" in set(load_silver_lake(("rounds",), include_quarantined=True)["rounds"]["match_id"])

        run_pipeline.run_checks = real_checks  # fixed data passes the next run: quarantine lifts
        assert run_pipeline.process_one(Path("qa-bad.dem"), "qa-bad", True) is True
        assert "qa-bad" not in quarantined()
    finally:
        run_pipeline.parse_demo_to_bronze, run_pipeline.run_checks = real_parse, real_checks


def test_gold_end_to_end_on_the_lake():
    make_match("syn-win", seed=3, fb_win_prob=1.0)
    make_match("syn-mix", seed=4, fb_win_prob=0.5)
    for mid in ("syn-win", "syn-mix"):
        bronze_to_silver(mid, force=True)
    lake = load_silver_lake(("kills", "rounds"))
    only = {t: df[df["match_id"] == "syn-win"] for t, df in lake.items()}
    assert compute_round_outcomes(only["kills"], only["rounds"])["first_blood_won"].all()
    run_gold(lake)  # must not raise, and must write its report
    from cs2_pipeline.config import GOLD_DIR
    assert (GOLD_DIR / "report.md").exists()
    assert (GOLD_DIR / "first_blood_impact.parquet").exists()


def test_real_demo_end_to_end():
    demo = os.environ.get("CS2_TEST_DEMO")
    if not demo:
        print("  (skipped: set CS2_TEST_DEMO)")
        return
    from cs2_pipeline.bronze import parse_demo_to_bronze
    parse_demo_to_bronze(demo, "real-demo", force=True)
    s = bronze_to_silver("real-demo", force=True)
    assert s["matches"].iloc[0]["n_rounds"] == len(s["rounds"]) > 0
    assert s["kills"]["is_first_kill"].sum() == len(s["rounds"])


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
