"""Team metrics and findings, on hand-built fact_round frames with known answers."""
import math

import pandas as pd

from cs2_pipeline.insights import METRIC_COLUMNS, compute_team_metrics, findings, player_leaderboard, team_rounds


def fact_round(rows):
    """rows: dicts with the fact_round columns the metrics read; the rest default to neutral values."""
    df = pd.DataFrame(rows)
    defaults = {"match_id": "m1", "map_name": "de_x", "planted": False, "fb_team": None,
                "ct_buy": "full", "t_buy": "full"}
    for col, val in defaults.items():
        if col not in df:
            df[col] = val
    return df


def duel_match(winners, *, match_id="m1", map_name="de_x", ct_first="X", t_first="Y", fb=None):
    """One match: X vs Y, X starts CT; `winners` is the list of round winners ('X'/'Y'); halves swap at 12."""
    rows = []
    for i, w in enumerate(winners, start=1):
        ct = ct_first if i <= 12 else t_first
        t = t_first if i <= 12 else ct_first
        rows.append({"match_id": match_id, "map_name": map_name, "round_num": i, "ct_team": ct, "t_team": t,
                     "winner_team": w, "fb_team": (fb[i - 1] if fb else None)})
    return rows


def pick(m, team, metric, map_name="ALL", side="ALL"):
    r = m[(m.team == team) & (m.metric == metric) & (m.map_name == map_name) & (m.side == side)]
    assert len(r) == 1, (team, metric, map_name, side, len(r))
    return r.iloc[0]


def test_dominant_team_is_flagged_and_its_opponent_mirrors_it():
    m = compute_team_metrics(fact_round(duel_match(["X"] * 20 + ["Y"] * 4)))
    x, y = pick(m, "X", "round_win"), pick(m, "Y", "round_win")
    assert (x["successes"], x["n"]) == (20, 24) and math.isclose(x["rate"], 20 / 24)
    assert x["verdict"] == "above" and y["verdict"] == "below" and x["confidence"] == "low"
    assert list(m.columns) == METRIC_COLUMNS


def test_sides_are_attributed_by_who_played_ct_each_half():
    # X wins every round it plays as CT (rounds 1-12) and none as T (13-24).
    m = compute_team_metrics(fact_round(duel_match(["X"] * 12 + ["Y"] * 12)))
    assert pick(m, "X", "round_win", "de_x", "CT")["rate"] == 1.0
    assert pick(m, "X", "round_win", "de_x", "T")["rate"] == 0.0
    assert pick(m, "Y", "round_win", "de_x", "T")["rate"] == 0.0 and pick(m, "Y", "round_win", "de_x", "CT")["rate"] == 1.0
    assert pick(m, "X", "round_win")["rate"] == 0.5          # ALL/ALL slice sees both sides


def test_nine_of_nine_is_not_a_finding():
    m = compute_team_metrics(fact_round(duel_match(["X"] * 9, )))
    r = pick(m, "X", "round_win")
    assert (r["rate"], r["n"], r["confidence"], r["verdict"]) == (1.0, 9, "insufficient", "inconclusive")
    assert findings(m).empty


def test_opening_duels_ignore_rounds_without_kills_and_conversion_is_conditional():
    fb = ["X", "X", "Y", None, "X", None] + [None] * 18
    w = ["X", "Y", "Y", "X", "X", "Y"] + ["X"] * 18
    m = compute_team_metrics(fact_round(duel_match(w, fb=fb)))
    o = pick(m, "X", "opening_duel_win")
    assert (o["successes"], o["n"]) == (3, 4)                 # 4 rounds had a first kill, X got 3
    c = pick(m, "X", "first_kill_conversion")
    assert (c["successes"], c["n"]) == (2, 3)                 # of X's 3 first kills, X won rounds 1 and 5


def test_field_baseline_is_the_pooled_rate_of_the_other_teams():
    # Three teams, all rounds on one side slice. X converts 8/10; Y 5/10; Z 3/10 -> X's baseline is (5+3)/20.
    rows = []
    for team, wins in (("X", 8), ("Y", 5), ("Z", 3)):
        for i in range(10):
            rows.append({"match_id": f"m-{team}", "map_name": "de_x", "round_num": i + 1, "ct_team": team,
                         "t_team": "OTHER", "winner_team": team if i < wins else "OTHER", "fb_team": team})
    m = compute_team_metrics(fact_round(rows))
    r = pick(m, "X", "first_kill_conversion", "de_x", "CT")
    assert (r["successes"], r["n"]) == (8, 10) and math.isclose(r["baseline"], 8 / 20)
    assert r["baseline_source"] == "field"


def test_field_baseline_falls_back_when_there_is_no_other_team():
    rows = [{"match_id": "m", "map_name": "de_x", "round_num": i, "ct_team": "X", "t_team": "OTHER",
             "winner_team": "X", "fb_team": "X"} for i in range(1, 13)]
    m = compute_team_metrics(fact_round(rows))
    r = pick(m, "X", "first_kill_conversion", "de_x", "CT")
    assert r["baseline"] == 0.5 and r["baseline_source"] == "fallback"   # nobody else on CT to compare with


def test_pistol_rounds_are_rounds_1_and_13_only():
    m = compute_team_metrics(fact_round(duel_match(["X", "Y"] * 12)))
    r = pick(m, "X", "pistol_win")
    assert r["n"] == 2 and r["successes"] == 2                # rounds 1 and 13 are both odd -> X wins both


def test_bomb_metrics_split_attackers_and_defenders():
    rows = duel_match(["X"] * 4 + ["Y"] * 4)
    for r in rows[:8]:
        r["planted"] = True
    m = compute_team_metrics(fact_round(rows))
    # X is CT in rounds 1-12: every planted round is a retake for X (won rounds 1-4) ...
    r = pick(m, "X", "retake_win", "de_x", "CT")
    assert (r["successes"], r["n"]) == (4, 8)
    # ... and a post-plant round for Y, which is T then (won rounds 5-8).
    p = pick(m, "Y", "post_plant_win", "de_x", "T")
    assert (p["successes"], p["n"]) == (4, 8)
    pr = pick(m, "Y", "plant_rate", "de_x", "T")
    assert (pr["successes"], pr["n"]) == (8, 8)                        # the match has only these 8 rounds
    assert not ((m.team == "X") & (m.metric == "post_plant_win") & (m.side == "T") & (m.map_name == "de_x")).any()


def test_findings_rank_by_interval_margin_not_raw_rate():
    rows = []
    for team, wins, n in (("A", 9, 10), ("B", 85, 100)):         # A has the higher rate, B the stronger evidence
        for i in range(n):
            rows.append({"match_id": f"m-{team}", "map_name": "de_x", "round_num": i + 1, "ct_team": team,
                         "t_team": "ZZ", "winner_team": team if i < wins else "ZZ", "fb_team": None})
    f = findings(compute_team_metrics(fact_round(rows)))
    fw = f[(f["metric"] == "round_win") & (f["map_name"] == "de_x") & (f["side"] == "CT")]
    assert fw["team"].tolist()[:2] == ["B", "A"] and (fw["verdict"] == "above").sum() == 2


def test_empty_input_gives_an_empty_table_with_the_schema():
    m = compute_team_metrics(fact_round([]).reindex(columns=["match_id"]))
    assert list(m.columns) == METRIC_COLUMNS and m.empty


def test_team_rounds_has_one_row_per_team_per_round():
    tr = team_rounds(fact_round(duel_match(["X"] * 24)))
    assert len(tr) == 48 and tr.groupby("round_num").size().eq(2).all()
    assert tr.groupby("round_num")["won"].sum().eq(1).all()      # exactly one winner per round


def test_leaderboard_weights_adr_and_kast_by_rounds_not_by_match():
    fp = pd.DataFrame([
        {"match_id": "a", "steamid": "1", "name": "p", "team": "T", "rounds": 10, "kills": 10, "deaths": 5, "assists": 0,
         "damage": 1000.0, "kast_rounds": 10, "opening_kills": 2, "opening_deaths": 0, "clutch_attempts": 0, "clutch_wins": 0},
        {"match_id": "b", "steamid": "1", "name": "p", "team": "T", "rounds": 30, "kills": 15, "deaths": 15, "assists": 0,
         "damage": 1500.0, "kast_rounds": 15, "opening_kills": 0, "opening_deaths": 3, "clutch_attempts": 0, "clutch_wins": 0},
    ])
    r = player_leaderboard(fp).iloc[0]
    assert r["rounds"] == 40 and r["n_matches"] == 2
    assert r["adr"] == 62.5 and r["kast_pct"] == 0.625      # a plain mean of per-match values would give 75 and 0.75
    assert r["kd"] == 1.25 and r["opening_win_rate"] == 0.4
    assert player_leaderboard(fp, min_rounds=41).empty


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
