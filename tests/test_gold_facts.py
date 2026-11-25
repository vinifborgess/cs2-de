"""run_gold_facts end to end on the lake, plus the report and leaderboard rules."""
from tests.test_pipeline import make_match

import pandas as pd

from cs2_pipeline.config import GOLD_DIR
from cs2_pipeline.gold import build_insights_report, run_gold_facts
from cs2_pipeline.silver import bronze_to_silver, quarantine, release


def test_facts_cover_every_clean_match_and_skip_quarantined_ones():
    ids = ["gf-1", "gf-2", "gf-3", "gf-bad"]
    for i, mid in enumerate(ids):
        make_match(mid, seed=40 + i)
        bronze_to_silver(mid, force=True)
    quarantine("gf-bad", ["test: pretend it failed the quality checks"])
    try:
        out = run_gold_facts()
    finally:
        release("gf-bad")
    mine = {"gf-1", "gf-2", "gf-3"}
    fr, fp = out["fact_round"], out["fact_player"]
    assert mine <= set(fr["match_id"]) and "gf-bad" not in set(fr["match_id"]) | set(fp["match_id"])
    assert fr[fr["match_id"].isin(mine)].groupby("match_id").size().eq(24).all()
    assert fp[fp["match_id"].isin(mine)].groupby("match_id").size().eq(10).all()
    for name in ("fact_round", "fact_player", "team_metrics", "player_leaderboard"):
        assert (GOLD_DIR / f"{name}.parquet").exists(), name
    assert (GOLD_DIR / "insights.md").read_text(encoding="utf-8").startswith("# CS2")
    # The lake may hold other matches (the tests share one); rounds must still add up exactly.
    assert out["leaderboard"]["rounds"].sum() == fp["rounds"].sum()


def test_a_corrupt_match_is_excluded_and_reported_instead_of_crashing_gold():
    from cs2_pipeline.silver import _silver_path
    make_match("gf-ok", seed=50)
    make_match("gf-corrupt", seed=51)
    for mid in ("gf-ok", "gf-corrupt"):
        bronze_to_silver(mid, force=True)
    path = _silver_path("rounds", "gf-corrupt")          # corrupted on disk, never quarantined
    r = pd.read_parquet(path)
    r.loc[1, "round_num"] = r.loc[0, "round_num"]
    r.to_parquet(path, index=False)
    out = run_gold_facts(["gf-ok", "gf-corrupt"])
    assert set(out["fact_round"]["match_id"]) == {"gf-ok"} and set(out["skipped"]) == {"gf-corrupt"}
    assert "rounds.unique_round_num" in out["skipped"]["gf-corrupt"]
    text = (GOLD_DIR / "insights.md").read_text(encoding="utf-8")
    assert "## Excluded matches" in text and "`gf-corrupt`" in text and "**1 match(es)**" in text


def test_run_gold_facts_on_an_empty_selection_does_not_crash():
    out = run_gold_facts([])
    assert out["fact_round"].empty and out["findings"].empty
    assert "0 match" in (GOLD_DIR / "insights.md").read_text(encoding="utf-8")


def _metrics(rate=0.8, n=100, verdict="above"):
    return pd.DataFrame([{"team": "FURIA", "map_name": "de_mirage", "side": "CT", "metric": "pistol_win",
                          "successes": int(rate * n), "n": n, "n_matches": 4, "rate": rate, "ci_low": 0.71, "ci_high": 0.87,
                          "baseline": 0.5, "baseline_source": "fixed", "field_successes": pd.NA,
                          "field_n": pd.NA, "lift": rate - 0.5, "confidence": "high", "verdict": verdict,
                          "p_value": 1e-8, "q_value": 2e-7, "significant": True, "margin": 0.2}])


def test_report_says_so_when_there_is_nothing_to_report_and_lists_findings_when_there_is():
    empty = build_insights_report(_metrics(0.5, 20, "inconclusive"), pd.DataFrame(), pd.DataFrame(), 1, 24)
    assert "No finding clears the evidence bar" in empty and "largest sample behind any rate is 20" in empty
    m = _metrics()
    full = build_insights_report(m, m, pd.DataFrame(), 4, 96)
    assert "No finding clears" not in full
    assert ("| FURIA | de_mirage | CT | pistol_win | 80.0% | 71.0%–87.0% | 50.0% (fixed) | 100 | 0.000 | high |"
            in full)
    assert "1 rates were tested in this run" in full


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
