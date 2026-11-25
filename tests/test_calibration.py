"""How often the engine reports a finding on data where there is nothing to find.

Every statistical guarantee in the Gold layer rests on this: the rounds are permuted so no
association between a team and an outcome survives, and the engine is then asked for
findings. Anything it reports is false by construction. The multiple-testing correction is
the only thing standing between the real rate and roughly one false finding per run.
"""
from tests.test_insights import duel_match, fact_round

import numpy as np
import pandas as pd

from cs2_pipeline.insights import compute_team_metrics, findings

#: Four teams over six matches on three maps, so 'field' baselines have teams to compare with.
PAIRS = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
MAPS = ["de_one", "de_two", "de_three"]


def _dataset() -> pd.DataFrame:
    rows = []
    for i, (x, y) in enumerate(PAIRS):
        winners = [x if r % 2 == 0 else y for r in range(24)]        # 12-12, no real edge
        rows += duel_match(winners, match_id=f"m{i}", map_name=MAPS[i % len(MAPS)],
                           ct_first=x, t_first=y, fb=[x if r % 3 else y for r in range(24)])
    d = fact_round(rows)
    d["planted"] = [r % 3 == 0 for r in range(len(d))]
    d["ct_buy"] = ["full" if r % 4 else "eco" for r in range(len(d))]
    d["t_buy"] = ["full" if r % 5 else "force" for r in range(len(d))]
    return d


def _permute(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Re-decide every round at random, keeping sample sizes and pooled rates."""
    d = d.copy()
    ct_wins = rng.random(len(d)) < 0.5
    d["winner_team"] = np.where(ct_wins, d["ct_team"], d["t_team"])
    d["winner"] = np.where(ct_wins, "CT", "T")
    had_fb = d["fb_team"].notna().to_numpy()
    d.loc[had_fb, "fb_team"] = np.where(rng.random(had_fb.sum()) < 0.5,
                                        d.loc[had_fb, "ct_team"], d.loc[had_fb, "t_team"])
    for col in ("planted", "ct_buy", "t_buy"):
        d[col] = rng.permutation(d[col].to_numpy())
    return d


def _false_findings(runs: int, *, corrected: bool) -> list[int]:
    data, rng = _dataset(), np.random.default_rng(20261125)
    out = []
    for _ in range(runs):
        m = compute_team_metrics(_permute(data, rng))
        if not corrected:                       # what the engine did before BH was added
            m = m.assign(significant=True)
        out.append(len(findings(m)))
    return out


RUNS = 40


def test_the_correction_is_what_keeps_false_findings_rare():
    with_bh = _false_findings(RUNS, corrected=True)
    without = _false_findings(RUNS, corrected=False)
    runs_with_any = sum(1 for c in with_bh if c)
    # The guarantee is about the share of runs that report anything at all, since on this
    # data every finding is false. Measured at 1/40 when written; the bound leaves room for
    # the randomness without letting a real regression through.
    assert runs_with_any <= RUNS * 0.15, f"{runs_with_any}/{RUNS} runs reported a false finding"
    assert sum(with_bh) / RUNS <= 0.5, f"mean {sum(with_bh) / RUNS:.2f} false findings per run"
    # And the correction has to be doing the work, not the interval test on its own.
    assert sum(without) > 4 * max(sum(with_bh), 1), (
        f"uncorrected {sum(without)} vs corrected {sum(with_bh)}: the correction is not biting")


def test_a_real_effect_still_survives_the_correction():
    # A wins 20 of 24 on de_home in three separate matches, and plays nowhere else, so the
    # slice is not diluted. Other teams fill the rest of the lake on other maps.
    rows = []
    for i in range(3):
        rows += duel_match(["A"] * 20 + ["B"] * 4, match_id=f"home{i}", map_name="de_home",
                           ct_first="A", t_first="B")
    for i, (x, y) in enumerate(PAIRS[1:]):
        rows += duel_match([x if r % 2 else y for r in range(24)], match_id=f"n{i}",
                           map_name=MAPS[i % len(MAPS)], ct_first=x, t_first=y)
    f = findings(compute_team_metrics(fact_round(rows)))
    assert not f.empty, "a team winning 60 of 72 rounds must survive the correction"
    hit = f[(f["team"] == "A") & (f["metric"] == "round_win") & (f["map_name"] == "de_home")]
    assert not hit.empty and (hit["verdict"] == "above").all()
    assert hit["q_value"].max() < 0.05 and hit["n"].max() == 72
    # Its opponent is the mirror image, and nobody else is reported on these rounds.
    mirror = f[(f["team"] == "B") & (f["metric"] == "round_win")]
    assert not mirror.empty and (mirror["verdict"] == "below").all()


def test_where_the_bar_sits_between_a_lopsided_map_and_a_close_one():
    """A one-map effect is reported when it is large, and not when it is ordinary."""
    def lake(winners):
        rows = duel_match(winners, match_id="once", map_name="de_home", ct_first="A", t_first="B")
        for i, (x, y) in enumerate(PAIRS):
            rows += duel_match([x if r % 2 else y for r in range(24)], match_id=f"n{i}",
                               map_name=MAPS[i % len(MAPS)], ct_first=x, t_first=y)
        return compute_team_metrics(fact_round(rows))

    def row(m, won):
        r = m[(m["team"] == "A") & (m["metric"] == "round_win") & (m["map_name"] == "de_home")
              & (m["side"] == "ALL")].iloc[0]
        assert (r["successes"], r["n"]) == (won, 24)
        return r

    strong = row(lake(["A"] * 20 + ["B"] * 4), 20)     # 83%: reported
    assert strong["significant"] and strong["q_value"] < 0.05
    close = row(lake(["A"] * 14 + ["B"] * 10), 14)     # 58%: one map says nothing
    assert not close["significant"] and close["p_value"] > 0.3
    # The team's own overall rate is diluted by its neutral maps, as it should be.
    everything = lake(["A"] * 20 + ["B"] * 4)
    overall = everything[(everything["team"] == "A") & (everything["metric"] == "round_win")
                         & (everything["map_name"] == "ALL") & (everything["side"] == "ALL")].iloc[0]
    assert not overall["significant"] and overall["n"] == 96


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
