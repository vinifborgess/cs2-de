"""Team metrics and findings on top of ``fact_round``.

Every metric is a binary event per team-round (won / did not win), so each one gets the
same treatment: count, rate, Wilson interval, a baseline to compare against, a confidence
label and a verdict. Nothing is called a finding unless the interval clears its baseline.

Baselines
* ``fixed``  - 0.5. Used where a coin flip is the honest null (a round, a pistol round,
               an opening duel: exactly one of two teams wins each).
* ``field``  - the pooled rate of every OTHER team in the same map/side slice (leave-one-out),
               so a team is compared with the rest of the data, not with itself. With no
               other team in the slice it falls back to 0.5 and says so in ``baseline_source``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .stats import summarize_rate

#: Slices every metric is computed on: (team, map, side) down to the team overall.
DIMENSIONS: list[tuple[str, ...]] = [
    ("team", "map_name", "side"), ("team", "map_name"), ("team", "side"), ("team",),
]

METRIC_COLUMNS = [
    "team", "map_name", "side", "metric", "successes", "n", "n_matches", "rate", "ci_low", "ci_high",
    "baseline", "baseline_source", "lift", "confidence", "verdict",
]


def team_rounds(fact_round: pd.DataFrame) -> pd.DataFrame:
    """Long form of fact_round: one row per team per round, from that team's point of view."""
    parts = []
    for side, team_col, opp_col in (("CT", "ct_team", "t_team"), ("T", "t_team", "ct_team")):
        d = fact_round.copy()
        d["team"], d["opponent"], d["side"] = d[team_col], d[opp_col], side
        d["buy"] = d[f"{side.lower()}_buy"]
        parts.append(d)
    tr = pd.concat(parts, ignore_index=True).dropna(subset=["team"])
    tr["won"] = tr["winner_team"] == tr["team"]
    tr["opening_duel"] = tr["fb_team"].notna()
    tr["got_fb"] = tr["fb_team"] == tr["team"]
    return tr


@dataclass(frozen=True)
class Metric:
    where: Callable[[pd.DataFrame], pd.Series]    # which team-rounds are eligible
    success: Callable[[pd.DataFrame], pd.Series]  # which of those count as a success
    baseline: str                                 # "fixed" | "field"
    description: str


METRICS: dict[str, Metric] = {
    "round_win": Metric(lambda d: d["won"].notna(), lambda d: d["won"], "fixed", "rounds won"),
    "opening_duel_win": Metric(lambda d: d["opening_duel"], lambda d: d["got_fb"], "fixed",
                               "rounds where the team got the first kill, of rounds with a kill"),
    "first_kill_conversion": Metric(lambda d: d["got_fb"], lambda d: d["won"], "field",
                                    "rounds won after getting the first kill"),
    "pistol_win": Metric(lambda d: d["round_num"].isin([1, 13]), lambda d: d["won"], "fixed",
                         "pistol rounds won (rounds 1 and 13)"),
    "plant_rate": Metric(lambda d: d["side"] == "T", lambda d: d["planted"], "field",
                         "T rounds in which the bomb was planted"),
    "post_plant_win": Metric(lambda d: (d["side"] == "T") & d["planted"], lambda d: d["won"], "field",
                             "T rounds won after planting"),
    "retake_win": Metric(lambda d: (d["side"] == "CT") & d["planted"], lambda d: d["won"], "field",
                         "CT rounds won after the bomb was planted"),
    "full_buy_win": Metric(lambda d: d["buy"] == "full", lambda d: d["won"], "field", "rounds won on a full buy"),
    "force_buy_win": Metric(lambda d: d["buy"] == "force", lambda d: d["won"], "field", "rounds won on a force buy"),
    "eco_win": Metric(lambda d: d["buy"] == "eco", lambda d: d["won"], "field", "rounds won on an eco"),
}


def compute_team_metrics(fact_round: pd.DataFrame) -> pd.DataFrame:
    """All metrics for every team across the slices in DIMENSIONS."""
    if fact_round.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    tr = team_rounds(fact_round)
    rows: list[pd.DataFrame] = []
    for name, spec in METRICS.items():
        sub = tr[spec.where(tr).fillna(False).astype(bool)].copy()
        if sub.empty:
            continue
        sub["ok"] = spec.success(sub).fillna(False).astype(int)
        for dims in DIMENSIONS:
            g = (sub.groupby(list(dims))
                 .agg(successes=("ok", "sum"), n=("ok", "size"), n_matches=("match_id", "nunique"))
                 .reset_index())
            keys = [d for d in dims if d != "team"]
            if spec.baseline == "field":
                tot = (g.groupby(keys)[["successes", "n"]].transform("sum") if keys
                       else pd.DataFrame({"successes": g["successes"].sum(), "n": g["n"].sum()}, index=g.index))
                o_s, o_n = tot["successes"] - g["successes"], tot["n"] - g["n"]
                g["baseline"] = (o_s / o_n).where(o_n > 0, 0.5)
                g["baseline_source"] = pd.Series("field", index=g.index).where(o_n > 0, "fallback")
            else:
                g["baseline"], g["baseline_source"] = 0.5, "fixed"
            for col in ("map_name", "side"):
                if col not in g:
                    g[col] = "ALL"
            g["metric"] = name
            rows.append(g)
    if not rows:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    m = pd.concat(rows, ignore_index=True)
    sm = [summarize_rate(int(s), int(n), float(b)) for s, n, b in zip(m["successes"], m["n"], m["baseline"])]
    m["rate"], m["ci_low"], m["ci_high"] = [x.rate for x in sm], [x.ci_low for x in sm], [x.ci_high for x in sm]
    m["lift"], m["confidence"], m["verdict"] = [x.lift for x in sm], [x.confidence for x in sm], [x.verdict for x in sm]
    return m[METRIC_COLUMNS].sort_values(["team", "metric", "map_name", "side"]).reset_index(drop=True)


def findings(metrics: pd.DataFrame, *, specific_first: bool = True) -> pd.DataFrame:
    """Metrics whose interval clears the baseline, strongest first.

    Ranked by the distance between the interval's nearest edge and the baseline (how sure we
    are the effect is at least that big), not by the raw rate: 9/10 ranks below 85/100.
    """
    f = metrics[metrics["verdict"].isin(["above", "below"])].copy()
    if f.empty:
        return f
    f["margin"] = (f["ci_low"] - f["baseline"]).where(f["verdict"] == "above", f["baseline"] - f["ci_high"])
    f["specificity"] = (f["map_name"] != "ALL").astype(int) + (f["side"] != "ALL").astype(int)
    order = ["margin"] if not specific_first else ["margin", "specificity"]
    return f.sort_values(order, ascending=False).reset_index(drop=True)
