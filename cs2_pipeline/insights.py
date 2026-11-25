"""Team metrics and findings on top of ``fact_round``.

Every metric is a binary event per team-round (won / did not win), so each one gets the
same treatment: count, rate, Wilson interval, a baseline to compare against, a confidence
label and a verdict. Nothing is called a finding unless the interval clears its baseline.

Baselines
* ``fixed``  - 0.5. Used where a coin flip is the honest null (a round, a pistol round,
               an opening duel: exactly one of two teams wins each).
* ``field``  - the pooled rate of the teams that are neither this team nor one of its opponents,
               in the same map/side slice. An opponent's rows are the same rounds seen from the
               other side (its win rate is 1 minus ours), so comparing with them would flag any
               team that simply won more rounds. With no such team the baseline is ``none``: the
               rate is still reported but its verdict is always "inconclusive".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .stats import benjamini_hochberg, binomial_p, fisher_exact_p, summarize_rate

#: Slices every metric is computed on: (team, map, side) down to the team overall.
DIMENSIONS: list[tuple[str, ...]] = [
    ("team", "map_name", "side"), ("team", "map_name"), ("team", "side"), ("team",),
]

METRIC_COLUMNS = [
    "team", "map_name", "side", "metric", "successes", "n", "n_matches", "rate", "ci_low", "ci_high",
    "baseline", "baseline_source", "field_successes", "field_n", "lift", "confidence", "verdict",
    "p_value", "q_value", "significant",
]

#: Share of reported findings allowed to be false, controlled across one run by BH.
FDR_ALPHA = 0.05


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


def _p_value(row: pd.Series) -> float:
    """Significance of one rate against its baseline, by the test the baseline calls for."""
    s, n = int(row["successes"]), int(row["n"])
    if row["baseline_source"] == "fixed":
        return binomial_p(s, n, float(row["baseline"]))
    # A field baseline was measured, not known, so the two rates are compared as two
    # proportions. Treating it as a constant made its p-values far too small.
    fs, fn = row["field_successes"], row["field_n"]
    if pd.isna(fs) or pd.isna(fn):
        return float("nan")
    return fisher_exact_p(s, n - s, int(fs), int(fn) - int(fs))


def compute_team_metrics(fact_round: pd.DataFrame, *, alpha: float = FDR_ALPHA) -> pd.DataFrame:
    """All metrics for every team across the slices in DIMENSIONS.

    One run tests dozens of team/map/side combinations at once, so each rate also carries an
    exact two-sided binomial p-value against its baseline and a Benjamini-Hochberg q-value
    over the whole run. ``significant`` is the q-value clearing ``alpha``; without that
    correction a run on data with no real effect reported a finding about half the time.
    """
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
                # Rows where the opponent is this team: the same rounds from the other side.
                vs = (sub.groupby(["opponent"] + keys).agg(vs_s=("ok", "sum"), vs_n=("ok", "size"))
                      .reset_index().rename(columns={"opponent": "team"}))
                g = g.merge(vs, on=["team"] + keys, how="left").fillna({"vs_s": 0, "vs_n": 0})
                o_s = tot["successes"].to_numpy() - g["successes"] - g["vs_s"]
                o_n = tot["n"].to_numpy() - g["n"] - g["vs_n"]
                g["baseline"] = (o_s / o_n).where(o_n > 0)
                g["baseline_source"] = pd.Series("field", index=g.index).where(o_n > 0, "none")
                # Kept: a field baseline is an estimate, and its own counts are what the
                # significance test needs in order to allow for that.
                g["field_successes"] = o_s.where(o_n > 0).astype("Int64")
                g["field_n"] = o_n.where(o_n > 0).astype("Int64")
                g = g.drop(columns=["vs_s", "vs_n"])
            else:
                g["baseline"], g["baseline_source"] = 0.5, "fixed"
                g["field_successes"] = pd.Series(pd.NA, index=g.index, dtype="Int64")
                g["field_n"] = pd.Series(pd.NA, index=g.index, dtype="Int64")
            for col in ("map_name", "side"):
                if col not in g:
                    g[col] = "ALL"
            g["metric"] = name
            rows.append(g)
    if not rows:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    m = pd.concat(rows, ignore_index=True)
    sm = [summarize_rate(int(s), int(n), 0.5 if pd.isna(b) else float(b))
          for s, n, b in zip(m["successes"], m["n"], m["baseline"])]
    m["rate"], m["ci_low"], m["ci_high"] = [x.rate for x in sm], [x.ci_low for x in sm], [x.ci_high for x in sm]
    m["lift"], m["confidence"], m["verdict"] = [x.lift for x in sm], [x.confidence for x in sm], [x.verdict for x in sm]
    no_ref = m["baseline_source"] == "none"          # nothing to compare with: never a finding
    m.loc[no_ref, ["lift", "verdict"]] = [float("nan"), "inconclusive"]

    # Rows with no baseline, or too small to judge, are not part of the family being tested.
    testable = ~no_ref & (m["confidence"] != "insufficient")
    m["p_value"] = [_p_value(row) if ok else float("nan")
                    for ok, (_, row) in zip(testable, m.iterrows())]
    rejected, q = benjamini_hochberg(m["p_value"].tolist(), alpha)
    m["q_value"], m["significant"] = q, rejected
    return m[METRIC_COLUMNS].sort_values(["team", "metric", "map_name", "side"]).reset_index(drop=True)


def findings(metrics: pd.DataFrame, *, specific_first: bool = True) -> pd.DataFrame:
    """Metrics that survive both the interval and the multiple-testing correction, strongest first.

    Ranked by the distance between the interval's nearest edge and the baseline (how sure we
    are the effect is at least that big), not by the raw rate: 9/10 ranks below 85/100.
    """
    f = metrics[metrics["verdict"].isin(["above", "below"]) & metrics["significant"]].copy()
    if f.empty:
        return f
    f["margin"] = (f["ci_low"] - f["baseline"]).where(f["verdict"] == "above", f["baseline"] - f["ci_high"])
    f["specificity"] = (f["map_name"] != "ALL").astype(int) + (f["side"] != "ALL").astype(int)
    order = ["margin"] if not specific_first else ["margin", "specificity"]
    return f.sort_values(order, ascending=False).reset_index(drop=True)


def player_leaderboard(fact_player: pd.DataFrame, min_rounds: int = 0) -> pd.DataFrame:
    """Players aggregated across matches: ADR and KAST are weighted by rounds, not averaged per match."""
    if fact_player.empty:
        return pd.DataFrame()
    g = (fact_player.groupby(["steamid"], as_index=False)
         .agg(name=("name", "last"), team=("team", "last"), n_matches=("match_id", "nunique"),
              rounds=("rounds", "sum"), kills=("kills", "sum"), deaths=("deaths", "sum"),
              assists=("assists", "sum"), damage=("damage", "sum"), kast_rounds=("kast_rounds", "sum"),
              opening_kills=("opening_kills", "sum"), opening_deaths=("opening_deaths", "sum"),
              clutch_attempts=("clutch_attempts", "sum"), clutch_wins=("clutch_wins", "sum")))
    g = g[g["rounds"] >= min_rounds].copy()
    g["adr"] = (g["damage"] / g["rounds"]).round(1)
    g["kast_pct"] = (g["kast_rounds"] / g["rounds"]).round(3)
    g["kd"] = (g["kills"] / g["deaths"].where(g["deaths"] > 0)).round(2)
    duels = g["opening_kills"] + g["opening_deaths"]
    sm = [summarize_rate(int(k), int(n)) for k, n in zip(g["opening_kills"], duels)]
    g["opening_win_rate"] = [x.rate for x in sm]
    g["opening_confidence"] = [x.confidence for x in sm]
    return g.sort_values("adr", ascending=False).reset_index(drop=True)
