"""
GOLD LAYER until now
What is the REAL impact of getting First Blood on winning the round?

Outputs (data/gold/):
- round_outcomes.parquet: 1 line per round (winner, FB side, converted?)
- first_blood_impact.parquet: aggregates (overall + by CT/T side) with lift and 95% CI
- first_blood_by_weapon.parquet: FB conversion by weapon
- report.md: human reading with tactical recommendation
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import GOLD_DIR, get_logger
from .facts import build_fact_player, build_fact_round
from .quality import check_match, summarize
from .insights import compute_team_metrics, findings, player_leaderboard
from .silver import list_matches, read_match
from .stats import wilson_ci

log = get_logger()


def compute_round_outcomes(kills: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Join the first kill of each round with the round outcome.

    Uses round_num as the key (awpy already assigns round_num to both kills and rounds),
    which is more robust than matching by tick ranges.
    """
    if kills.empty or rounds.empty:
        return pd.DataFrame()

    needed_k = {"match_id", "round_num", "tick", "attacker_side"}
    needed_r = {"match_id", "round_num", "winner"}
    if not needed_k.issubset(kills.columns) or not needed_r.issubset(rounds.columns):
        log.warning("Insufficient columns for Gold generation. kills=%s rounds=%s",
                    list(kills.columns), list(rounds.columns))
        return pd.DataFrame()

    k = kills.sort_values(["match_id", "round_num", "tick"])
    first = k.groupby(["match_id", "round_num"], as_index=False).first()
    keep = ["match_id", "round_num", "attacker_side", "attacker_name", "tick"]
    keep = [c for c in keep if c in first.columns]
    first = first[keep].rename(columns={
        "attacker_side": "first_blood_side",
        "attacker_name": "first_blood_player",
        "tick": "first_blood_tick",
    })
    if "weapon" in k.columns:
        fb_weapon = (k.groupby(["match_id", "round_num"], as_index=False)
                        .first()[["match_id", "round_num", "weapon"]]
                        .rename(columns={"weapon": "first_blood_weapon"}))
        first = first.merge(fb_weapon, on=["match_id", "round_num"], how="left")

    out = rounds[["match_id", "round_num", "winner"]].merge(
        first, on=["match_id", "round_num"], how="inner"
    )
    # Normalize side encoding: awpy may return "ct"/"t" (lowercase) or stray
    # whitespace depending on version. Canonicalize to "CT"/"T" so the
    # per-side breakdown and comparisons are robust.
    for col in ("first_blood_side", "winner"):
        out[col] = out[col].astype("string").str.strip().str.upper()
    out["first_blood_won"] = out["first_blood_side"] == out["winner"]
    return out

_wilson_ci = wilson_ci  # kept for backwards compatibility


def compute_first_blood_impact(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Win rate conditioned on First Blood, overall and per side, with lift vs baseline."""
    if outcomes.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    n_rounds = len(outcomes)

    fb_wins = int(outcomes["first_blood_won"].sum())
    lo, hi = _wilson_ci(fb_wins, n_rounds)
    rows.append({
        "scope": "OVERALL",
        "n_rounds": n_rounds,
        "fb_win_rate": fb_wins / n_rounds,
        "ci95_low": lo,
        "ci95_high": hi,
        "baseline_win_rate": 0.50,
        "lift_vs_baseline": fb_wins / n_rounds - 0.50,
    })

    for side in ("CT", "T"):
        side_base = (outcomes["winner"] == side).mean()
        sub = outcomes[outcomes["first_blood_side"] == side]
        n = len(sub)
        if n == 0:
            continue
        wins = int((sub["winner"] == side).sum())
        lo, hi = _wilson_ci(wins, n)
        rows.append({
            "scope": f"FIRST_BLOOD_{side}",
            "n_rounds": n,
            "fb_win_rate": wins / n,
            "ci95_low": lo,
            "ci95_high": hi,
            "baseline_win_rate": side_base,
            "lift_vs_baseline": wins / n - side_base,
        })

    return pd.DataFrame(rows)


def first_blood_by_weapon(outcomes: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """First Blood conversion rate by weapon (which opening weapons are most effective)."""
    if outcomes.empty or "first_blood_weapon" not in outcomes.columns:
        return pd.DataFrame()
    g = (outcomes.groupby("first_blood_weapon")
                 .agg(n_rounds=("first_blood_won", "size"),
                      fb_win_rate=("first_blood_won", "mean"))
                 .reset_index())
    g = g[g["n_rounds"] >= min_n].sort_values("fb_win_rate", ascending=False)
    return g.reset_index(drop=True)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%" if pd.notna(x) else "n/a"


def build_report(impact: pd.DataFrame, by_weapon: pd.DataFrame, n_matches: int) -> str:
    if impact.empty:
        return "# CS2 Tactical Report\n\nInsufficient Silver data to generate insights.\n"

    overall = impact[impact["scope"] == "OVERALL"].iloc[0]
    n = int(overall["n_rounds"])
    wr = overall["fb_win_rate"]
    small = "  ⚠️ small sample size — collect more demos" if n < 30 else ""

    lines = [
        "# CS2 — Tactical Intelligence (Gold)",
        "",
        f"Base: **{n_matches} match(es)**, **{n} rounds** analyzed.{small}",
        "",
        "## Key Finding: First Blood Impact",
        "",
        f"When your team secures the **first kill of the round**, they win the round "
        f"**{_fmt_pct(wr)}** of the time "
        f"(95% CI: {_fmt_pct(overall['ci95_low'])}–{_fmt_pct(overall['ci95_high'])}).",
        "",
        f"This represents a **{_fmt_pct(overall['lift_vs_baseline'])} lift** over the "
        f"50% baseline. Tactical translation: prioritizing early-round control "
        f"(first contact) has a direct and measurable impact on the score.",
        "",
        "## Breakdown by Side",
        "",
        "| Scenario | Rounds | Win rate with First Blood | Side baseline win rate | Lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in impact[impact["scope"].str.startswith("FIRST_BLOOD")].iterrows():
        side = r["scope"].replace("FIRST_BLOOD_", "")
        lines.append(
            f"| {side} gets First Blood | {int(r['n_rounds'])} | "
            f"{_fmt_pct(r['fb_win_rate'])} | {_fmt_pct(r['baseline_win_rate'])} | "
            f"{_fmt_pct(r['lift_vs_baseline'])} |"
        )

    if not by_weapon.empty:
        lines += ["", "## First Blood Conversion by Weapon", "",
                  "| Weapon | Rounds | Win rate |", "|---|---:|---:|"]
        for _, r in by_weapon.head(8).iterrows():
            lines.append(
                f"| {r['first_blood_weapon']} | {int(r['n_rounds'])} | "
                f"{_fmt_pct(r['fb_win_rate'])} |"
            )

    lines += [
        "",
        "## How to read this (for IGL / Coach)",
        "- *First Blood* here = first effective kill of the round (engine noise already filtered).",
        "- Lift shows how much the opening is worth ABOVE random chance.",
        "- With few matches, the CI is wide: the number indicates a trend, not a verdict.",
        "  Value increases significantly as Silver accumulates dozens of demos.",
        "",
    ]
    return "\n".join(lines)


def run_gold(silver: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    kills, rounds = silver["kills"], silver["rounds"]
    n_matches = rounds["match_id"].nunique() if "match_id" in rounds.columns else 0

    outcomes = compute_round_outcomes(kills, rounds)
    impact = compute_first_blood_impact(outcomes)
    by_weapon = first_blood_by_weapon(outcomes)

    if not outcomes.empty:
        outcomes.to_parquet(GOLD_DIR / "round_outcomes.parquet", index=False)
    if not impact.empty:
        impact.to_parquet(GOLD_DIR / "first_blood_impact.parquet", index=False)
    if not by_weapon.empty:
        by_weapon.to_parquet(GOLD_DIR / "first_blood_by_weapon.parquet", index=False)

    report = build_report(impact, by_weapon, n_matches)
    Path(GOLD_DIR / "report.md").write_text(report, encoding="utf-8")
    log.info("Gold generated: %d rounds, %d matches. Report at %s",
             len(outcomes), n_matches, GOLD_DIR / "report.md")

    return {"outcomes": outcomes, "impact": impact, "by_weapon": by_weapon}


def build_facts(match_ids: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """fact_round and fact_player across matches, read one match at a time.

    A match whose Silver fails the data-quality errors (even if nobody quarantined it) or whose
    facts cannot be built is left out and reported in the third value, so one bad match never
    takes the whole Gold run down or leaks wrong numbers into it.
    """
    rounds, players, skipped = [], [], {}
    for mid in (match_ids if match_ids is not None else list_matches()):
        try:
            t = read_match(mid)
            errors = [c for c in check_match(t) if not c.ok and c.level == "error"]
            if errors:
                skipped[mid] = "; ".join(c.name for c in errors)
                log.warning("Gold: skipping '%s' - failed quality checks: %s", mid, skipped[mid])
                continue
            fr, fp = build_fact_round(t), build_fact_player(t)
        except Exception as exc:  # noqa: BLE001 - isolate the match, keep the batch
            skipped[mid] = f"{type(exc).__name__}: {exc}"
            log.error("Gold: skipping '%s' - %s", mid, skipped[mid])
            continue
        rounds.append(fr)
        players.append(fp)
    return (pd.concat(rounds, ignore_index=True) if rounds else pd.DataFrame(),
            pd.concat(players, ignore_index=True) if players else pd.DataFrame(), skipped)


def build_insights_report(metrics: pd.DataFrame, found: pd.DataFrame, board: pd.DataFrame,
                          n_matches: int, n_rounds: int, top: int = 15,
                          skipped: dict[str, str] | None = None) -> str:
    lines = ["# CS2 — Team & Player Insights (Gold)", "",
             f"Base: **{n_matches} match(es)**, **{n_rounds} rounds**.", ""]
    lines += ["## Findings", ""]
    if found.empty:
        best = int(metrics["n"].max()) if not metrics.empty else 0
        lines += ["No finding clears the evidence bar yet: every rate is still compatible with its baseline.",
                  f"The largest sample behind any rate is {best} events. Add more matches "
                  "(a team needs dozens of rounds per map and side before a difference is believable).", ""]
    else:
        lines += ["A finding means the 95% interval is entirely above/below its baseline *and*",
                  "it survives the multiple-testing correction across every rate in this run.", "",
                  "| Team | Map | Side | Metric | Rate | 95% CI | vs baseline | n | q | Confidence |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
        for _, r in found.head(top).iterrows():
            lines.append(f"| {r['team']} | {r['map_name']} | {r['side']} | {r['metric']} | {_fmt_pct(r['rate'])} | "
                         f"{_fmt_pct(r['ci_low'])}–{_fmt_pct(r['ci_high'])} | {_fmt_pct(r['baseline'])} "
                         f"({r['baseline_source']}) | {int(r['n'])} | {r['q_value']:.3f} | {r['confidence']} |")
        lines.append("")
    if not board.empty:
        lines += ["## Players", "",
                  "| Player | Team | Rounds | ADR | KAST | K-D | K/D | Opening duels won |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in board.head(10).iterrows():
            lines.append(f"| {r['name']} | {r['team']} | {int(r['rounds'])} | {r['adr_total']} | "
                         f"{_fmt_pct(r['kast_pct'])} | {int(r['kills_total'])}-{int(r['deaths_total'])} | "
                         f"{r['kd_total']} | {_fmt_pct(r['opening_win_rate'])} ({r['opening_confidence']}) |")
        lines.append("")
    if skipped:
        lines += ["## Excluded matches", "", *[f"- `{m}`: {why}" for m, why in sorted(skipped.items())], ""]
    tested = int(metrics["p_value"].notna().sum()) if "p_value" in metrics else 0
    lines += ["## How to read this",
              f"- {tested} rates were tested in this run. Benjamini-Hochberg caps the share of false",
              "  findings at 5% across all of them; `q` is each finding's adjusted significance.",
              "- `field` baselines compare a team with teams that never played it, in the same map/side slice;",
              "  without such teams a rate is shown but never called a finding. A fixed baseline is tested",
              "  with an exact binomial test, a field baseline with Fisher's exact test, because a field",
              "  baseline is itself measured and carries its own error.",
              "- Confidence follows the number of events behind a rate: <10 insufficient, <30 low, <100 medium.",
              "- ADR and KAST are weighted by rounds played across matches.",
              "- K-D and ADR include kills and damage after the round was decided, which is how public",
              "  stat pages count; KAST and every rate above cover the live phase only.", ""]
    return "\n".join(lines)


def run_gold_facts(match_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Fact tables, team metrics, findings and the insights report, from the Silver lake."""
    ids = match_ids if match_ids is not None else list_matches()
    fr, fp, skipped = build_facts(ids)
    metrics = compute_team_metrics(fr) if not fr.empty else pd.DataFrame()
    found = findings(metrics) if not metrics.empty else pd.DataFrame()
    board = player_leaderboard(fp)
    for name, df in (("fact_round", fr), ("fact_player", fp), ("team_metrics", metrics),
                     ("findings", found), ("player_leaderboard", board)):
        path = GOLD_DIR / f"{name}.parquet"
        if df.empty:
            path.unlink(missing_ok=True)  # never leave a previous run's numbers next to a new report
        else:
            df.to_parquet(path, index=False)
    report = build_insights_report(metrics, found, board, len(ids) - len(skipped), len(fr), skipped=skipped)
    Path(GOLD_DIR / "insights.md").write_text(report, encoding="utf-8")
    log.info("Gold insights: %d matches (%d excluded), %d rounds, %d metrics, %d findings.",
             len(ids) - len(skipped), len(skipped), len(fr), len(metrics), len(found))
    return {"fact_round": fr, "fact_player": fp, "team_metrics": metrics, "findings": found,
            "leaderboard": board, "skipped": skipped}
