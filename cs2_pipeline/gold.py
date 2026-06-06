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
import math
from pathlib import Path
import pandas as pd
from .config import GOLD_DIR, get_logger

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

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval — more accurate than normal approximation
    for proportions with small sample sizes."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


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