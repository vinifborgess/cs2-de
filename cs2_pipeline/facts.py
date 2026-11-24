"""Gold fact tables: one row per analytical grain, built from Silver.

``fact_player``  one row per player per match: kills, ADR, KAST, opening duels, trades,
                 multi-kills and clutches.
``fact_round``   one row per round of a match, with teams, sides, economy, opening
                 duel and bomb outcome already joined - the base of every team/map metric.

Facts are built per match (cheap, parallelisable, cacheable) and concatenated by the
callers, so the lake never needs to hold every position of every match in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACT_ROUND_COLUMNS = [
    "match_id", "map_name", "round_num", "half", "ct_team", "t_team", "winner", "winner_team",
    "reason", "duration_s", "bomb_site", "planted", "plant_t_s",
    "fb_team", "fb_side", "fb_player", "fb_victim", "fb_weapon", "fb_t_s", "fb_won",
    "ct_buy", "t_buy", "ct_equip", "t_equip", "n_kills",
]


def half_of(round_num: pd.Series) -> pd.Series:
    """1 = first half, 2 = second half, 3 = overtime (MR12)."""
    return pd.Series(np.select([round_num <= 12, round_num <= 24], [1, 2], default=3),
                     index=round_num.index)


def build_fact_round(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fact table for ONE match from its Silver tables."""
    rounds, kills, bomb, eco, matches = (tables[k] for k in ("rounds", "kills", "bomb", "economy", "matches"))
    f = rounds[["match_id", "round_num", "ct_team", "t_team", "winner", "winner_team",
                "reason", "duration_s", "bomb_site"]].copy()
    f.insert(1, "map_name", matches["map_name"].iloc[0] if len(matches) else pd.NA)
    f.insert(3, "half", half_of(f["round_num"]))

    plants = (bomb[bomb["event"] == "finish_plant"].sort_values("t_s").drop_duplicates("round_num")
              .set_index("round_num")["t_s"])
    f["planted"] = f["round_num"].isin(plants.index)
    f["plant_t_s"] = f["round_num"].map(plants)

    first = (kills.sort_values(["round_num", "kill_order"]).drop_duplicates("round_num")
             .set_index("round_num"))
    for col, src in (("fb_team", "attacker_team"), ("fb_side", "attacker_side"), ("fb_player", "attacker_name"),
                     ("fb_victim", "victim_name"), ("fb_weapon", "weapon"), ("fb_t_s", "t_s")):
        f[col] = f["round_num"].map(first[src])
    f["fb_won"] = (f["fb_team"] == f["winner_team"]).astype("boolean").mask(f["fb_team"].isna())

    for side in ("CT", "T"):
        e = eco[eco["side"] == side].drop_duplicates("round_num").set_index("round_num")
        f[f"{side.lower()}_buy"] = (f["round_num"].map(e["buy_type"]) if not e.empty
                                    else pd.Series(pd.NA, index=f.index)).astype("string")
        f[f"{side.lower()}_equip"] = (f["round_num"].map(e["equipment_value"]) if not e.empty
                                      else pd.Series(np.nan, index=f.index)).astype("Int64")
    f["n_kills"] = f["round_num"].map(kills.groupby("round_num").size()).fillna(0).astype(int)
    return f[FACT_ROUND_COLUMNS].reset_index(drop=True)


FACT_PLAYER_COLUMNS = [
    "match_id", "steamid", "name", "team", "rounds", "kills", "deaths", "assists", "headshots",
    "damage", "adr", "kast_rounds", "kast_pct", "opening_kills", "opening_deaths",
    "trade_kills", "traded_deaths", "k2", "k3", "k4", "k5", "clutch_attempts", "clutch_wins",
]


def _clutches(kills: pd.DataFrame, rounds: pd.DataFrame, roster: dict[str, list[str]]) -> pd.DataFrame:
    """One row per round where a player was left alone against at least one enemy.

    Derived from kill order alone: a team starts each round with its roster, a team that is
    down to one player facing at least one enemy is in a clutch. Only the first player to be
    left alone counts, so a 1v1 reached after a 1v2 is not a second clutch. The round winner
    comes from the rounds table (bomb plants and defuses are not needed to decide it).
    """
    winner = rounds.set_index("round_num")["winner_team"]
    out = []
    for rnd, grp in kills.sort_values(["round_num", "kill_order"]).groupby("round_num"):
        dead: set[str] = set()
        for _, k in grp.iterrows():
            dead.add(k["victim_steamid"])
            team = k["victim_team"]
            if team not in roster:
                continue
            for t in roster:
                mine = [p for p in roster[t] if p not in dead]
                other = [p for t2, ps in roster.items() if t2 != t for p in ps if p not in dead]
                if len(roster[t]) == 5 and len(mine) == 1 and len(other) >= 1:
                    out.append({"round_num": rnd, "steamid": mine[0], "team": t,
                                "vs": len(other), "won": winner.get(rnd) == t})
                    break
            else:
                continue
            break
    return pd.DataFrame(out, columns=["round_num", "steamid", "team", "vs", "won"]).astype({"won": bool})


def build_fact_player(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-player match stats for ONE match from its Silver tables."""
    players, kills, damages, rounds, pos = (tables[k] for k in ("players", "kills", "damages", "rounds", "positions"))
    f = players[["match_id", "steamid", "name", "team"]].copy()
    n_rounds = len(rounds)
    seen = pos.groupby("steamid")["round_num"].nunique() if not pos.empty else pd.Series(dtype=int)
    f["rounds"] = f["steamid"].map(seen).fillna(n_rounds).astype(int)

    by_att, by_vic = kills.groupby("attacker_steamid"), kills.groupby("victim_steamid")
    f["kills"] = f["steamid"].map(by_att.size()).fillna(0).astype(int)
    f["deaths"] = f["steamid"].map(by_vic.size()).fillna(0).astype(int)
    f["headshots"] = f["steamid"].map(kills[kills["headshot"].fillna(False).astype(bool)]
                                      .groupby("attacker_steamid").size()).fillna(0).astype(int)
    # Assists are recorded by name; a name shared by two players of the match is ambiguous and skipped.
    uniq = players.drop_duplicates("name", keep=False).set_index("name")["steamid"]
    assist_ids = kills["assister_name"].map(uniq).dropna()
    f["assists"] = f["steamid"].map(assist_ids.value_counts()).fillna(0).astype(int)

    dmg = damages[~damages["is_team_damage"].fillna(False).astype(bool)]
    f["damage"] = f["steamid"].map(dmg.groupby("attacker_steamid")["dmg_health_real"].sum()).fillna(0.0)
    f["adr"] = (f["damage"] / f["rounds"]).round(2)

    first = kills[kills["is_first_kill"].fillna(False).astype(bool)]
    f["opening_kills"] = f["steamid"].map(first.groupby("attacker_steamid").size()).fillna(0).astype(int)
    f["opening_deaths"] = f["steamid"].map(first.groupby("victim_steamid").size()).fillna(0).astype(int)
    f["trade_kills"] = f["steamid"].map(kills[kills["is_trade"].fillna(False).astype(bool)]
                                        .groupby("attacker_steamid").size()).fillna(0).astype(int)
    f["traded_deaths"] = f["steamid"].map(kills[kills["victim_traded"].fillna(False).astype(bool)]
                                          .groupby("victim_steamid").size()).fillna(0).astype(int)

    per_round = kills.groupby(["attacker_steamid", "round_num"]).size()
    for n in (2, 3, 4, 5):
        counts = per_round[per_round == n].groupby(level=0).size() if n < 5 else per_round[per_round >= 5].groupby(level=0).size()
        f[f"k{n}"] = f["steamid"].map(counts).fillna(0).astype(int)

    # KAST: rounds with a Kill, an Assist, the player Survived, or the death was Traded.
    kast = set(zip(kills["attacker_steamid"], kills["round_num"]))
    a_rows = kills[kills["assister_name"].isin(uniq.index)]
    kast |= set(zip(a_rows["assister_name"].map(uniq), a_rows["round_num"]))
    died = set(zip(kills["victim_steamid"], kills["round_num"]))
    traded = kills[kills["victim_traded"].fillna(False).astype(bool)]
    kast |= set(zip(traded["victim_steamid"], traded["round_num"]))
    round_ids = set(rounds["round_num"])
    counts = {}
    for sid in f["steamid"]:
        survived = {r for r in round_ids if (sid, r) not in died}
        k_rounds = {r for (s, r) in kast if s == sid and r in round_ids}
        counts[sid] = len(survived | k_rounds)
    f["kast_rounds"] = f["steamid"].map(counts).astype(int)
    f["kast_pct"] = (f["kast_rounds"] / f["rounds"]).clip(upper=1.0).round(4)

    roster = players.groupby("team")["steamid"].apply(list).to_dict()
    cl = _clutches(kills, rounds, roster)
    f["clutch_attempts"] = f["steamid"].map(cl.groupby("steamid").size()).fillna(0).astype(int)
    f["clutch_wins"] = f["steamid"].map(cl[cl["won"]].groupby("steamid").size()).fillna(0).astype(int)
    return f[FACT_PLAYER_COLUMNS].reset_index(drop=True)
