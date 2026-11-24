"""Gold fact tables: one row per analytical grain, built from Silver.

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
