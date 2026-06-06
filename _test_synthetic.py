"""Valida silver+gold com dados sintéticos no schema do awpy 2.x."""
import numpy as np
import pandas as pd

from cs2_pipeline.silver import bronze_to_silver, load_silver_lake
from cs2_pipeline.gold import run_gold

rng = np.random.default_rng(42)


def make_match(match_id, n_rounds=24, fb_win_prob=0.70):
    """Gera kills+rounds cruos. First Blood ganha o round com prob fb_win_prob."""
    rounds, kills = [], []
    tick = 1000
    for rnum in range(1, n_rounds + 1):
        fb_side = rng.choice(["CT", "T"])
        other = "T" if fb_side == "CT" else "CT"
        winner = fb_side if rng.random() < fb_win_prob else other

        start_tick = tick
        # Ruído de engine: round-fantasma ocasional (tick 0) que a Silver deve dropar
        if rng.random() < 0.05:
            rounds.append({"round_num": rnum, "start": 0, "end": 0,
                           "winner": winner, "reason": "ct_killed",
                           "freeze_end": 0, "bomb_site": "not_planted"})
            continue

        # 3 a 6 abates por round; o primeiro (menor tick) define o First Blood
        n_k = rng.integers(3, 7)
        for i in range(n_k):
            atk = fb_side if i == 0 else rng.choice([fb_side, other])
            vic = "T" if atk == "CT" else "CT"
            weapon = rng.choice(["ak47", "m4a1", "awp", "deagle", "ak47"])
            kills.append({
                "round_num": rnum, "tick": start_tick + 100 + i * 50,
                "attacker_name": f"{atk}_player{rng.integers(1,6)}",
                "attacker_side": atk,
                "victim_name": f"{vic}_player{rng.integers(1,6)}",
                "victim_side": vic,
                "weapon": str(weapon), "headshot": bool(rng.random() < 0.4),
            })
        # Ruído: 1 world kill (queda) que a Silver deve remover
        if rng.random() < 0.1:
            kills.append({"round_num": rnum, "tick": start_tick + 90,
                          "attacker_name": None, "attacker_side": None,
                          "victim_name": f"{other}_player1", "victim_side": other,
                          "weapon": "world", "headshot": False})

        rounds.append({"round_num": rnum, "start": start_tick, "end": start_tick + 5000,
                       "winner": winner, "reason": "ct_killed",
                       "freeze_end": start_tick + 300, "bomb_site": "not_planted"})
        tick += 6000
    return {"kills": pd.DataFrame(kills), "rounds": pd.DataFrame(rounds)}


# Duas partidas -> exercita a agregação do lake
for mid, p in [("g2-vs-legacy-m1", 0.72), ("navi-vs-faze-m1", 0.68)]:
    bronze = make_match(mid, n_rounds=24, fb_win_prob=p)
    bronze_to_silver(bronze, mid, force=True)

silver = load_silver_lake()
print(f"\nLake: {len(silver['kills'])} kills, {len(silver['rounds'])} rounds, "
      f"{silver['rounds']['match_id'].nunique()} partidas\n")

res = run_gold(silver)
print("\n=== first_blood_impact ===")
print(res["impact"].to_string(index=False))
print("\n=== report.md ===\n")
print((__import__('pathlib').Path('data/gold/report.md')).read_text())
