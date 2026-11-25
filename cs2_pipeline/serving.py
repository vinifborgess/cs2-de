"""Serving layer: the files a browser needs to replay a round, with no backend.

The product is a page an analyst opens, filters ("every opening kill FURIA won on Mirage"),
and watches. That page cannot query Parquet, so Gold is exported as static JSON that a static
host (Vercel) can serve and the browser can fetch one round at a time:

``index.json``            every match and every round, with the facts a filter needs
                          (winner, opening duel and its zone, plant, buy types, half).
``maps/<map>.json``       the map's extent and zone anchors, so the page can draw without a
                          radar image.
``<match_id>/r<N>.json``  one round's replay: player tracks, kills, bomb events, smokes,
                          flashes and grenades.

Two things keep the payload small enough to fetch per round. Positions are stored as arrays,
not objects - one row is ``[t_s, x, y, health]``, and the player's identity is written once
above its track. And coordinates are normalised to 0..1 over the map's extent, so the page
needs no map-specific maths; ``maps/<map>.json`` carries the extent for anything that does.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, get_logger
from .zones import MapZones

log = get_logger()

SERVING_DIR = DATA_DIR / "serving"

#: Round-level fields the page filters on, taken straight from fact_round.
ROUND_FILTER_FIELDS = [
    "round_num", "half", "ct_team", "t_team", "winner_team", "winner", "reason", "duration_s",
    "bomb_site", "planted", "plant_t_s", "fb_team", "fb_player", "fb_victim", "fb_weapon",
    "fb_t_s", "fb_won", "fb_zone", "ct_buy", "t_buy", "n_kills",
]

#: Decimal places kept for normalised coordinates: 1/10000 of a map is well under a step.
_PRECISION = 4


def _clean(value):
    """JSON-safe: NaN and pandas NA become null, numpy scalars become Python ones."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            value = item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 6)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _records(df: pd.DataFrame, fields: list[str]) -> list[dict]:
    present = [f for f in fields if f in df.columns]
    return [{f: _clean(row[f]) for f in present} for _, row in df.iterrows()]


class Normaliser:
    """Maps world coordinates onto 0..1 over a map's extent, y pointing up."""

    def __init__(self, extent: list[float]):
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in extent)
        self.w = max(self.x1 - self.x0, 1e-6)
        self.h = max(self.y1 - self.y0, 1e-6)

    def xy(self, x, y) -> tuple[float | None, float | None]:
        if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
            return None, None
        return (round((float(x) - self.x0) / self.w, _PRECISION),
                round((float(y) - self.y0) / self.h, _PRECISION))

    def frame(self, df: pd.DataFrame, x: str = "x", y: str = "y") -> pd.DataFrame:
        out = df.copy()
        # Cast to float64 BEFORE rounding: rounding a float32 leaves a value whose nearest
        # double is 0.9646999835968018, which is both noise and 12 extra bytes per number.
        for src, dst in ((x, "nx"), (y, "ny")):
            col = pd.to_numeric(out[src], errors="coerce").astype("float64")
            origin, span = (self.x0, self.w) if dst == "nx" else (self.y0, self.h)
            out[dst] = ((col - origin) / span).round(_PRECISION)
        return out


def map_payload(zones: MapZones) -> dict:
    """What the page needs to draw a map: its extent, and the anchors with normalised centres."""
    n = Normaliser(zones.extent)
    anchors = []
    for name in zones.names:
        x, y = zones.anchors[name]
        nx, ny = n.xy(x, y)
        anchors.append({"zone": name, "x": nx, "y": ny, "world_x": x, "world_y": y,
                        "support": zones.support.get(name, 0), "weak": name in zones.weak()})
    return {"map_name": zones.map_name, "extent": zones.extent, "cell_size": zones.cell_size,
            "zones": anchors, "fitted_from": zones.fitted_from,
            "note": "Zones are derived from plants and spawns in the demos, not official callouts."}


def round_payload(round_num: int, tables: dict[str, pd.DataFrame], norm: Normaliser,
                  *, hz: float | None = None) -> dict:
    """One round's replay payload. ``hz`` thins the player tracks; None keeps every sample."""
    pos = tables["positions"]
    pos = pos[pos["round_num"] == round_num].sort_values("tick", kind="stable")
    if hz:
        # Keep at most one sample per player per 1/hz seconds, by bucketing the clock.
        pos = pos.assign(_b=(pos["t_s"] * hz).round()).drop_duplicates(["steamid", "_b"])
    pos = norm.frame(pos)

    players = []
    for sid, grp in pos.groupby("steamid", sort=False):
        head = grp.iloc[0]
        track = [[round(float(t), 2),
                  None if pd.isna(x) else round(float(x), _PRECISION),
                  None if pd.isna(y) else round(float(y), _PRECISION), int(h)]
                 for t, x, y, h in zip(grp["t_s"], grp["nx"], grp["ny"], grp["health"].fillna(0))]
        players.append({"steamid": str(sid), "name": _clean(head.get("name")),
                        "team": _clean(head.get("team")), "side": _clean(head.get("side")),
                        "track": track})

    def events(table: str, fields: list[str], x: str = "x", y: str = "y") -> list[dict]:
        df = tables.get(table)
        if df is None or df.empty:
            return []
        df = df[df["round_num"] == round_num]
        if df.empty:
            return []
        df = norm.frame(df, x, y)
        return _records(df, fields + ["nx", "ny"])

    kills = tables["kills"]
    kills = kills[kills["round_num"] == round_num]
    kill_rows = []
    if not kills.empty:
        k = norm.frame(kills, "victim_x", "victim_y").rename(columns={"nx": "victim_nx", "ny": "victim_ny"})
        k = norm.frame(k, "attacker_x", "attacker_y").rename(columns={"nx": "attacker_nx", "ny": "attacker_ny"})
        kill_rows = _records(k, ["t_s", "kill_order", "is_first_kill", "is_post_round", "attacker_name",
                                 "attacker_team", "attacker_side", "victim_name", "victim_team",
                                 "victim_side", "weapon", "headshot", "is_trade", "attacker_nx",
                                 "attacker_ny", "victim_nx", "victim_ny"])
    return {
        "round_num": int(round_num),
        "players": players,
        "kills": kill_rows,
        "bomb": events("bomb", ["t_s", "event", "bombsite", "name"]),
        "smokes": events("smokes", ["t_start_s", "t_end_s", "thrower_name", "thrower_team"]),
        "blinds": events("blinds", ["t_s", "attacker_name", "victim_name", "duration",
                                    "is_team_flash", "is_self_flash"], "victim_x", "victim_y"),
        "grenades": events("grenades", ["t_start_s", "t_end_s", "grenade_type", "thrower_name",
                                        "thrower_team"], "land_x", "land_y"),
    }


def export(match_ids: list[str] | None = None, *, hz: float | None = 2.0,
           out_dir: Path | None = None) -> dict:
    """Write the whole serving tree and return what was written.

    Reads one match at a time so memory stays flat as the lake grows, and reuses the stored
    map zones rather than refitting, so a page cached by a browser cannot disagree with a
    newer export about where the A site is.
    """
    from .facts import build_fact_round
    from .gold import ensure_map_zones
    from .quality import check_match
    from .silver import list_matches, read_match

    out = Path(out_dir or SERVING_DIR)
    ids = match_ids if match_ids is not None else list_matches()
    zones = ensure_map_zones(ids)
    matches, skipped, n_rounds, n_bytes = [], {}, 0, 0

    for mid in ids:
        try:
            tables = read_match(mid)
            errors = [c for c in check_match(tables) if not c.ok and c.level == "error"]
            if errors:
                skipped[mid] = "; ".join(c.name for c in errors)
                log.warning("serving: skipping '%s' - %s", mid, skipped[mid])
                continue
            meta = tables["matches"].iloc[0]
            map_name = str(meta["map_name"])
            z = zones.get(map_name)
            if z is None:
                skipped[mid] = f"no zones fitted for {map_name}"
                log.warning("serving: skipping '%s' - %s", mid, skipped[mid])
                continue
            norm = Normaliser(z.extent)
            fr = build_fact_round(tables, z)
            match_dir = out / mid
            match_dir.mkdir(parents=True, exist_ok=True)
            for rnd in sorted(fr["round_num"].tolist()):
                payload = round_payload(rnd, tables, norm, hz=hz)
                path = match_dir / f"r{rnd}.json"
                path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
                n_bytes += path.stat().st_size
                n_rounds += 1
            matches.append({
                "match_id": mid, "map_name": map_name,
                **{f: _clean(meta.get(f)) for f in ("event", "stage", "played_on", "series_id",
                                                    "map_order", "best_of", "team_a", "team_b",
                                                    "score_a", "score_b", "tick_rate")},
                "n_rounds": int(len(fr)),
                "rounds": _records(fr, ROUND_FILTER_FIELDS),
            })
        except Exception as exc:  # noqa: BLE001 - isolate the match, keep the batch
            skipped[mid] = f"{type(exc).__name__}: {exc}"
            log.error("serving: skipping '%s' - %s", mid, skipped[mid])

    (out / "maps").mkdir(parents=True, exist_ok=True)
    for name, z in zones.items():
        (out / "maps" / f"{name}.json").write_text(
            json.dumps(map_payload(z), separators=(",", ":")), encoding="utf-8")

    index = {"generated_from": "cs2-de Gold", "sample_hz": hz,
             "maps": sorted(zones), "matches": matches,
             "skipped": skipped,
             "note": "Coordinates are normalised to 0..1 over each map's extent; see maps/<map>.json."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    log.info("serving: %d matches, %d rounds, %.1f MB in %s%s",
             len(matches), n_rounds, n_bytes / 1e6, out,
             f" ({len(skipped)} skipped)" if skipped else "")
    return {"matches": len(matches), "rounds": n_rounds, "bytes": n_bytes,
            "skipped": skipped, "dir": str(out)}
