"""Named map zones, derived from the demos themselves.

Official CS2 callouts live in the map's navigation and radar files, which this environment
cannot download, and guessing them would put invented names on real coordinates. So the
zones here are derived from ground truth the demos do carry:

* **A site** / **B site** - the median of every bomb plant recorded at that site. awpy labels
  each plant with its site, so these are the sites themselves, not an estimate of them.
* **T spawn** / **CT spawn** - the median position of each side at the first live sample of a
  round. Sides swap at half-time, but the spawn belongs to the side, so it is stable.
* **Mid** - the median of the positions that are far from all four, which is the ground
  between them.

A position is labelled with the nearest anchor. These are coarse areas, not callouts: this
is "A site", never "Jungle" or "Short A". Alongside the name every point also gets a grid
cell of a fixed size in map units, which is what heat maps should aggregate on.

A map's zones are fitted ONCE from the matches in the lake and stored in
``data/zones/<map>.json``, so a later run cannot silently move an anchor and make two matches
of the same map incomparable. ``fit`` must be asked for explicitly to refit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import DATA_DIR, get_logger

log = get_logger()

ZONES_DIR = DATA_DIR / "zones"

#: Grid cell size in map units. A competitive map spans roughly 4000 units, so 256 gives a
#: ~16x16 grid: fine enough to see a position, coarse enough to aggregate over a few matches.
CELL_SIZE = 256.0

#: How far from every anchor a point must be to count as "between them" when fitting Mid.
_MID_MIN_DIST = 900.0

#: An anchor backed by fewer observations than this is kept but reported as weak.
MIN_SUPPORT = 3

#: Resolution of the walkable-area silhouette: where players have actually been. The official
#: radar image cannot be downloaded here, and this is derived from the demos themselves.
OCCUPANCY_GRID = 96


@dataclass
class MapZones:
    """Anchors for one map, with what they were fitted from."""
    map_name: str
    anchors: dict[str, list[float]]
    support: dict[str, int]
    extent: list[float]                      # x_min, y_min, x_max, y_max
    cell_size: float = CELL_SIZE
    fitted_from: list[str] = field(default_factory=list)
    #: One string per row, bottom row first, '1' where a player has been observed.
    occupancy: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return sorted(self.anchors)

    def weak(self) -> list[str]:
        return sorted(n for n, k in self.support.items() if k < MIN_SUPPORT)

    def label(self, x, y) -> pd.Series:
        """Name of the nearest anchor for each point; NA where the point has no coordinates."""
        xs = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
        ys = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
        out = pd.Series(pd.NA, index=xs.index, dtype="string")
        ok = xs.notna() & ys.notna()
        if not ok.any() or not self.anchors:
            return out
        names = self.names
        pts = np.column_stack([xs[ok].to_numpy(float), ys[ok].to_numpy(float)])
        centres = np.array([self.anchors[n] for n in names], dtype=float)
        d = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        out.loc[ok] = pd.Series([names[i] for i in d.argmin(axis=1)], index=xs[ok].index, dtype="string")
        return out

    def cell(self, x, y) -> pd.Series:
        """Grid cell of each point, as ``x<i>y<j>`` counted from the map's lower-left corner."""
        xs = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
        ys = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
        i = ((xs - self.extent[0]) // self.cell_size)
        j = ((ys - self.extent[1]) // self.cell_size)
        out = "x" + i.astype("Int64").astype("string") + "y" + j.astype("Int64").astype("string")
        return out.where(xs.notna() & ys.notna()).astype("string")


def _path(map_name: str) -> "object":
    return ZONES_DIR / f"{map_name}.json"


def load(map_name: str) -> MapZones | None:
    """Stored zones for a map, or None when it has not been fitted yet."""
    p = _path(map_name)
    if not p.exists():
        return None
    try:
        return MapZones(**json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError) as exc:
        log.error("zones: %s is unreadable (%s); refit the map", p, exc)
        return None


def save(zones: MapZones) -> None:
    ZONES_DIR.mkdir(parents=True, exist_ok=True)
    _path(zones.map_name).write_text(json.dumps(asdict(zones), indent=2, sort_keys=True), encoding="utf-8")


def fit(map_name: str, positions: pd.DataFrame, bomb: pd.DataFrame,
        match_ids: list[str] | None = None) -> MapZones:
    """Fit a map's anchors from its positions and bomb plants. Medians, so outliers do not pull."""
    pos = positions.dropna(subset=["x", "y"])
    if pos.empty:
        raise ValueError(f"cannot fit zones for {map_name}: no positions")
    anchors: dict[str, list[float]] = {}
    support: dict[str, int] = {}

    plants = bomb[bomb["event"] == "finish_plant"].dropna(subset=["x", "y"]) if not bomb.empty else bomb
    for site, grp in (plants.groupby("bombsite") if len(plants) else []):
        if str(site).upper() in ("A", "B"):
            anchors[f"{str(site).upper()} site"] = [float(grp["x"].median()), float(grp["y"].median())]
            support[f"{str(site).upper()} site"] = int(len(grp))

    # Where each side starts: the first live sample a player has in a round.
    first = pos.sort_values("tick", kind="stable").groupby(["match_id", "round_num", "steamid"]).first()
    for side, grp in first.reset_index().groupby("side"):
        if side in ("CT", "T"):
            anchors[f"{side} spawn"] = [float(grp["x"].median()), float(grp["y"].median())]
            support[f"{side} spawn"] = int(grp["round_num"].nunique())

    extent = [float(pos["x"].min()), float(pos["y"].min()), float(pos["x"].max()), float(pos["y"].max())]
    zones = MapZones(map_name=map_name, anchors=anchors, support=support, extent=extent,
                     fitted_from=sorted(match_ids or pos["match_id"].unique().tolist()),
                     occupancy=_occupancy(pos, extent))

    # Mid is whatever is left: the ground far from every anchor so far.
    if anchors:
        centres = np.array(list(anchors.values()), dtype=float)
        pts = pos[["x", "y"]].to_numpy(float)
        d = np.sqrt(((pts[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        far = pts[d > _MID_MIN_DIST]
        if len(far) >= 50:
            zones.anchors["Mid"] = [float(np.median(far[:, 0])), float(np.median(far[:, 1]))]
            zones.support["Mid"] = int(len(far))

    weak = zones.weak()
    log.info("zones[%s]: %s%s", map_name, ", ".join(zones.names),
             f" (weak support: {weak})" if weak else "")
    return zones


def _occupancy(pos: pd.DataFrame, extent: list[float], grid: int = OCCUPANCY_GRID) -> list[str]:
    """Silhouette of the walkable area: a grid marking every cell a player has been seen in."""
    x0, y0, x1, y1 = extent
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    i = np.clip(((pos["x"].to_numpy(float) - x0) / w * grid).astype(int), 0, grid - 1)
    j = np.clip(((pos["y"].to_numpy(float) - y0) / h * grid).astype(int), 0, grid - 1)
    seen = np.zeros((grid, grid), dtype=bool)
    seen[j, i] = True
    return ["".join("1" if v else "0" for v in row) for row in seen]


def ensure(map_name: str, positions: pd.DataFrame, bomb: pd.DataFrame,
           match_ids: list[str] | None = None, *, refit: bool = False) -> MapZones:
    """Stored zones for a map, fitting and storing them the first time (or when asked to refit)."""
    if not refit:
        stored = load(map_name)
        if stored is not None:
            return stored
    zones = fit(map_name, positions, bomb, match_ids)
    save(zones)
    return zones


def as_frame(zones: list[MapZones]) -> pd.DataFrame:
    """The anchors of several maps as one table, for the Gold layer to publish."""
    rows = []
    for z in zones:
        for name in z.names:
            x, y = z.anchors[name]
            rows.append({"map_name": z.map_name, "zone": name, "x": x, "y": y,
                         "support": z.support.get(name, 0), "weak": name in z.weak(),
                         "cell_size": z.cell_size, "fitted_from": ",".join(z.fitted_from)})
    return pd.DataFrame(rows, columns=["map_name", "zone", "x", "y", "support", "weak",
                                       "cell_size", "fitted_from"])
