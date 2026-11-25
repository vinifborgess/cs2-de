"""Match catalog: the context a demo file does not carry.

A CS2 demo knows its map, its server and its players, but not which event it belongs to,
which stage it was played at or on what date. Without that, questions like "FURIA on Mirage
at the Major" cannot be asked of the lake, so the context lives in a small JSON file the
user edits (``data/catalog.json``) and Silver copies into the ``matches`` table.

Nothing here invents facts. Fields the catalog does not provide stay null, except the two
that can be read off the match id when it follows the ``<a>-vs-<b>-m<N>-<map>`` convention
(series and map order); those are marked ``derived`` in ``meta_source`` so a reader can tell
them apart from ``catalog`` entries. ``meta_source`` is ``none`` when there is neither.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from .config import DATA_DIR, get_logger

log = get_logger()

CATALOG_PATH = DATA_DIR / "catalog.json"

#: Fields a catalog entry may set. Everything else in the file is kept but ignored here.
CATALOG_FIELDS: tuple[str, ...] = ("event", "stage", "played_on", "series_id", "map_order", "best_of")

#: ``<left>-vs-<right>-m<N>[-<map>]``: the naming HLTV downloads tend to follow. ``left`` may
#: carry the event in front of the team, which only the match's real team names can resolve.
_ID_PATTERN = re.compile(r"^(?P<left>.+?)-vs-(?P<right>.+?)-m(?P<order>\d+)(?:-(?P<map>.+))?$")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def _parse_date(value: Any) -> str | None:
    """ISO date (YYYY-MM-DD) or None. Anything unparseable is reported and dropped."""
    if value in (None, ""):
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        log.warning("catalog: ignoring unparseable date %r (expected YYYY-MM-DD)", value)
        return None


def derive_from_match_id(match_id: str, teams: list[str] | None = None) -> dict[str, Any]:
    """What a ``<a>-vs-<b>-m<N>-<map>`` match id gives away; ``{}`` when it is not one.

    The series and the map order come from the two sides of ``-vs-`` and the ``m<N>``, with the
    teams sorted so both halves of a series share one id.

    An event in front of the team name is only read when the match's own team names are
    passed in, because the file name alone cannot resolve it: ``natus-vincere-vs-furia`` is
    one team called Natus Vincere, not an event called "natus". Given the real names, the
    event is whatever is left once a name is removed, kept verbatim as the file spelled it -
    turning ``blast-rivals-fall-2026`` into a display name would mean guessing where the
    capitals go, and the catalog is where the real name belongs.

    The date is never here: a CS2 demo carries no timestamp and a file's own mtime is when it
    was downloaded, so ``played_on`` comes from the catalog or stays empty.
    """
    m = _ID_PATTERN.match(match_id)
    if not m:
        return {}
    left, right = m.group("left").lower(), m.group("right").lower()
    out: dict[str, Any] = {"map_order": int(m.group("order"))}
    event = None
    if teams:
        known = sorted({_slug(t) for t in teams if str(t).strip()}, key=len, reverse=True)
        # The event is whatever sits before a name the demo itself reports.
        for slug in known:
            if left == slug:
                break
            if left.endswith(f"-{slug}"):
                left, event = slug, left[: -len(slug) - 1]
                break
    pair = sorted((left, right))
    out["series_id"] = f"{pair[0]}-vs-{pair[1]}"
    if event:
        out["event"] = event
        out["series_id"] = f"{event}-{out['series_id']}"
    return out


def load_catalog() -> dict[str, dict]:
    """``{match_id: entry}`` from the catalog file; an empty dict when there is none."""
    if not CATALOG_PATH.exists():
        return {}
    try:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("catalog: %s is not valid JSON (%s); treating it as empty", CATALOG_PATH, exc)
        return {}
    entries = raw.get("matches", raw) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        log.error("catalog: expected an object of match_id -> fields in %s", CATALOG_PATH)
        return {}
    return {str(k): v for k, v in entries.items() if isinstance(v, dict)}


def entry_for(match_id: str, catalog: dict[str, dict] | None = None,
              teams: list[str] | None = None) -> dict[str, Any]:
    """Catalog fields for one match, with the derivable ones filled in and the source named.

    ``teams`` are the match's own clan names, which is what lets an event in the file name be
    told apart from a team whose name has a hyphen in it.
    """
    cat = load_catalog() if catalog is None else catalog
    given = {k: v for k, v in (cat.get(match_id) or {}).items() if k in CATALOG_FIELDS and v not in (None, "")}
    derived = {k: v for k, v in derive_from_match_id(match_id, teams).items() if k not in given}
    out: dict[str, Any] = {f: None for f in CATALOG_FIELDS}
    out.update(derived)
    out.update(given)
    out["played_on"] = _parse_date(out["played_on"])
    out["map_order"] = int(out["map_order"]) if str(out["map_order"] or "").isdigit() else None
    out["best_of"] = int(out["best_of"]) if str(out["best_of"] or "").isdigit() else None
    out["meta_source"] = "catalog" if given else ("derived" if derived else "none")
    return out


def upsert(match_id: str, **fields: Any) -> dict[str, Any]:
    """Write fields for one match into the catalog file, keeping everything else."""
    unknown = set(fields) - set(CATALOG_FIELDS)
    if unknown:
        raise ValueError(f"unknown catalog field(s): {sorted(unknown)}; allowed: {list(CATALOG_FIELDS)}")
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else {}
    if not isinstance(raw, dict) or "matches" not in raw:
        raw = {"matches": raw if isinstance(raw, dict) else {}}
    entry = dict(raw["matches"].get(match_id) or {})
    entry.update({k: v for k, v in fields.items() if v is not None})
    raw["matches"][match_id] = entry
    CATALOG_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    log.info("catalog: %s -> %s", match_id, entry)
    return entry


def incomplete(match_ids: list[str]) -> dict[str, list[str]]:
    """``{match_id: [missing fields]}`` for matches without event context, worth telling the user."""
    cat = load_catalog()
    out = {}
    for mid in match_ids:
        e = entry_for(mid, cat)
        # An event slug read off the file name is enough to group by; a date never is derivable.
        gaps = [f for f in ("event", "played_on")
                if e.get(f) in (None, "") or (f == "event" and e["meta_source"] == "derived")]
        if gaps:
            out[mid] = gaps
    return out


def _cli(argv: list[str] | None = None) -> int:
    """``python -m cs2_pipeline.catalog list|set`` - inspect or fill the catalog."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m cs2_pipeline.catalog",
                                 description="Inspect or edit the match catalog (event context).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show every match in the Silver lake and its event context.")
    s = sub.add_parser("set", help="Set event context for one match.")
    s.add_argument("match_id")
    s.add_argument("--event"), s.add_argument("--stage")
    s.add_argument("--played-on", help="Date of the match, YYYY-MM-DD.")
    s.add_argument("--series-id"), s.add_argument("--map-order", type=int), s.add_argument("--best-of", type=int)
    args = ap.parse_args(argv)

    from .silver import list_matches

    if args.cmd == "set":
        upsert(args.match_id, event=args.event, stage=args.stage, played_on=args.played_on,
               series_id=args.series_id, map_order=args.map_order, best_of=args.best_of)
        print(f"{args.match_id}: {entry_for(args.match_id)}")
        return 0

    ids = list_matches(include_quarantined=True)
    if not ids:
        print("No matches in the Silver lake yet.")
        return 0
    for mid in ids:
        e = entry_for(mid)
        print(f"{mid}\n  event={e['event']!r} stage={e['stage']!r} played_on={e['played_on']!r} "
              f"series={e['series_id']!r} map_order={e['map_order']!r} source={e['meta_source']}")
    gaps = incomplete(ids)
    if gaps:
        print(f"\nWithout event context ({len(gaps)}): fill with "
              f"`python -m cs2_pipeline.catalog set <match_id> --event ... --played-on YYYY-MM-DD`")
        for mid, fields in gaps.items():
            print(f"  {mid}: missing {', '.join(fields)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
