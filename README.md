# CS2 Tactical Intelligence Pipeline

> Turning Counter-Strike 2 .dem replays into a **data product**: clean, documented, queryable data and tactical insight that analysts, coaches, players and fans can explore, customise and replay.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![awpy](https://img.shields.io/badge/awpy-3.x-FF4655)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Parquet](https://img.shields.io/badge/Apache_Parquet-50ABF1?logo=apacheparquet&logoColor=white)
![Architecture](https://img.shields.io/badge/architecture-Medallion-C0C0C0)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-WIP-orange)

---

## The problem

The competitive CS2 scene produces **gigabytes of telemetry per match** as `.dem` files. Tick-by-tick binary replays full of Source Engine noise (warm-up, server resets, world-entity deaths). But, raw data doesn't win championships. Coaches and IGLs need *answers*, not logs.

This is an end-to-end **Data Engineering & Analytics pipeline** that parses those replays, enforces data quality, and ships analysis-ready tables and a tactical report - built to scale from Tier 1 down to Tier 3, as long as there's a supply of demos.

## Scope

The goal is a product, not a one-off report. The same lake serves four audiences:

| Audience | What they do with it |
|---|---|
| Analysts and coaches | Tactical metrics, opponent prep, team comparisons |
| Players | Review their own rounds and recurring mistakes |
| Fans and community | Explore matches and stats |
| Builders | Query the Parquet lake directly (SQL, DuckDB) |

The flagship use case: pick *all first kills by a team on a map in a tournament*, jump to each one on a **2D replay** synced with player positions, utility and round stats. A replay is rebuilt from tick data (a `.dem` contains game state, not video).

Status: all four layers are in place — Bronze, Silver, Gold and a serving layer feeding a round explorer page. A `.dem` dropped into `data/raw/` is the only input the pipeline needs; the only thing it cannot get from the file is an event's real name and date, because a CS2 demo carries neither (see **Match catalog**).

## The headline question it answers

> **Does winning the opening duel actually win you the round — and by how much?**

The pipeline measures the real-world impact of **First Blood** on round wins, broken down by side (CT/T) and by opening weapon, with honest statistics instead of hand-wavy claims.

```text
## Key Finding: First Blood Impact

When your team secures the first kill of the round, they win it 63.3% of the time
(95% CI: 45.5%–78.1%) — a +13.3% lift over the 50% baseline.

| Scenario               | Rounds | Win rate w/ First Blood | Side baseline | Lift   |
|------------------------|-------:|------------------------:|--------------:|-------:|
| CT secures First Blood |     16 |                   68.8% |         48.0% | +20.8% |
| T secures First Blood  |     14 |                   57.1% |         52.0% |  +5.1% |
```
<sub>Example report. Confidence intervals are wide on a single match by design - they tighten as the Silver layer accumulates more demos.</sub>

---

## Architecture — Medallion (Bronze → Silver → Gold)

| Layer | Input | Output | What happens |
|---|---|---|---|
| 🥉 **Bronze** | `.dem` | raw Parquet (13 tables) | `awpy` 3 parses the binary into raw tables: rounds, kills, damages, bomb, smokes, grenades, shots, blinds, item events, economy, roster and player state sampled at `CS2_SNAPSHOT_HZ` (default 4 Hz). Engine noise is preserved as-is. A manifest records the demo hash and parser settings. |
| 🥈 **Silver** | Bronze | typed Parquet (13 tables) | **Data contracts** (`config.SILVER_SCHEMAS`): `CT`/`T` sides, `round_num` from tick ranges, a per-round clock (`t_s`), teams on every row, world and team kills removed. Kills and damage after a round was decided are kept but flagged, so round analysis ignores them while player totals still reconcile with public stat pages. 44 quality checks gate every match; one that fails is quarantined instead of feeding an insight. |
| 🥇 **Gold** | Silver lake | facts + metrics + report | `fact_round`, `fact_kill` (with map zones), `fact_player`; 10 team metrics across team × map × side; a player leaderboard; and `insights.md`. Every rate carries its sample size, a Wilson interval, a confidence label and a Benjamini–Hochberg q-value. |
| 📡 **Serving** | Gold | static JSON | `index.json` (filterable without fetching a round), `maps/<map>.json`, `<match>/r<N>.json` — ~39 KB a round, which `web/index.html` replays in the browser with no backend. |

The Gold layer reads the **entire Silver lake**, not a single match - that's where statistical power comes from. Add demos, and the signal sharpens automatically.

## Tech stack

`Python 3.11+` · `awpy 3.x` (demo parsing, Rust backend) · `polars` (Bronze I/O) · `pandas` · `Apache Parquet / PyArrow` (columnar storage) · static JSON + vanilla JS for the browser · `DuckDB-ready` query layer. No statistics package: the Wilson interval, the exact binomial test, Fisher's exact test and Benjamini–Hochberg are implemented in `stats.py` and checked against known values.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Drop one or more .dem files into data/raw/
# 2. Run the full pipeline (raw → bronze → silver → gold)
python run_pipeline.py --all

# 3. Read the insights, or open the round explorer
cat data/gold/insights.md
python -m http.server -d data/serving 8000 &   # then open web/index.html?data=http://localhost:8000
```

`--all` runs every layer: parse → contracts → quality → facts → metrics → serving export. It is idempotent, so re-running it after adding a demo only parses the new one.

```bash
python run_pipeline.py --check          # quality report for the whole lake
python run_pipeline.py --gold-only      # recompute Gold from existing Silver
python run_pipeline.py --serve-only     # re-export only the browser JSON
python -m cs2_pipeline.catalog list     # what each match knows about its event
```

No demo handy? Validate the install and the analytics logic with the synthetic tests - they need no `.dem`:

```bash
python -m tests
# optional: also run end to end on a real demo
CS2_TEST_DEMO=data/raw/match.dem python -m tests
```

The player-state sampling rate is configurable: `CS2_SNAPSHOT_HZ=8 python run_pipeline.py --all --force`.

## Project structure

```text
cs2-de/
├── cs2_pipeline/
│   ├── config.py        # paths, Bronze table list, Silver data contracts, logging
│   ├── utils.py         # polars→pandas bridge, defensive schema helpers
│   ├── bronze.py        # .dem → raw Parquet via awpy 3, manifest, atomic writes
│   ├── silver.py        # rounds, clock, teams, noise removal, schema enforcement
│   ├── quality.py       # 44 invariants every valid match must satisfy
│   ├── catalog.py       # event / stage / date per match (+ CLI)
│   ├── zones.py         # map zones and outline, derived from the demos
│   ├── stats.py         # Wilson CI, binomial & Fisher exact tests, Benjamini-Hochberg
│   ├── facts.py         # fact_round, fact_kill, fact_player
│   ├── insights.py      # team metrics, baselines, findings, leaderboard
│   ├── gold.py          # Gold orchestration + reports
│   └── serving.py       # static JSON export for the browser
├── web/index.html       # round explorer: filter rounds, replay them in 2D
├── run_pipeline.py      # CLI orchestrator, single- and batch-mode
├── tests/               # 113 synthetic tests (no real demo needed)
├── requirements.txt
└── data/                # the lake (git-ignored): raw / bronze / silver / gold / serving / zones
```

## Match catalog

A `.dem` knows its map, server and players, but not which event it belongs to or when it was played — there is no timestamp in the file, and a file's mtime is when it was downloaded. So that context lives in `data/catalog.json`:

```bash
python -m cs2_pipeline.catalog list
python -m cs2_pipeline.catalog set spirit-vs-furia-m1-ancient \
    --event "BLAST Rivals Fall 2026" --stage "Grand Final" --played-on 2026-11-15 --best-of 5
```

Series and map order are read off the file name when it follows `[<event>-]<a>-vs-<b>-m<N>-<map>`, using the team names the demo itself reports — so `natus-vincere-vs-furia-m1-nuke` is one team called Natus Vincere, not an event called "natus". `meta_source` on every match says whether its context came from the catalog, from the file name, or from nowhere.

## Map zones

Official callouts live in the map's navigation and radar files, which cannot be downloaded here, and putting guessed names on real coordinates would be worse than having none. So the zones are derived from what the demos do carry: **A site** and **B site** from the median of every recorded bomb plant, **T spawn** and **CT spawn** from each side's position at the first live sample, **Mid** from the ground far from all four. The map outline is a 96×96 grid of where players have actually walked.

They are coarse areas, named as such — `A site`, never `Jungle`. The check that matters: every bomb plant in the three reference demos is labelled with its own site, with no mistakes.

## Why this is more than a notebook

- **Idempotent & cached.** Each `.dem` is parsed once (the expensive step); re-runs skip layers that already exist. Bronze keys the cache on the demo's SHA-256 and the parser settings, so a changed file or setting re-parses. Re-ingesting 100 demos doesn't re-parse the 99 already done. Writes are atomic: an interrupted run never leaves a half-written match.
- **Data contracts.** Every layer has a typed schema; columns are coerced and validated. A malformed or unusual demo degrades gracefully (logged) instead of crashing.
- **Batch-resilient.** One corrupt demo is logged and skipped - it never takes down the whole run.
- **Statistically honest, and measured.** Every rate carries its sample size, a Wilson interval and a confidence label, and nothing is called a *finding* unless its interval clears its baseline **and** it survives a Benjamini–Hochberg correction across every rate tested in that run. That bar is not a guess: a permutation test shuffles the rounds until no real effect remains and counts what the engine still reports. Before the correction, 46% of such runs produced a finding; after it, 3%. The test is in the suite, so the number cannot quietly regress.
- **Validated against the public record.** On the three reference demos the pipeline reproduces the published result exactly — 13-11 on Ancient, 13-5 on Dust2, 13-8 on Mirage, the 6-6/7-5 and 7-5/6-0 half splits, and KSCERATO's 57-31 over the series.
- **Lake-portable.** The data location is a single environment variable (`CS2_DATA_DIR`) - local disk, Google Drive, or a mounted S3/GCS bucket, with zero code change. The Parquet lake is queryable directly with DuckDB.

## Roadmap

- [x] **Bronze** — automated `.dem` parsing with caching
- [x] **Silver** — data contracts, engine-noise filtering, typed Parquet
- [x] **Gold** — First Blood → round-win impact (with 95% CI)
- [x] Expand Bronze: damage, bomb, smokes, grenades, shots, economy and tick-level player state (awpy 3)
- [x] Silver for the remaining Bronze tables (grenades, shots, blinds, item events)
- [x] Match catalog: event, stage, teams and date per match (a `.dem` does not carry them)
- [x] Gold metrics: opening duels, trades, post-plant and retake win rate, clutches, buy types
- [x] Map zones and a map outline derived from the demos
- [x] Multiple-testing correction, calibrated against a permutation test
- [x] Serving layer: per-round JSON sized for the browser
- [x] Round explorer page: filter rounds and replay them in 2D synced with positions and stats
- [ ] Deploy the page and the lake to Vercel
- [ ] DuckDB query layer over the Parquet lake
- [ ] Clustering of positions and tactics
- [ ] Multi-match aggregation dashboard

## License

MIT — see [LICENSE](LICENSE).

---

> **Work in progress.** If you have trouble understanding the premise or replicating the project, feel free to reach out - happy to walk through it.
