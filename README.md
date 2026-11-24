# CS2 Tactical Intelligence Pipeline

> Turning Counter-Strike 2 .dem replays into a **data product**: clean, documented, queryable data and tactical insight that analysts, coaches, players and fans can explore, customise and replay.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![awpy](https://img.shields.io/badge/awpy-2.x-FF4655)
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

Status: the data layers (Bronze, Silver, Gold) and a first replay prototype exist; the serving layer and the web app are planned (see the Roadmap).

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
| 🥈 **Silver** | Bronze | typed Parquet (9 tables) | **Data contracts** (`config.SILVER_SCHEMAS`): `CT`/`T` sides, `round_num` assigned from tick ranges, a per-round clock (`t_s`), teams on every row, world/team kills and warm-up/knife/post-round events removed. |
| 🥇 **Gold** | Silver lake | report + Parquet | Fact/dimension cross-referencing → tactical metrics. Flagship: **First Blood → round-win impact** with Wilson confidence intervals. |

The Gold layer reads the **entire Silver lake**, not a single match - that's where statistical power comes from. Add demos, and the signal sharpens automatically.

## Tech stack

`Python 3.10+` · `awpy 3.x` (demo parsing, Rust backend) · `polars` (Bronze I/O) · `pandas` · `Apache Parquet / PyArrow` (columnar storage) · `DuckDB-ready` query layer

## Quickstart

```bash
pip install -r requirements.txt

# 1. Drop one or more .dem files into data/raw/
# 2. Run the full pipeline (raw → bronze → silver → gold)
python run_pipeline.py --all

# 3. Read the tactical report
cat data/gold/report.md
```

No demo handy? Validate the install and the analytics logic with the synthetic tests - they need no `.dem`:

```bash
python -m tests.test_pipeline
# optional: also run end to end on a real demo
CS2_TEST_DEMO=data/raw/match.dem python -m tests.test_pipeline
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
│   └── gold.py          # First Blood analysis + report generation
├── run_pipeline.py      # CLI orchestrator, single- and batch-mode
├── tests/test_pipeline.py  # synthetic tests (no real demo needed)
├── requirements.txt
└── data/                # the lake (git-ignored): raw / bronze / silver / gold
```

## Why this is more than a notebook

- **Idempotent & cached.** Each `.dem` is parsed once (the expensive step); re-runs skip layers that already exist. Bronze keys the cache on the demo's SHA-256 and the parser settings, so a changed file or setting re-parses. Re-ingesting 100 demos doesn't re-parse the 99 already done. Writes are atomic: an interrupted run never leaves a half-written match.
- **Data contracts.** Every layer has a typed schema; columns are coerced and validated. A malformed or unusual demo degrades gracefully (logged) instead of crashing.
- **Batch-resilient.** One corrupt demo is logged and skipped - it never takes down the whole run.
- **Statistically honest.** Gold reports Wilson 95% confidence intervals and lift over baseline. Small samples are flagged, not disguised as certainty.
- **Lake-portable.** The data location is a single environment variable (`CS2_DATA_DIR`) - local disk, Google Drive, or a mounted S3/GCS bucket, with zero code change. The Parquet lake is queryable directly with DuckDB.

## Roadmap

- [x] **Bronze** — automated `.dem` parsing with caching
- [x] **Silver** — data contracts, engine-noise filtering, typed Parquet
- [x] **Gold** — First Blood → round-win impact (with 95% CI)
- [x] Expand Bronze: damage, bomb, smokes, grenades, shots, economy and tick-level player state (awpy 3)
- [ ] Silver for the remaining Bronze tables (grenades, shots, blinds, item events)
- [ ] Match catalog: event, stage, teams and date per match (a `.dem` does not carry them)
- [ ] Gold metrics: trade-kill conversion, post-plant win rate, clutch success
- [ ] DuckDB query layer over the Parquet lake
- [ ] Serving layer: per-round Parquet/JSON sized for the browser
- [ ] Web app (Vercel): filter by team, map, event and side; list moments (e.g. first kills) and replay them in 2D synced with positions and stats
- [ ] Clustering of positions and tactics
- [ ] Multi-match aggregation dashboard

## License

MIT — see [LICENSE](LICENSE).

---

> **Work in progress.** If you have trouble understanding the premise or replicating the project, feel free to reach out - happy to walk through it.
