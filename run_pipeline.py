#!/usr/bin/env python3
"""
Orquestrador do pipeline CS2 — ponto de entrada deployável.

Exemplos
--------
# Uma partida (informando o match_id):
python run_pipeline.py --demo data/raw/g2-vs-legacy-m1-inferno.dem --match-id g2-vs-legacy-m1

# Lote: processa TODOS os .dem em data/raw/ (match_id = nome do arquivo):
python run_pipeline.py --all

# Só reprocessar Gold a partir da Silver já existente (rápido):
python run_pipeline.py --gold-only

# Forçar reparsing/relimpeza mesmo se já houver cache:
python run_pipeline.py --all --force

Robustez: no modo lote, uma partida que falhar (demo corrompido, POV demo,
etc.) é logada e PULADA — não derruba o batch inteiro.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cs2_pipeline.config import RAW_DIR, get_logger
from cs2_pipeline.bronze import parse_demo_to_bronze
from cs2_pipeline.silver import bronze_to_silver, load_silver_lake
from cs2_pipeline.gold import run_gold

log = get_logger()


def process_one(demo_path: Path, match_id: str, force: bool) -> bool:
    """raw -> bronze -> silver para uma partida. True se ok, False se pulou."""
    try:
        bronze = parse_demo_to_bronze(demo_path, match_id, force=force)
        bronze_to_silver(bronze, match_id, force=force)
        return True
    except Exception as exc:  # noqa: BLE001 — queremos resiliência no lote
        log.error("Falha em '%s' (%s) — pulando. Motivo: %s",
                  match_id, demo_path.name, exc)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline CS2 (Medallion).")
    ap.add_argument("--demo", type=Path, help="Caminho para um .dem específico.")
    ap.add_argument("--match-id", help="ID da partida (default: nome do arquivo).")
    ap.add_argument("--all", action="store_true", help="Processa todos os .dem em data/raw/.")
    ap.add_argument("--gold-only", action="store_true",
                    help="Recalcula apenas o Gold a partir da Silver existente.")
    ap.add_argument("--force", action="store_true", help="Ignora cache (reparsing/relimpeza).")
    args = ap.parse_args(argv)

    t0 = time.time()

    if args.gold_only:
        silver = load_silver_lake()
        run_gold(silver)
        log.info("Concluído em %.1fs.", time.time() - t0)
        return 0

    # Monta a lista de (demo_path, match_id) a processar.
    jobs: list[tuple[Path, str]] = []
    if args.all:
        demos = sorted(RAW_DIR.glob("*.dem"))
        if not demos:
            log.error("Nenhum .dem em %s", RAW_DIR)
            return 1
        jobs = [(p, p.stem) for p in demos]
    elif args.demo:
        match_id = args.match_id or args.demo.stem
        jobs = [(args.demo, match_id)]
    else:
        ap.print_help()
        return 1

    ok = sum(process_one(p, mid, args.force) for p, mid in jobs)
    log.info("Ingestão: %d/%d partidas OK.", ok, len(jobs))

    if ok == 0:
        log.error("Nenhuma partida ingerida — Gold não será recalculado.")
        return 1

    # Gold sempre roda sobre o lake inteiro (poder estatístico vem da agregação).
    silver = load_silver_lake()
    run_gold(silver)
    log.info("Concluído em %.1fs.", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
