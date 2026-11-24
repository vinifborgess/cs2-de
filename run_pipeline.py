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

# Re-checar a qualidade de TODO o lake (rotina; não reprocessa nada):
python run_pipeline.py --check            # também: --match-id <id>, --strict

Códigos de saída: 0 = tudo certo; 1 = nenhuma partida ingerida, erro de uso ou
checagem de qualidade com erro (--check; com --strict, avisos também contam);
2 = lote concluído, mas alguma partida falhou (o Gold foi gerado com as que deram certo).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cs2_pipeline.config import RAW_DIR, get_logger
from cs2_pipeline.bronze import parse_demo_to_bronze
from cs2_pipeline.silver import bronze_to_silver, load_silver_lake, quarantine, release
from cs2_pipeline.gold import run_gold
from cs2_pipeline.quality import list_matches, run_checks, summarize

log = get_logger()


def process_one(demo_path: Path, match_id: str, force: bool) -> bool:
    """raw -> bronze -> silver para uma partida. True se ok, False se pulou."""
    try:
        parse_demo_to_bronze(demo_path, match_id, force=force)
        bronze_to_silver(match_id, force=force)
        checks = run_checks(match_id)
        errors = [str(c) for c in checks if not c.ok and c.level == "error"]
        if errors:
            quarantine(match_id, errors)
            log.error("'%s' reprovada na checagem de qualidade e posta em quarentena "
                      "(fora do Gold): %s", match_id, "; ".join(errors))
            return False
        if release(match_id):
            log.info("'%s' passou nas checagens: quarentena removida.", match_id)
        for c in checks:
            if not c.ok:
                log.warning("Qualidade[%s]: %s", match_id, c)
        return True
    except Exception as exc:  # noqa: BLE001 — queremos resiliência no lote
        log.error("Falha em '%s' (%s) — pulando. Motivo: %s",
                  match_id, demo_path.name, exc)
        return False


def run_quality(match_ids: list[str], *, strict: bool = False) -> int:
    """Check Silver matches; 0 when clean, 1 on any error (or any warning with strict)."""
    if not match_ids:
        log.error("Nenhuma partida na Silver para checar.")
        return 1
    bad = 0
    for mid in match_ids:
        checks = run_checks(mid)
        errors, warnings = summarize(checks)
        log.info("Qualidade[%s]: %d checagens, %d erro(s), %d aviso(s).", mid, len(checks), errors, warnings)
        for c in checks:
            if not c.ok:
                (log.error if c.level == "error" else log.warning)("  %s", c)
        bad += errors + (warnings if strict else 0)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline CS2 (Medallion).")
    ap.add_argument("--demo", type=Path, help="Caminho para um .dem específico.")
    ap.add_argument("--match-id", help="ID da partida (default: nome do arquivo).")
    ap.add_argument("--all", action="store_true", help="Processa todos os .dem em data/raw/.")
    ap.add_argument("--gold-only", action="store_true",
                    help="Recalcula apenas o Gold a partir da Silver existente.")
    ap.add_argument("--force", action="store_true", help="Ignora cache (reparsing/relimpeza).")
    ap.add_argument("--check", action="store_true",
                    help="Apenas checa a qualidade da Silver (todas as partidas, ou --match-id).")
    ap.add_argument("--strict", action="store_true", help="Com --check: avisos também reprovam.")
    args = ap.parse_args(argv)

    t0 = time.time()

    if args.check:
        return run_quality([args.match_id] if args.match_id else list_matches(), strict=args.strict)

    if args.gold_only:
        silver = load_silver_lake(("kills", "rounds"))  # Gold only needs these
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

    results = {p.name: process_one(p, mid, args.force) for p, mid in jobs}
    ok = sum(results.values())
    failed = [name for name, done in results.items() if not done]
    log.info("Ingestão: %d/%d partidas OK.", ok, len(jobs))
    if failed:
        log.warning("Falharam: %s", ", ".join(failed))

    if ok == 0:
        log.error("Nenhuma partida ingerida — Gold não será recalculado.")
        return 1

    # Gold sempre roda sobre o lake inteiro (poder estatístico vem da agregação).
    silver = load_silver_lake(("kills", "rounds"))  # Gold only needs these
    run_gold(silver)
    log.info("Concluído em %.1fs.", time.time() - t0)
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
