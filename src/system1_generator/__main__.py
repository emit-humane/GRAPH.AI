"""Entry point: `python -m src.system1_generator` runs the full pipeline.

Order:
    G1 → G2 → G3 → G4 (main population, four exports including warmup).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from .common import CONFIG_PATH, DATA_DIR, GeneratorConfig, ensure_dirs


def run(config: GeneratorConfig, out_dir: Path = DATA_DIR, with_warmup: bool = True) -> None:
    ensure_dirs()
    t0 = time.time()
    print(f"[G1] Building {config.num_accounts:,} accounts ...")
    ctx = g1_account_builder.build_accounts(config)
    g1_account_builder.save_accounts(ctx)
    print(f"     done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[G2] Generating ~{config.total_transactions:,} normal transactions ...")
    normal = g2_normal_generator.generate_normal(config, ctx)
    g2_normal_generator.save_normal(normal)
    print(f"     produced {len(normal):,} rows in {time.time() - t1:.1f}s")

    t2 = time.time()
    print(f"[G3] Injecting laundering scenarios "
          f"(target ~{int(config.fraud_ratio * config.total_transactions):,} suspicious) ...")
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(config, ctx, normal)
    g3_scenario_injector.save_artifacts(combined, suspicious, gt)
    print(f"     injected {len(suspicious):,} suspicious rows in {time.time() - t2:.1f}s")

    t3 = time.time()
    print("[G4] Splitting 90/10, fixing reverse causality, exporting CSVs ...")
    hist, stream = g4_splitter.split_and_export(combined, gt, out_dir=out_dir)
    print(f"     historical={len(hist):,} stream={len(stream):,} in {time.time() - t3:.1f}s")

    if with_warmup:
        t4 = time.time()
        print("[G4-warmup] Generating warmup set with seed+1000 ...")
        warmup = g4_splitter.generate_warmup(config, out_dir=out_dir)
        print(f"     warmup_labeled={len(warmup):,} rows in {time.time() - t4:.1f}s")

    print(f"[done] total {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="System 1 — AML Ecosystem Generator")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    cfg = GeneratorConfig.from_file(Path(args.config))
    run(cfg, Path(args.out_dir), with_warmup=not args.no_warmup)


if __name__ == "__main__":
    main()
