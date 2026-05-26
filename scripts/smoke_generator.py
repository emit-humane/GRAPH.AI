"""Tiny smoke run of the generator. Used to surface bugs without waiting 5+ minutes
for the full 500K-row pipeline."""

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import GeneratorConfig, CONFIG_PATH, DATA_DIR, ensure_dirs


def main():
    ensure_dirs()
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        num_accounts=400,
        num_historical_transactions=8_000,
        num_stream_transactions=2_000,
    )
    print(f"[smoke] total={cfg.total_transactions}, accounts={cfg.num_accounts}")
    ctx = g1_account_builder.build_accounts(cfg)
    print(f"[smoke] G1 accounts={len(ctx.accounts)} ring_candidates="
          f"{ctx.accounts['is_potential_ring_member'].sum()}")
    normal = g2_normal_generator.generate_normal(cfg, ctx)
    print(f"[smoke] G2 normal rows={len(normal)}")
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal)
    print(f"[smoke] G3 combined={len(combined)} suspicious={len(suspicious)} gt={len(gt)}")
    hist, stream = g4_splitter.split_and_export(combined, gt, out_dir=DATA_DIR / "_smoke")
    print(f"[smoke] G4 hist={len(hist)} stream={len(stream)}")
    warmup = g4_splitter.generate_warmup(cfg, out_dir=DATA_DIR / "_smoke")
    print(f"[smoke] warmup={len(warmup)}")


if __name__ == "__main__":
    main()
