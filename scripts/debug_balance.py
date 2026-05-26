"""Diagnose the balance-replay mismatch — find the EARLIEST divergence."""
from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector
from src.system1_generator.common import GeneratorConfig, CONFIG_PATH

cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
    num_accounts=600,
    num_historical_transactions=10_000,
    num_stream_transactions=2_000,
)

ctx = g1_account_builder.build_accounts(cfg)
normal_df = g2_normal_generator.generate_normal(cfg, ctx)
combined, suspicious, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
combined = combined.reset_index(drop=True)

initial_balance = ctx.accounts.set_index("account_id")["initial_balance"].to_dict()
running = {}

sender_arr = combined["sender_account"].to_numpy()
receiver_arr = combined["receiver_account"].to_numpy()
amount_arr = combined["amount"].to_numpy()
status_arr = combined["transaction_status"].to_numpy()
sbb = combined["sender_balance_before"].to_numpy()
rbb = combined["receiver_balance_before"].to_numpy()
is_sus = combined["is_suspicious"].to_numpy()

mismatch_count = 0
self_transfers = 0
for i in range(len(combined)):
    s, r, amt, st = sender_arr[i], receiver_arr[i], amount_arr[i], status_arr[i]
    send_known = s in initial_balance
    recv_known = r in initial_balance

    if s == r:
        self_transfers += 1

    if send_known:
        expect = running.get(s, initial_balance[s])
        if abs(sbb[i] - expect) > 0.05:
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"--- mismatch #{mismatch_count} at i={i} ---")
                print(f"   sender={s} receiver={r} amt={amt:.2f} status={st} sus={is_sus[i]}")
                print(f"   sender_before stored={sbb[i]:.2f} expected={expect:.2f} diff={sbb[i]-expect:.2f}")
                # Walk back to find earlier rows touching this sender
                back = combined.iloc[:i]
                touch = back[(back["sender_account"] == s) | (back["receiver_account"] == s)]
                print(f"   prior rows touching this account: {len(touch)}")
                if len(touch) > 0:
                    last = touch.tail(3)
                    for _, row in last.iterrows():
                        role = "S" if row["sender_account"] == s else "R"
                        print(f"     i={row.name} role={role} amt={row['amount']:.2f} status={row['transaction_status']} sus={row['is_suspicious']}")
    if recv_known:
        expect_r = running.get(r, initial_balance[r])
        if abs(rbb[i] - expect_r) > 0.05:
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"--- mismatch #{mismatch_count} at i={i} (receiver) ---")
                print(f"   sender={s} receiver={r} amt={amt:.2f} status={st} sus={is_sus[i]}")
                print(f"   receiver_before stored={rbb[i]:.2f} expected={expect_r:.2f} diff={rbb[i]-expect_r:.2f}")
                back = combined.iloc[:i]
                touch = back[(back["sender_account"] == r) | (back["receiver_account"] == r)]
                print(f"   prior rows touching this account: {len(touch)}")
                if len(touch) > 0:
                    last = touch.tail(3)
                    for _, row in last.iterrows():
                        role = "S" if row["sender_account"] == r else "R"
                        print(f"     i={row.name} role={role} amt={row['amount']:.2f} status={row['transaction_status']} sus={row['is_suspicious']}")

    if st == "Failed":
        continue
    if send_known:
        running[s] = running.get(s, initial_balance[s]) - amt
    if recv_known:
        running[r] = running.get(r, initial_balance[r]) + amt

print(f"\ntotal rows: {len(combined)}")
print(f"mismatches: {mismatch_count}")
print(f"self transfers: {self_transfers}")
print(f"failed: {(status_arr == 'Failed').sum()}")
