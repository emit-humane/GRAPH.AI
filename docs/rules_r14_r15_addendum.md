# Layer 1 Addendum — Rules R14 (Fan-In) and R15 (Layering Chain)

**Purpose:** The baseline 13-rule engine has no dedicated detector for fan-in aggregation or layering-chain relays. R09 catches fan-*out* (sender's beneficiary count) but not fan-*in*; layering chains are only caught indirectly via R03 (velocity) or the graph layer. This addendum adds two structural rules that close both gaps.

**Status:** Normative. Append to `docs/architecture.md` directly after the L1 Rules table. Update the rule count from 13 to 15 everywhere it appears (L1 purpose line, RuleEngineOutput, "Execute 13 deterministic AML rules" → "Execute 15…", E4 per-rule precision, etc.).

**Key design point:** Both new rules are *structural* (multi-transaction) and read from `LiveGraphFeatureVector`, not just `LiveFeatureVector`. This is consistent with R10 (cycle closure), which already reads graph features. Two small schema additions to S5/S6 are required and specified below.

---

## Schema additions (prerequisite)

### S6 — add to LiveGraphFeatureVector

```python
# Fan-in support (for R14)
receiver_in_degree_unique_24h: int   # distinct senders into receiver in last 24h
receiver_inflow_amount_cv: float     # coefficient of variation of inbound amounts (24h);
                                     # low CV = uniform amounts = structured aggregation
# Layering relay support (for R15)
sender_is_relay_node: bool           # in_degree_unique ~1 and out_degree_unique ~1 over 7d
sender_last_inflow_amount: float     # most recent amount RECEIVED by the sender before this tx
sender_last_inflow_gap_seconds: float # seconds between sender's last inflow and this outflow
```

### S6 — processing additions

In `LiveGraphUpdater.process_event`, after the existing 2-hop neighborhood extraction:

```
# --- Fan-in features (R14) ---
# Look at the RECEIVER's inbound edges in the last 24h from the live multigraph
receiver_in_edges_24h = [
    e for e in G.in_edges(receiver, data=True)
    if (current_ts - e["timestamp"]).total_seconds() <= 86400
]
receiver_in_degree_unique_24h = len({e_src for e_src, _, _ in receiver_in_edges_24h})
inbound_amounts = [e["amount"] for *_, e in receiver_in_edges_24h]
receiver_inflow_amount_cv = (
    np.std(inbound_amounts) / (np.mean(inbound_amounts) + 1e-9)
    if len(inbound_amounts) >= 2 else 999.0
)

# --- Layering relay features (R15) ---
sender_in_unique_7d  = count_unique_senders_to(sender, window_days=7)
sender_out_unique_7d = count_unique_receivers_from(sender, window_days=7)
sender_is_relay_node = (sender_in_unique_7d <= 2) and (sender_out_unique_7d <= 2) \
                        and (sender_in_unique_7d >= 1) and (sender_out_unique_7d >= 1)

# Most recent inflow to the sender BEFORE this outgoing transaction
sender_inflows = sorted(
    [e for e in G.in_edges(sender, data=True) if e["timestamp"] < current_ts],
    key=lambda e: e["timestamp"]
)
if sender_inflows:
    last_inflow = sender_inflows[-1]
    sender_last_inflow_amount = last_inflow["amount"]
    sender_last_inflow_gap_seconds = (current_ts - last_inflow["timestamp"]).total_seconds()
else:
    sender_last_inflow_amount = 0.0
    sender_last_inflow_gap_seconds = float("inf")
```

---

## R14 — Fan-In Aggregation

| Rule ID | Name | Condition | Weight |
|---|---|---|---|
| **R14** | Fan-in aggregation | `receiver_in_degree_unique_24h >= 5 AND receiver_inflow_amount_cv < 0.35` | **0.8** |

### Rationale
A money mule's collection account sits at the center of a fan-in star: 5–15 distinct sources push funds in over a short window, often in similar amounts (to look like routine deposits). The signal is **on the receiver side** — many distinct senders in 24h *and* low amount variance (CV < 0.35 means inbound amounts are suspiciously uniform). This is the structural mirror of R09 (which detects fan-*out* via the sender's beneficiary count). R09 + R14 together cover both star typologies.

The CV guard matters: a popular merchant also has high `receiver_in_degree`, but its inbound amounts vary wildly (CV high), so it won't trip R14. A structured aggregation account has uniform inbound amounts (CV low).

### Condition (exact)
```python
def r14_fan_in(features, graph, event):
    return (graph.receiver_in_degree_unique_24h >= 5
            and graph.receiver_inflow_amount_cv < 0.35)
```

### Explanation template
```
"Fan-in aggregation: receiver collected from {receiver_in_degree_unique_24h} distinct "
"sources in 24h with near-uniform amounts (CV={receiver_inflow_amount_cv:.2f}), "
"consistent with a collection/mule account."
```

---

## R15 — Layering Chain Relay

| Rule ID | Name | Condition | Weight |
|---|---|---|---|
| **R15** | Layering chain relay | `sender_is_relay_node AND 0.85 <= (amount / sender_last_inflow_amount) <= 1.0 AND sender_last_inflow_gap_seconds < 7200` | **0.9** |

### Rationale
A layering hop is a pass-through: funds arrive, and a near-equal amount (85–100%, the small shortfall being the launderer's cut or fees) is forwarded onward shortly after. The account behaves as a relay — roughly one source in, one destination out (`sender_is_relay_node`) — rather than a genuine endpoint that accumulates or originates funds. The `< 7200s` (2-hour) gap captures the "maintained speed" signal: real layering moves fast to outpace investigation.

This is precisely the "C → D reduced value and maintained speed, strengthening the structuring hypothesis" case from the reference dashboards — a mid-chain relay forwarding most of what it just received.

### Condition (exact)
```python
def r15_layering_relay(features, graph, event):
    if graph.sender_last_inflow_amount <= 0:
        return False  # no prior inflow to relay
    retain_ratio = event.amount / graph.sender_last_inflow_amount
    return (graph.sender_is_relay_node
            and 0.85 <= retain_ratio <= 1.0
            and graph.sender_last_inflow_gap_seconds < 7200)
```

### Explanation template
```
"Layering chain relay: account forwarded {retain_ratio:.0%} of a recent inflow "
"(₹{sender_last_inflow_amount:,.0f}) within {gap_minutes:.0f} minutes as a "
"single-in/single-out relay node — a classic mid-chain layering hop."
```

---

## Updated scoring math

The scoring formula in the L1 spec is unchanged in form, but `max_possible_raw` now sums **15** weights:

```
Original 13 weights:
  R01=1.0, R02=1.0, R03=0.8, R04=0.9, R05=0.9, R06=0.7, R07=0.6,
  R08=0.7, R09=0.8, R10=1.0, R11=0.6, R12=0.8, R13=0.7
New:
  R14=0.8, R15=0.9

max_possible_raw = sum(all 15 weights) = 11.2   (was 9.5)

rule_score = min(100, (raw_score / max_possible_raw) × 100)
```

**Important:** recomputing `max_possible_raw` from 9.5 to 11.2 slightly lowers every transaction's `rule_score` (the denominator grew). If you have already tuned the P1 fusion weights or risk-level thresholds against the 13-rule denominator, re-verify them after adding R14/R15. The cleanest approach: define `max_possible_raw = sum(r.weight for r in RULES)` programmatically so it auto-updates and you never hardcode 9.5 or 11.2.

---

## Tests (add to tests/test_layer1.py)

```python
def test_r14_fan_in_fires():
    # 6 distinct sources, uniform amounts → CV low → R14 fires
    graph = make_graph_vec(receiver_in_degree_unique_24h=6,
                           receiver_inflow_amount_cv=0.20)
    out = engine.evaluate(features=normal_features(), graph=graph, event=normal_event())
    assert "R14" in out.triggered_rules

def test_r14_ignores_busy_merchant():
    # Many sources but highly variable amounts (real merchant) → R14 does NOT fire
    graph = make_graph_vec(receiver_in_degree_unique_24h=12,
                           receiver_inflow_amount_cv=1.4)
    out = engine.evaluate(features=normal_features(), graph=graph, event=normal_event())
    assert "R14" not in out.triggered_rules

def test_r15_layering_relay_fires():
    # Received 1,000,000 then forwards 950,000 (95%) 30 min later, relay node → R15 fires
    graph = make_graph_vec(sender_is_relay_node=True,
                           sender_last_inflow_amount=1_000_000,
                           sender_last_inflow_gap_seconds=1800)
    event = make_event(amount=950_000)
    out = engine.evaluate(features=normal_features(), graph=graph, event=event)
    assert "R15" in out.triggered_rules

def test_r15_ignores_partial_spend():
    # Received 1,000,000 but only spends 100,000 (10%) → not a relay forward → R15 does NOT fire
    graph = make_graph_vec(sender_is_relay_node=True,
                           sender_last_inflow_amount=1_000_000,
                           sender_last_inflow_gap_seconds=1800)
    event = make_event(amount=100_000)
    out = engine.evaluate(features=normal_features(), graph=graph, event=event)
    assert "R15" not in out.triggered_rules

def test_r15_ignores_slow_forward():
    # 95% forwarded but 5 hours later → exceeds 2h relay window → R15 does NOT fire
    graph = make_graph_vec(sender_is_relay_node=True,
                           sender_last_inflow_amount=1_000_000,
                           sender_last_inflow_gap_seconds=18000)
    event = make_event(amount=950_000)
    out = engine.evaluate(features=normal_features(), graph=graph, event=event)
    assert "R15" not in out.triggered_rules

def test_max_possible_raw_is_computed():
    # Guard against hardcoded denominator
    from src.system2_detection.layer1_rules.rule_definitions import RULES
    assert abs(sum(r.weight for r in RULES) - 11.2) < 1e-6
    assert len(RULES) == 15
```

---

## Downstream files to update

Adding R14/R15 touches several places beyond `rule_definitions.py`:

1. **`rule_definitions.py`** — add the two `Rule` entries with conditions + explanation templates above.
2. **S6 (`s6_dynamic_graph.py`)** — add the 6 new fields to `LiveGraphFeatureVector` and the feature computations.
3. **L2A (`l2a_gfp.py`)** — already computes `fan_in_score`/`fan_out_score`; no change needed, but R14 uses the *live* 24h-windowed count from S6, not the static L2A score. Keep them distinct.
4. **E4 Pattern Coverage Analyzer** — `per_rule_precision` will now include `R14_fan_in` and `R15_layering_relay`. Update the expected-keys test in `tests/test_system3.py`.
5. **Scenario Studio** (if built) — the fan_in typology should now list R14 in its "targets detection rules" chips; layering_chain should list R15. Update the `TYPOLOGIES` rule arrays: `fan_in.rules` → add "R14"; `layering_chain.rules` → add "R15".
6. **Reference doc text** — every "13 rules" / "13 deterministic AML rules" mention becomes "15".

---

## Claude Code prompt to implement this

Paste into a session AFTER Session 5 (L1) and Session 4 (S6) are done:

```
Read docs/architecture.md (L1 Rules section) and docs/rules_r14_r15_addendum.md in full.
The addendum is normative — implement both new rules and the schema additions exactly.

1. In src/system2_detection/shared/s6_dynamic_graph.py:
   - Add the 6 new fields to the LiveGraphFeatureVector Pydantic model:
     receiver_in_degree_unique_24h, receiver_inflow_amount_cv, sender_is_relay_node,
     sender_last_inflow_amount, sender_last_inflow_gap_seconds
   - Implement their computation in process_event() per the addendum's "S6 processing
     additions" block, reading from the live MultiDiGraph.

2. In src/system2_detection/layer1_rules/rule_definitions.py:
   - Add Rule R14 (fan-in aggregation, weight 0.8) and R15 (layering chain relay,
     weight 0.9) with the exact conditions and explanation templates from the addendum.
   - Ensure max_possible_raw is computed as sum(r.weight for r in RULES) — NOT hardcoded.
     The RULES list must now have 15 entries.

3. Update tests/test_layer1.py with the six tests in the addendum
   (test_r14_fan_in_fires, test_r14_ignores_busy_merchant, test_r15_layering_relay_fires,
   test_r15_ignores_partial_spend, test_r15_ignores_slow_forward,
   test_max_possible_raw_is_computed).

4. Update tests/test_system3.py expected per_rule_precision keys to include
   R14_fan_in and R15_layering_relay.

5. Grep the repo for "13 rule" and "13 deterministic" and update those strings to "15".

Run pytest tests/test_layer1.py and confirm all 6 new tests pass. Then run the dry
pipeline (scripts/run_pipeline_dry.py --limit 1000) and confirm R14 and R15 appear in
at least some triggered_rules across the fan_in and layering_chain ground-truth scenarios.
```
