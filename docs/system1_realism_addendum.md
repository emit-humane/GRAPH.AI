# System 1 — Realism Patch (Addendum to G1–G4)

**Purpose:** Close four realism gaps in the baseline generator spec that would otherwise let detection models learn artifacts of the synthesis process rather than real laundering signal.

**Status:** This addendum is normative. Where it conflicts with the original G1–G4 text, this addendum wins. Append to `docs/architecture.md` directly below the System 1 section.

---

## Patch 1 — Temporal Realism (modifies G3, G4)

### Problem

Original G3 says scenarios occur "within a 24h window" but does not require the injected transactions to interleave with the surrounding normal-transaction stream. A literal implementation can dump all 12 structuring edges back-to-back inside one second, then leave the rest of the window empty. This breaks:

- TGN memory updates (no plausible time gaps between events)  
- Velocity rules (R02, R03) become trivially detectable on perfectly clustered bursts  
- Streaming order is degenerate

### Required behavior

**G3 Constraint T1 — Spread within scenario window.** For every injected scenario instance:

- The scenario's logical timestamps must be sampled uniformly (or per a scenario-specific distribution) across the full declared window, not back-to-back.  
- For structuring (window: implicit \~24h): 5–15 transactions sampled with mean inter-arrival gap \= window / (count \+ 1), with Gaussian jitter σ \= 0.3 × mean gap, clipped to minimum 60 seconds.  
- For circular\_laundering (48h cycle): each hop's timestamp \= previous hop \+ Exponential(mean \= window / num\_hops), clipped to minimum 5 minutes.  
- For layering\_chain (no fixed window — define as 6h × num\_hops): each hop timestamp \= prev \+ Exponential(mean \= 6h).  
- For velocity\_burst (defined as ≥15 txns in 1h): use the burst distribution — Exponential(mean \= 3 minutes), capped at 60 minutes total. This one IS intentionally clustered; that's the typology.  
- For all other scenarios: spread uniformly across a window proportional to scenario count.

**G3 Constraint T2 — Causality preserved.** For directed-path scenarios (circular\_laundering, layering\_chain, round\_tripping, fan\_in, fan\_out):

- `timestamp(edge_k+1) > timestamp(edge_k)` strictly. Money cannot arrive at hop k+1 before it leaves hop k.  
- Add a minimum settlement delay sampled from `Uniform(30s, 300s)` between consecutive hops. NEFT batches in India settle in 30-minute slots in reality; using 30s+ keeps the simulation tractable while preserving causal ordering.

**G3 Constraint T3 — Interleaving with normal traffic.** After G3 finishes injecting, the COMBINED population (normal \+ suspicious) must be re-sorted globally by timestamp, then verified:

- No two consecutive transactions in the final stream share the exact same timestamp (add ε microseconds if collisions occur).  
- For every account that participates in a ring: the account's own transactions (normal \+ suspicious combined) must form a monotonically increasing timestamp sequence. Run this check explicitly in G4 and assert.

**G4 Constraint T4 — Split preserves scenario integrity.** When splitting 90/10 chronologically:

- A scenario whose timestamps span the 90/10 boundary is acceptable and desirable (System 2's online stream sees the latter half of partially-historical rings — this is realistic).  
- BUT: a single scenario instance must not have its first edge in the 10% stream and its last edge in the 90% historical set (causality across the split is fine; reverse causality is not).  
- Implementation: after sorting and splitting, scan stream\_transactions for any `fraud_ring_id` whose minimum stream-set timestamp is less than that ring's maximum historical-set timestamp. If found, push the entire affected ring forward into the stream set or backward into history (whichever moves fewer txns).

### Test (add to tests/test\_generator.py)

def test\_temporal\_realism():

    \# Constraint T1: scenarios are spread, not bursts (except velocity\_burst)

    for ring\_id, ring\_txns in suspicious.groupby("fraud\_ring\_id"):

        pattern \= ring\_txns\["synthetic\_pattern\_type"\].iloc\[0\]

        if pattern \== "velocity\_burst":

            continue

        gaps \= ring\_txns\["timestamp"\].sort\_values().diff().dropna().dt.total\_seconds()

        assert gaps.min() \>= 60, f"Ring {ring\_id} has sub-minute gap"

        assert gaps.std() \> 0, f"Ring {ring\_id} timestamps are evenly spaced (not jittered)"

    \# Constraint T2: causal scenarios are monotonic

    causal\_patterns \= {"circular\_laundering", "layering\_chain", "round\_tripping"}

    for ring\_id, ring\_txns in suspicious.groupby("fraud\_ring\_id"):

        if ring\_txns\["synthetic\_pattern\_type"\].iloc\[0\] not in causal\_patterns:

            continue

        ts \= ring\_txns.sort\_values("timestamp")\["timestamp"\]

        assert ts.is\_monotonic\_increasing

    \# Constraint T3: every account's combined timeline is monotonic

    combined \= pd.concat(\[normal, suspicious\]).sort\_values("timestamp")

    for acc, acc\_txns in combined.groupby("sender\_account"):

        ts \= acc\_txns\["timestamp"\]

        assert ts.is\_monotonic\_increasing, f"Account {acc} has out-of-order txns"

    \# Constraint T4: no reverse causality across split

    for ring\_id in stream\_df\["fraud\_ring\_id"\].dropna().unique():

        hist\_max \= hist\_df\[hist\_df\["fraud\_ring\_id"\] \== ring\_id\]\["timestamp"\].max()

        stream\_min \= stream\_df\[stream\_df\["fraud\_ring\_id"\] \== ring\_id\]\["timestamp"\].min()

        if pd.notna(hist\_max) and pd.notna(stream\_min):

            assert stream\_min \> hist\_max, f"Ring {ring\_id} has reverse causality across split"

---

## Patch 2 — Balance Consistency (modifies G2, G3)

### Problem

Original G2 computes balance fields. G3 inherits the schema but never specifies that balance fields must be recomputed after injection. If a normal transaction at t=10:00 leaves account A with ₹50,000, and then G3 injects a suspicious ₹9,00,000 transaction at t=09:30, the original 10:00 row still says `sender_balance_before=₹50,000` even though the account should now have ₹50,000 − ₹9,00,000 \= ₹−8,50,000 by then. Anomaly models will learn the artifact: "suspicious accounts have impossible balance arithmetic."

### Required behavior

**G3 Constraint B1 — Recompute all balances post-injection.** After G3 has injected all scenarios and G4 has merged \+ sorted globally by timestamp, run a balance-replay pass:

For each account A (as sender or receiver):

    starting\_balance \= accounts\_table\[A\].initial\_balance

    running\_balance \= starting\_balance

    for each transaction T involving A, in timestamp order:

        if A \== T.sender:

            T.sender\_balance\_before \= running\_balance

            running\_balance \-= T.amount

            T.sender\_balance\_after \= running\_balance

        elif A \== T.receiver:

            T.receiver\_balance\_before \= running\_balance

            running\_balance \+= T.amount

            T.receiver\_balance\_after \= running\_balance

**G3 Constraint B2 — Handle negative balances realistically.** If balance replay produces a negative balance:

- Set `transaction_status = "Failed"` for that transaction.  
- Do NOT update the balance (failed txns don't move money).  
- Continue replay with the prior balance.  
- This is realistic: \~1–3% of real bank transactions fail for insufficient funds, and failure logs are themselves a fraud signal.

**G3 Constraint B3 — Top up suspicious accounts to plausibility.** When G3 plans a scenario that requires large outflows (structuring, layering\_chain, fan\_out), pre-stage the funds: inject a "Salary credit" or "Wire credit" deposit into the entry account within the 7 days preceding the scenario, sized to match the scenario's total value. Label these top-ups as `is_suspicious = False, fraud_ring_id = null` — they are *normal* incoming transactions that just happen to enable the ring. Real launderers also need their dirty money to arrive somehow; the model should learn to find what happens AFTER the deposit, not flag the deposit itself.

### Test (add to tests/test\_generator.py)

def test\_balance\_consistency():

    combined \= pd.concat(\[normal, suspicious\]).sort\_values("timestamp")

    accounts\_df \= pd.read\_parquet("data/internal/accounts.parquet")

    for acc\_id, acc\_row in accounts\_df.iterrows():

        running \= acc\_row\["initial\_balance"\]

        acc\_txns \= combined\[

            (combined\["sender\_account"\] \== acc\_id) | (combined\["receiver\_account"\] \== acc\_id)

        \].sort\_values("timestamp")

        for \_, t in acc\_txns.iterrows():

            if t\["transaction\_status"\] \== "Failed":

                continue

            if t\["sender\_account"\] \== acc\_id:

                assert abs(t\["sender\_balance\_before"\] \- running) \< 0.01

                running \-= t\["amount"\]

                assert abs(t\["sender\_balance\_after"\] \- running) \< 0.01

            else:

                assert abs(t\["receiver\_balance\_before"\] \- running) \< 0.01

                running \+= t\["amount"\]

                assert abs(t\["receiver\_balance\_after"\] \- running) \< 0.01

    \# Failure rate sanity

    failure\_rate \= (combined\["transaction\_status"\] \== "Failed").mean()

    assert 0.005 \< failure\_rate \< 0.05, "Implausible failure rate"

---

## Patch 3 — Blended Suspicious \+ Normal Behavior (modifies G1, G2, G3)

### Problem

Original G3 picks ring members and gives them suspicious transactions. If those accounts have no other activity, the model learns: "any account that only ever does ring transactions is suspicious." That's a giveaway artifact, not a real laundering signal. Real money mules also receive salary, pay for groceries, top up phone bills.

### Required behavior

**G1 Constraint X1 — Tag ring candidates upfront.** During G1 account generation, designate \~10% of all accounts as `is_potential_ring_member = True`. These accounts have NORMAL G2 baselines (salaries, merchant payments, etc. as G2 would generate for any account). G3 will preferentially select ring members from this tagged pool, but the model never sees the tag — it must learn behavior from the transactions alone.

**G2 Constraint X2 — Generate normal activity for ALL accounts including future ring members.** G2 runs first and generates normal traffic for every account in the population, including the 10% tagged as potential ring members. Specifically:

- Each account, regardless of tag, receives at least one "Salary/Income" credit per 30 days (amount drawn from declared\_income\_bracket if Individual; from business revenue distribution if Corporate). Use `merchant_category = "Salary"` or `"Business_Income"`.  
- Each account makes at least 2–8 merchant payments per week (low-value `Card` or `UPI` transactions, amounts ₹100–₹5,000, merchant\_category sampled from real Indian MCC distribution: groceries, fuel, telecom, food delivery).  
- This is BASELINE behavior — must exist whether or not the account later becomes a ring member.

**G3 Constraint X3 — Suspicious-to-normal ratio per ring member.** For every account participating in a ring:

- The ratio `suspicious_txn_count / total_txn_count` for that account MUST fall in \[0.05, 0.40\].  
- If after injection a ring member has ratio \> 0.40, inject additional normal G2-style transactions for that account in the time window surrounding the ring activity (different timestamps, normal merchant categories, normal amounts) until the ratio is ≤ 0.40.  
- If ratio \< 0.05 (member only does one ring transaction in an otherwise heavy normal account), this is fine — small involvement is realistic.

**G3 Constraint X4 — Suspicious accounts continue normal activity AFTER ring activity.** A ring member's normal G2 traffic does not stop the day the ring activates. The full 90 days of normal transactions for that account remain in the dataset, just with the ring transactions interleaved. Verify: every ring member has at least one non-suspicious transaction both BEFORE and AFTER the median timestamp of their ring activity.

### Test

def test\_blended\_behavior():

    combined \= pd.concat(\[normal, suspicious\]).sort\_values("timestamp")

    ring\_accounts \= suspicious\["sender\_account"\].unique()

    for acc in ring\_accounts:

        acc\_txns \= combined\[

            (combined\["sender\_account"\] \== acc) | (combined\["receiver\_account"\] \== acc)

        \]

        sus\_count \= acc\_txns\["is\_suspicious"\].sum()

        total \= len(acc\_txns)

        ratio \= sus\_count / total

        assert 0.05 \<= ratio \<= 0.40, f"Account {acc} suspicious ratio \= {ratio:.2f}"

        \# Has both before and after normal activity

        sus\_times \= acc\_txns\[acc\_txns\["is\_suspicious"\]\]\["timestamp"\]

        median\_sus \= sus\_times.median()

        normal\_times \= acc\_txns\[\~acc\_txns\["is\_suspicious"\]\]\["timestamp"\]

        assert (normal\_times \< median\_sus).any(), f"{acc} no normal txns before ring"

        assert (normal\_times \> median\_sus).any(), f"{acc} no normal txns after ring"

    \# Salary check: every account has ≥1 income txn per 30 days

    income\_categories \= {"Salary", "Business\_Income"}

    for acc in accounts\_df\["account\_id"\]:

        income \= combined\[

            (combined\["receiver\_account"\] \== acc) &

            (combined\["merchant\_category"\].isin(income\_categories))

        \]

        assert len(income) \>= 2, f"{acc} has no income transactions over 90 days"

---

## Patch 4 — Shared Devices / IPs / Geographies in Rings (modifies G3)

### Problem

Original G2 assigns each account its own device/IP fingerprint. G3 doesn't override this. Result: ring members each transact from their own device, which is unrealistic — real fraud rings often share infrastructure (one money mule operator running multiple accounts from the same laptop), and Rule R08 (shared\_device\_count \> 5\) and graph-based device co-occurrence features become useless because no ring members share anything.

### Required behavior

**G3 Constraint D1 — Per-ring infrastructure pools.** For each generated ring, create a ring-level infrastructure pool with:

- `shared_device_pool`: 1–3 device IDs (size depends on ring size: rings of 3–5 use 1 device; 6–10 use 2; 10+ use 3).  
- `shared_ip_subnet`: a single /24 IPv4 subnet (e.g., `203.0.113.0/24`). Individual IPs are sampled from this subnet.  
- `shared_geo_cluster`: a single lat/long centroid with σ \= 0.05° (\~5 km radius). Individual transaction geos are sampled from a Gaussian around this centroid.

**G3 Constraint D2 — Ring members use the shared pool with probability p.** For each suspicious transaction by a ring member:

- With probability `p_shared = 0.70`: device\_id sampled from `shared_device_pool`, ip\_address sampled from `shared_ip_subnet`, geo sampled from `shared_geo_cluster`.  
- With probability `1 - p_shared = 0.30`: use the account's own G2-assigned device/IP/geo (the member is operating from their own home that day).  
- This 70/30 split is critical: 100% sharing would be a giveaway artifact; 0% sharing erases the signal entirely.

**G3 Constraint D3 — Sharing applies ONLY to suspicious transactions.** Ring members' NORMAL transactions (from Patch 3\) continue to use their own G2 device/IP/geo. So Rule R08 will find shared-device usage concentrated on suspicious transactions specifically — exactly the kind of pattern that matters.

**G3 Constraint D4 — Pattern-specific overrides.**

- `cross_border_layering`: each hop must use a different country's geo (this contradicts ring-level geo sharing — for this scenario, drop D2 geo sharing, keep device and IP sharing). Use IPs from the country's IP space for that hop.  
- `fraud_ring` (dense clique typology): increase `p_shared` to 0.85 because a tightly-knit ring is more likely to share everything.  
- `dormant_activation`: the dormant account that suddenly reactivates uses a NEW device that none of its prior normal activity used. This is the "stolen credentials" signal. The downstream 3+ disbursement accounts share that new device.

**G2 Constraint D5 — Background shared-device noise.** To avoid making "any shared device" a perfect fraud signal, G2 must also create some background sharing among NORMAL accounts:

- Family accounts: \~5% of Individual accounts share a device with one other Individual account (spouse, family member). Sampled randomly across the normal population.  
- Public terminals: \~1% of transactions across the entire dataset come from a small pool of 50 "public" device IDs (cyber café terminals, kiosks).  
- Without this noise, R08 has 100% precision on synthetic data and 50% precision in production — a classic dataset bias.

### Test

def test\_shared\_infrastructure():

    \# Constraint D2: shared device usage in rings is \~70%, not 100%

    for ring\_id, ring\_txns in suspicious.groupby("fraud\_ring\_id"):

        if ring\_txns\["synthetic\_pattern\_type"\].iloc\[0\] \== "cross\_border\_layering":

            continue  \# geo sharing waived

        devices \= ring\_txns\["device\_id"\].value\_counts()

        if len(ring\_txns) \>= 5:

            top\_share \= devices.iloc\[0\] / len(ring\_txns)

            assert 0.5 \< top\_share \< 0.95, f"Ring {ring\_id} device sharing {top\_share:.2f}"

    \# Constraint D3: normal txns of ring members don't share ring's devices

    for ring\_id, ring\_txns in suspicious.groupby("fraud\_ring\_id"):

        ring\_devices \= set(ring\_txns\["device\_id"\].unique())

        ring\_members \= set(ring\_txns\["sender\_account"\].unique())

        for member in ring\_members:

            member\_normal \= normal\[normal\["sender\_account"\] \== member\]

            normal\_devices \= set(member\_normal\["device\_id"\].unique())

            overlap \= ring\_devices & normal\_devices

            assert len(overlap) \<= 1, f"{member} normal & ring devices overlap heavily"

    \# Constraint D5: background shared-device noise exists in normal data

    normal\_device\_counts \= normal\["device\_id"\].value\_counts()

    shared\_normal\_devices \= (normal\_device\_counts \> 50).sum()

    assert shared\_normal\_devices \>= 50, "Missing public-terminal device noise"

---

## Updated G3 Processing Order

The original G3 was a single-pass injector. To honor all four patches, G3 becomes a 6-step pipeline:

Step 1: SAMPLE — Select ring members (preferentially from G1's tagged candidates).

        Create ring-level infrastructure pools (Patch 4: D1).

Step 2: PRE-STAGE — For scenarios needing entry funds, inject normal-labeled deposits

        in the 7 days preceding scenario start (Patch 2: B3).

Step 3: PLAN — For each ring, compute the logical timestamp sequence with proper

        spread \+ causality \+ settlement delays (Patch 1: T1, T2).

Step 4: GENERATE — Emit suspicious transactions. Each transaction:

        \- With prob 0.70, uses shared infra pool. Else uses member's own (Patch 4: D2).

        \- is\_suspicious \= True, fraud\_ring\_id, laundering\_stage, synthetic\_pattern\_type

          all populated.

        \- Balance fields TEMPORARILY set to NaN. Will be recomputed in Step 6\.

Step 5: REBALANCE NORMAL TRAFFIC — For each ring member whose suspicious\_ratio

        now exceeds 0.40, inject additional normal G2-style traffic until ratio ≤ 0.40

        (Patch 3: X3). These additions are is\_suspicious=False, ring\_id=null.

Step 6: BALANCE REPLAY — Globally sort the combined population by timestamp.

        For each account, replay transactions in order and set sender/receiver

        balance\_before/after. Mark Failed where insufficient funds (Patch 2: B1, B2).

Step 7: INTERLEAVE VERIFY — Assert per-account timestamp monotonicity (Patch 1: T3).

The original G3 output schemas (suspicious\_transactions.parquet, ground\_truth\_records.parquet) remain identical. Only the generation logic changes.

---

## Updated G4 Processing

After G4's existing concat \+ sort \+ 90/10 split, add:

Step G4.5: SCAN — For each unique fraud\_ring\_id, find min/max timestamps in both halves.

           If any ring violates Patch 1 Constraint T4 (reverse causality across split),

           move the entire ring to whichever side requires fewer transaction moves.

           Re-sort both halves and verify monotonicity.

---

## Summary checklist

After implementing patches 1–4, the generator output must satisfy:

- [ ] No two consecutive scenario edges in the same ring share a timestamp (T1).  
- [ ] Causal scenarios are timestamp-monotonic edge-by-edge (T2).  
- [ ] Every account's combined transaction sequence is timestamp-monotonic (T3).  
- [ ] No scenario has reverse causality across the 90/10 split (T4).  
- [ ] Every account's balance\_before/after is consistent with the running balance from initial\_balance (B1).  
- [ ] Insufficient-fund transactions are marked Failed and do not move money (B2).  
- [ ] Suspicious-ratio per ring member ∈ \[0.05, 0.40\] (X3).  
- [ ] Every ring member has normal txns before AND after their ring activity (X4).  
- [ ] Every account has ≥2 income-category transactions over 90 days (X2).  
- [ ] Ring device-sharing concentration is 50–95% per ring (not 100%) (D2).  
- [ ] Ring members' normal txns use different devices than the ring's pool (D3).  
- [ ] Background public/family shared devices exist in normal data (D5).

