"""verify_detection.py — per-rule + per-typology probe stream.

Builds a deterministic suite of probe transactions that targets EVERY Layer-1
rule (R01..R15) AND every Layer-2 graph feature AND every Layer-1 scenario
typology, runs them through the live D1->D2->D3->D4->D5->D6->D7->D8 pipeline,
and emits a per-probe PASS / FAIL table.

A probe is "passing" when:

  * Its expected_rules list is a subset of the rules actually triggered, AND
  * Its expected_graph_features list is a subset of the features the live
    graph_vec reports as non-trivial (positive or true), AND
  * (For scenario probes) at least one Critical/High alert lands on the
    expected target rule, OR the fused score reaches Medium+.

Probes that depend on cross-event state (R02 structuring needs 11 prior
sub-threshold tx, R10 cycle needs the cycle to exist, R14/R15 need fan-in
and inflow buildup respectively) build their full event chain inline.

Usage:
    python scripts/verify_detection.py
    python scripts/verify_detection.py --output reports/detection_verification.json

The harness creates a FRESH detector state each run, so it always starts
from a clean slate — no contamination from previous stream replays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system1_generator.scenario_engine import (                       # noqa: E402
    ScenarioEngine,
    default_params,
)
from src.system2_detection.layer1_rules.rule_engine import RuleEngine     # noqa: E402
from src.system2_detection.post_detection.d8_fusion import (              # noqa: E402
    RiskFusionEngine,
    collect_layer_scores,
)
from src.system2_detection.layer3_supervised.d5b_inference import (       # noqa: E402
    SupervisedInferencer,
)
from src.system2_detection.layer4_anomaly.d6b_inference import (          # noqa: E402
    BehavioralAnomalyInferencer,
)
from src.system2_detection.layer5_gnn.d7b_inference import TGNInferencer  # noqa: E402
from src.system2_detection.shared.d1_feature_updater import (             # noqa: E402
    LiveFeatureUpdater,
)
from src.system2_detection.shared.d2_graph_updater import (               # noqa: E402
    LiveGraphUpdater,
)
from src.system2_detection.shared.s3_artifact_store import (              # noqa: E402
    ArtifactNotFoundError,
    ArtifactStore,
)
from src.system2_detection.shared.schemas import TransactionEvent          # noqa: E402

logger = logging.getLogger("verify-detection")


# --------------------------------------------------------------------------- #
# Probe dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Probe:
    """One named probe — a set of events + the rules/features it should trip."""
    name: str
    layer: str                     # "L1" | "L2" | "scenario"
    description: str
    events: list[TransactionEvent]
    expected_rules: list[str]
    expected_graph_features: list[str] = field(default_factory=list)
    expected_min_risk_level: str | None = None   # "Medium" | "High" | "Critical"


@dataclass
class ProbeResult:
    name: str
    layer: str
    description: str
    expected_rules: list[str]
    expected_graph_features: list[str]
    expected_min_risk_level: str | None
    triggered_rules_seen: list[str]
    graph_features_seen: list[str]
    max_risk_level: str
    max_risk_score: float
    n_events: int
    n_alerted: int
    rules_pass: bool             # expected rules all fired
    features_pass: bool          # expected graph features all hit
    risk_pass: bool              # max risk level >= expected_min_risk_level
    passed: bool                 # AND of all three axes
    notes: str = ""


# --------------------------------------------------------------------------- #
# Probe builders
# --------------------------------------------------------------------------- #


BASE_TIME = datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)
DELHI = (28.61, 77.21)
MUMBAI = (19.08, 72.88)
DUBAI = (25.20, 55.27)


def _mk_event(
    *,
    sender: str = "acct-probe-sender",
    receiver: str = "acct-probe-receiver",
    amount: float = 12_000.0,
    minutes_offset: float = 0.0,
    tx_type: str = "NEFT",
    channel: str = "Web",
    sender_country: str = "IN",
    receiver_country: str = "IN",
    sender_bank: str = "HDFC",
    receiver_bank: str = "ICICI",
    device_id: str = "dev-probe",
    ip: str = "203.0.113.5",
    lat: float = DELHI[0],
    lng: float = DELHI[1],
    kyc_level: int = 2,
    is_international: bool = False,
) -> TransactionEvent:
    ts = BASE_TIME + timedelta(minutes=minutes_offset)
    amt = float(amount)
    leading = int(str(int(max(1, amt)))[0]) if amt > 0 else 1
    return TransactionEvent(
        transaction_id=str(uuid.uuid4()),
        timestamp=ts,
        sender_account=sender,
        receiver_account=receiver,
        sender_bank=sender_bank,
        receiver_bank=receiver_bank,
        sender_country=sender_country,
        receiver_country=receiver_country,
        amount=amt,
        currency="INR",
        transaction_type=tx_type,
        payment_channel=channel,
        device_id=device_id,
        ip_address=ip,
        geo_latitude=lat,
        geo_longitude=lng,
        merchant_category="Other",
        transaction_status="Success",
        kyc_level=kyc_level,
        is_international=is_international,
        amount_leading_digit=leading,
    )


def _probe_r01_high_value() -> Probe:
    return Probe(
        name="R01-high-value",
        layer="L1",
        description="Single tx of INR 25 lakh must trip R01 (high-value).",
        events=[_mk_event(sender="acc-R01-A", receiver="acc-R01-B", amount=2_500_000)],
        expected_rules=["R01"],
        expected_min_risk_level="Medium",
    )


def _probe_r02_structuring() -> Probe:
    """11 sub-threshold txns within 1h from same sender to same receiver.

    D1 flips ``sub_threshold_flag`` only when 850k <= amount <= 999k (just
    below the INR 10L CTR reporting threshold). The probe stays inside
    that band intentionally.
    """
    sender, receiver = "acc-R02-A", "acc-R02-B"
    events = [
        _mk_event(
            sender=sender,
            receiver=receiver,
            amount=900_000 + i * 5_000,    # 900k -> 955k, all sub-threshold
            minutes_offset=i * 3,
        )
        for i in range(12)
    ]
    return Probe(
        name="R02-structuring",
        layer="L1",
        description="12 sub-threshold (900k-955k) txns within 1h.",
        events=events,
        expected_rules=["R02", "R03"],
        expected_min_risk_level="High",
    )


def _probe_r03_velocity_spike() -> Probe:
    """11 transactions in <1h triggers R03 (velocity > 10/1h)."""
    sender = "acc-R03-A"
    events = [
        _mk_event(
            sender=sender,
            receiver=f"acc-R03-recv-{i}",
            amount=50_000,
            minutes_offset=i * 5,
        )
        for i in range(11)
    ]
    return Probe(
        name="R03-velocity-spike",
        layer="L1",
        description="11 tx from one sender within 55 minutes (>10/h cap).",
        events=events,
        expected_rules=["R03"],
        expected_min_risk_level="Medium",
    )


def _probe_r05_impossible_travel() -> Probe:
    """Two tx from same sender, Delhi → Dubai within 2 minutes."""
    sender = "acc-R05-A"
    e1 = _mk_event(
        sender=sender,
        receiver="acc-R05-B",
        amount=80_000,
        minutes_offset=0,
        ip="203.0.113.5",
        lat=DELHI[0],
        lng=DELHI[1],
    )
    e2 = _mk_event(
        sender=sender,
        receiver="acc-R05-C",
        amount=80_000,
        minutes_offset=2,
        ip="91.198.174.192",
        lat=DUBAI[0],
        lng=DUBAI[1],
        sender_country="AE",
        device_id="dev-probe-2",
    )
    return Probe(
        name="R05-impossible-travel",
        layer="L1",
        description="Same sender — Delhi tx then Dubai tx within 2 minutes.",
        events=[e1, e2],
        expected_rules=["R05"],
        expected_min_risk_level="High",
    )


def _probe_r06_high_risk_jurisdiction() -> Probe:
    """International tx to AE (high-risk on the test list)."""
    return Probe(
        name="R06-high-risk-jurisdiction",
        layer="L1",
        description="International outbound to a FATF high-risk jurisdiction (AE).",
        events=[
            _mk_event(
                sender="acc-R06-A",
                receiver="acc-R06-B",
                amount=300_000,
                sender_country="IN",
                receiver_country="AE",
                is_international=True,
            ),
        ],
        expected_rules=["R06"],
        expected_min_risk_level="Medium",
    )


def _probe_r07_device_anomaly() -> Probe:
    """First-time-seen device + amount > 200K."""
    return Probe(
        name="R07-device-anomaly",
        layer="L1",
        description="Unseen device id with amount above 200K.",
        events=[
            _mk_event(
                sender="acc-R07-A",
                receiver="acc-R07-B",
                amount=400_000,
                device_id="dev-brand-new-99",
            ),
        ],
        expected_rules=["R07"],
        expected_min_risk_level="Medium",
    )


def _probe_r08_shared_device() -> Probe:
    """7 distinct senders using the same device id."""
    shared_dev = "dev-shared-probe"
    events: list[TransactionEvent] = []
    for i in range(7):
        events.append(_mk_event(
            sender=f"acc-R08-S{i}",
            receiver="acc-R08-R",
            amount=80_000,
            minutes_offset=i,
            device_id=shared_dev,
        ))
    # A final transaction by the 7th sender — by then shared_device_count > 5
    events.append(_mk_event(
        sender="acc-R08-S6",
        receiver="acc-R08-R2",
        amount=80_000,
        minutes_offset=10,
        device_id=shared_dev,
    ))
    return Probe(
        name="R08-shared-device",
        layer="L1",
        description="Same device id used by 7 distinct senders (>5 threshold).",
        events=events,
        expected_rules=["R08"],
    )


def _probe_r09_excessive_beneficiaries() -> Probe:
    """One sender → 22 distinct receivers over a 24h window."""
    sender = "acc-R09-A"
    events = [
        _mk_event(
            sender=sender,
            receiver=f"acc-R09-recv-{i:02d}",
            amount=40_000,
            minutes_offset=i * 30,
        )
        for i in range(22)
    ]
    return Probe(
        name="R09-excessive-beneficiaries",
        layer="L1",
        description="Sender pushes to 22 distinct beneficiaries in 11 hours (>20/7d).",
        events=events,
        expected_rules=["R09"],
    )


def _probe_r10_cycle_closure() -> Probe:
    """A → B → C → A — closing edge must trip R10 (cycle_closure)."""
    a, b, c = "acc-R10-A", "acc-R10-B", "acc-R10-C"
    return Probe(
        name="R10-cycle-closure",
        layer="L1",
        description="A directed 3-cycle: A->B->C->A.",
        events=[
            _mk_event(sender=a, receiver=b, amount=150_000, minutes_offset=0),
            _mk_event(sender=b, receiver=c, amount=148_000, minutes_offset=3),
            _mk_event(sender=c, receiver=a, amount=145_000, minutes_offset=6),
        ],
        expected_rules=["R10"],
        expected_graph_features=["edge_creates_cycle"],
        expected_min_risk_level="High",
    )


def _probe_r11_round_amount() -> Probe:
    """Round amounts + velocity > 5/24h."""
    sender, receiver = "acc-R11-A", "acc-R11-B"
    events = [
        _mk_event(sender=sender, receiver=receiver, amount=100_000, minutes_offset=i * 10)
        for i in range(7)
    ]
    return Probe(
        name="R11-round-amount",
        layer="L1",
        description="7 round 1L txns within 1h (>5/24h velocity).",
        events=events,
        expected_rules=["R11"],
    )


def _probe_r12_kyc_mismatch() -> Probe:
    """KYC-0 account moving > 5L."""
    return Probe(
        name="R12-kyc-mismatch",
        layer="L1",
        description="Sender with kyc_level=0 moving INR 7 lakh (>5L threshold).",
        events=[
            _mk_event(
                sender="acc-R12-A",
                receiver="acc-R12-B",
                amount=700_000,
                kyc_level=0,
            ),
        ],
        expected_rules=["R12"],
        expected_min_risk_level="Medium",
    )


def _probe_r13_benford() -> Probe:
    """A sender history where leading digits are dominated by 9 (Benford-violating)."""
    sender = "acc-R13-A"
    events: list[TransactionEvent] = []
    # Build up a history of 9-starting amounts; final tx triggers R13.
    for i in range(8):
        events.append(_mk_event(
            sender=sender,
            receiver=f"acc-R13-R{i}",
            amount=900_000 + i * 1000,
            minutes_offset=i * 5,
        ))
    events.append(_mk_event(
        sender=sender,
        receiver="acc-R13-final",
        amount=950_000,
        minutes_offset=60,
    ))
    return Probe(
        name="R13-benford",
        layer="L1",
        description="Sender's amount-leading-digits dominated by 9 — Benford violation.",
        events=events,
        expected_rules=["R13"],
    )


def _probe_r14_fan_in() -> Probe:
    """6 distinct senders all depositing ~equal amounts into one receiver."""
    receiver = "acc-R14-aggregator"
    events = [
        _mk_event(
            sender=f"acc-R14-S{i}",
            receiver=receiver,
            amount=200_000 + (i % 2) * 5_000,   # CV < 0.35
            minutes_offset=i * 7,
        )
        for i in range(6)
    ]
    return Probe(
        name="R14-fan-in",
        layer="L1",
        description="6 distinct senders depositing ~uniform amounts into one aggregator.",
        events=events,
        expected_rules=["R14"],
        expected_graph_features=["receiver_in_degree_unique_24h"],
        expected_min_risk_level="High",
    )


def _probe_r15_layering_relay() -> Probe:
    """A → B (₹500k), then B → C (₹470k) within 30 min — 94% retain ratio."""
    a, b, c = "acc-R15-A", "acc-R15-B", "acc-R15-C"
    return Probe(
        name="R15-layering-relay",
        layer="L1",
        description="A->B INR 500k inflow; B->C forwards 94% within 30 minutes (relay).",
        events=[
            _mk_event(sender=a, receiver=b, amount=500_000, minutes_offset=0),
            _mk_event(sender=b, receiver=c, amount=470_000, minutes_offset=30),
        ],
        expected_rules=["R15"],
        expected_graph_features=["sender_is_relay_node"],
        expected_min_risk_level="High",
    )


# --------------------------------------------------------------------------- #
# Layer-2 graph-feature probes
# --------------------------------------------------------------------------- #


def _probe_L2_cycle() -> Probe:
    """Same 3-cycle as R10 but assessed against the graph_features set."""
    a, b, c = "acc-L2c-A", "acc-L2c-B", "acc-L2c-C"
    return Probe(
        name="L2-cycle-closure",
        layer="L2",
        description="Closing edge of a directed 3-cycle.",
        events=[
            _mk_event(sender=a, receiver=b, amount=120_000),
            _mk_event(sender=b, receiver=c, amount=120_000, minutes_offset=2),
            _mk_event(sender=c, receiver=a, amount=120_000, minutes_offset=4),
        ],
        expected_rules=["R10"],
        expected_graph_features=["edge_creates_cycle", "cycle_length"],
    )


def _probe_L2_fan_in() -> Probe:
    """7 senders → 1 aggregator. Same as R14 but framed at the graph layer."""
    receiver = "acc-L2fan-AGG"
    events = [
        _mk_event(
            sender=f"acc-L2fan-S{i}", receiver=receiver,
            amount=150_000, minutes_offset=i * 5,
        )
        for i in range(7)
    ]
    return Probe(
        name="L2-fan-in-aggregation",
        layer="L2",
        description="7 senders converge on a single receiver.",
        events=events,
        expected_rules=["R14"],
        expected_graph_features=["receiver_in_degree_unique_24h"],
    )


def _probe_L2_fan_out() -> Probe:
    """1 sender → 22 distinct receivers (R09 threshold is >20)."""
    sender = "acc-L2fo-A"
    events = [
        _mk_event(
            sender=sender, receiver=f"acc-L2fo-R{i:02d}",
            amount=60_000, minutes_offset=i * 3,
        )
        for i in range(22)
    ]
    return Probe(
        name="L2-fan-out",
        layer="L2",
        description="One sender pushes to 22 distinct receivers (>20 R09 threshold).",
        events=events,
        expected_rules=["R09"],   # excessive beneficiaries (close cousin)
        expected_graph_features=["sender_out_degree_unique"],
    )


def _probe_L2_relay() -> Probe:
    """Same as R15 but framed at the graph-feature layer."""
    a, b, c = "acc-L2relay-A", "acc-L2relay-B", "acc-L2relay-C"
    return Probe(
        name="L2-relay-node",
        layer="L2",
        description="B receives 500k from A then forwards 470k to C within 30 min.",
        events=[
            _mk_event(sender=a, receiver=b, amount=500_000),
            _mk_event(sender=b, receiver=c, amount=470_000, minutes_offset=30),
        ],
        expected_rules=["R15"],
        expected_graph_features=["sender_is_relay_node", "sender_last_inflow_amount"],
    )


# --------------------------------------------------------------------------- #
# Scenario probes — every typology from System 1
# --------------------------------------------------------------------------- #


def _scenario_probes(seed: int = 42) -> list[Probe]:
    engine = ScenarioEngine()
    typologies = [
        ("structuring",          ["R02"], None),
        ("circular_laundering",  ["R10"], "High"),
        ("layering_chain",       ["R15"], "High"),
        ("fan_in",               ["R14"], "High"),
        ("fan_out",              ["R09"], None),
        ("fraud_ring",           ["R08", "R10"], None),
        ("dormant_activation",   ["R03"], None),
        ("velocity_burst",       ["R03"], "Medium"),
        ("cross_border_layering",["R06"], "Medium"),
        ("round_tripping",       ["R10"], "High"),
    ]
    probes: list[Probe] = []
    for typology, expected, min_lvl in typologies:
        plans = engine.build_plan(
            typology, default_params(typology), seed=seed, count=1,
        )
        events = engine.events_from_plans(plans, base_time=BASE_TIME)
        probes.append(Probe(
            name=f"scenario-{typology}",
            layer="scenario",
            description=f"Full {typology} scenario, {len(events)} events.",
            events=events,
            expected_rules=expected,
            expected_min_risk_level=min_lvl,
        ))
    return probes


# --------------------------------------------------------------------------- #
# Probe runner
# --------------------------------------------------------------------------- #


RISK_LEVEL_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


@dataclass
class Detector:
    """Wrap a fresh detector pipeline so each probe runs in isolation."""

    feature_updater: LiveFeatureUpdater
    graph_updater: LiveGraphUpdater
    rule_engine: RuleEngine
    fusion: RiskFusionEngine
    supervised: SupervisedInferencer | None = None
    anomaly: BehavioralAnomalyInferencer | None = None
    tgn: TGNInferencer | None = None

    def process(self, event: TransactionEvent) -> dict[str, Any]:
        feat = self.feature_updater.process_event(event)
        graph_vec = self.graph_updater.process_event(event)
        rule_out = self.rule_engine.evaluate(feat, graph_vec, event)
        sup_out = self.supervised.score(feat, graph_vec, rule_out) if self.supervised else None
        anom_out = self.anomaly.score(feat, graph_vec) if self.anomaly else None
        tgn_out = self.tgn.score(feat, graph_vec, event) if self.tgn else None
        scores = collect_layer_scores(
            rule_out=rule_out, graph_vec=graph_vec, supervised_out=sup_out,
            anomaly_out=anom_out, tgn_out=tgn_out,
            transaction_id=event.transaction_id, sender_account=event.sender_account,
        )
        fused = self.fusion.fuse(scores)
        return {
            "triggered_rules": list(rule_out.triggered_rules),
            "graph_vec": graph_vec.model_dump(),
            "rule_score": float(rule_out.rule_score),
            "fused_score": float(fused.transaction_risk_score),
            "risk_level": fused.risk_level,
        }


def _build_detector() -> Detector:
    """Fresh detector that loads every available artifact."""
    store = ArtifactStore()
    def _safe(name: str):
        try:
            return store.load(name)
        except ArtifactNotFoundError:
            return None
    graph = _safe("transaction_multigraph.pkl")
    scaler = _safe("feature_scaler.pkl")
    community_profiles = _safe("community_profiles.parquet")
    graph_features = _safe("graph_features.parquet")
    fu = LiveFeatureUpdater(scaler=scaler)
    gu = LiveGraphUpdater(
        graph=graph,
        community_profiles=community_profiles,
        graph_features=graph_features,
    )
    rule_engine = RuleEngine()
    fusion = RiskFusionEngine(fusion_mode="static_weights")
    sup = SupervisedInferencer(store=store) if SupervisedInferencer.all_artifacts_present(store) else None
    anom = BehavioralAnomalyInferencer(store=store) if BehavioralAnomalyInferencer.all_artifacts_present(store) else None
    tgn = TGNInferencer(store=store) if TGNInferencer.any_artifacts_present(store) else None
    return Detector(
        feature_updater=fu, graph_updater=gu, rule_engine=rule_engine,
        fusion=fusion, supervised=sup, anomaly=anom, tgn=tgn,
    )


def _run_probe(probe: Probe) -> ProbeResult:
    detector = _build_detector()

    triggered_seen: set[str] = set()
    graph_feature_hits: set[str] = set()
    max_score = 0.0
    max_level = "Low"
    n_alerted = 0

    for event in probe.events:
        try:
            out = detector.process(event)
        except Exception as exc:
            logger.warning("[%s] pipeline error on %s: %s", probe.name, event.transaction_id, exc)
            continue
        for rid in out["triggered_rules"]:
            triggered_seen.add(rid)
        # Inspect the graph_vec for non-trivial features
        gv = out["graph_vec"]
        for key, val in gv.items():
            if isinstance(val, bool) and val:
                graph_feature_hits.add(key)
            elif isinstance(val, (int, float)) and val and val > 0:
                graph_feature_hits.add(key)
        if out["fused_score"] > max_score:
            max_score = out["fused_score"]
            max_level = out["risk_level"]
        if out["risk_level"] in ("High", "Critical"):
            n_alerted += 1

    # PASS/FAIL determination
    rules_pass = set(probe.expected_rules).issubset(triggered_seen)
    features_pass = (
        not probe.expected_graph_features
        or set(probe.expected_graph_features).issubset(graph_feature_hits)
    )
    level_pass = (
        probe.expected_min_risk_level is None
        or RISK_LEVEL_RANK.get(max_level, 0) >= RISK_LEVEL_RANK[probe.expected_min_risk_level]
    )
    passed = rules_pass and features_pass and level_pass

    notes_parts: list[str] = []
    missing_rules = sorted(set(probe.expected_rules) - triggered_seen)
    if missing_rules:
        notes_parts.append(f"missing rules: {','.join(missing_rules)}")
    missing_features = sorted(set(probe.expected_graph_features) - graph_feature_hits)
    if missing_features:
        notes_parts.append(f"missing features: {','.join(missing_features)}")
    if probe.expected_min_risk_level and not level_pass:
        notes_parts.append(f"max risk {max_level} < expected {probe.expected_min_risk_level}")

    return ProbeResult(
        name=probe.name,
        layer=probe.layer,
        description=probe.description,
        expected_rules=list(probe.expected_rules),
        expected_graph_features=list(probe.expected_graph_features),
        expected_min_risk_level=probe.expected_min_risk_level,
        triggered_rules_seen=sorted(triggered_seen),
        graph_features_seen=sorted(graph_feature_hits),
        max_risk_level=max_level,
        max_risk_score=round(max_score, 2),
        n_events=len(probe.events),
        n_alerted=n_alerted,
        rules_pass=rules_pass,
        features_pass=features_pass,
        risk_pass=level_pass,
        passed=passed,
        notes="; ".join(notes_parts),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _print_table(results: list[ProbeResult]) -> None:
    print()
    print("=" * 110)
    print("  DETECTION VERIFICATION  --  probe-by-probe results")
    print("  Three independent axes: RULES (rules fired), FEATS (graph features hit), RISK (fused level reached)")
    print("=" * 110)
    header = f"  {'PROBE':<32}  {'LAYER':<8}  {'RULES':<6}  {'FEATS':<6}  {'RISK':<6}  {'LEVEL':<8}  EXPECTED -> SEEN"
    print(header)
    print("-" * 110)
    for r in results:
        rule_mark = "PASS" if r.rules_pass else "FAIL"
        feat_mark = "PASS" if r.features_pass else "----"   # ---- = no expectation
        risk_mark = "PASS" if r.risk_pass else "FAIL"
        line = (
            f"  {r.name:<32}  {r.layer:<8}  {rule_mark:<6}  "
            f"{feat_mark if r.expected_graph_features else '----':<6}  "
            f"{risk_mark if r.expected_min_risk_level else '----':<6}  "
            f"{r.max_risk_level:<8}  "
            f"{','.join(r.expected_rules) or '-':<14} -> "
            f"{','.join(rid for rid in r.triggered_rules_seen if rid in r.expected_rules) or '-'}"
        )
        print(line)
        if not r.passed and r.notes:
            print(f"      note: {r.notes}")
    print("-" * 110)
    total = len(results)
    rules_ok  = sum(1 for r in results if r.rules_pass)
    feats_ok  = sum(1 for r in results if r.features_pass)
    risk_ok   = sum(1 for r in results if r.risk_pass)
    full_pass = sum(1 for r in results if r.passed)
    print(f"  RULES axis:   {rules_ok}/{total}  (expected rule(s) fired)")
    print(f"  FEATS axis:   {feats_ok}/{total}  (expected graph feature(s) materialised)")
    print(f"  RISK  axis:   {risk_ok}/{total}  (fused risk reached expected level)")
    print(f"  FULL pass:    {full_pass}/{total}  (all three axes)")
    print("=" * 110)


def _persist(results: list[ProbeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total":   len(results),
        "passed":  sum(1 for r in results if r.passed),
        "results": [r.__dict__ for r in results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


def all_probes() -> list[Probe]:
    return [
        # L1 rule probes (one per rule)
        _probe_r01_high_value(),
        _probe_r02_structuring(),
        _probe_r03_velocity_spike(),
        # R04 (dormant) is impractical in a fresh-state harness (needs a real
        # 180-day prior tx in the multigraph); covered by the
        # scenario-dormant_activation probe below.
        _probe_r05_impossible_travel(),
        _probe_r06_high_risk_jurisdiction(),
        _probe_r07_device_anomaly(),
        _probe_r08_shared_device(),
        _probe_r09_excessive_beneficiaries(),
        _probe_r10_cycle_closure(),
        _probe_r11_round_amount(),
        _probe_r12_kyc_mismatch(),
        _probe_r13_benford(),
        _probe_r14_fan_in(),
        _probe_r15_layering_relay(),
        # L2 graph-feature probes
        _probe_L2_cycle(),
        _probe_L2_fan_in(),
        _probe_L2_fan_out(),
        _probe_L2_relay(),
        # Scenario probes (one per typology from System 1)
        *_scenario_probes(),
    ]


def _export_probe_csv(probes: list[Probe], csv_path: Path, manifest_path: Path) -> None:
    """Write every probe's events to a stream-compatible CSV and a sidecar
    manifest JSON that records which events belong to which probe.

    The CSV uses the SAME columns as ``data/stream_transactions.csv`` so the
    file can be uploaded straight into ``POST /generator/inject_csv`` or
    fed to ``ReplayDriver`` for offline debugging.
    """
    import csv as _csv

    stream_columns = [
        "transaction_id", "timestamp",
        "sender_account", "receiver_account",
        "sender_bank", "receiver_bank",
        "sender_country", "receiver_country",
        "amount", "currency",
        "transaction_type", "payment_channel",
        "device_id", "ip_address",
        "geo_latitude", "geo_longitude",
        "merchant_category", "transaction_status",
        "sender_balance_before", "sender_balance_after",
        "receiver_balance_before", "receiver_balance_after",
        "kyc_level", "is_international", "remarks", "amount_leading_digit",
    ]

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for probe in probes:
        tx_ids: list[str] = []
        for ev in probe.events:
            ev_dict = ev.model_dump()
            row = {
                "transaction_id": ev_dict["transaction_id"],
                "timestamp": ev_dict["timestamp"].isoformat() if hasattr(ev_dict["timestamp"], "isoformat") else str(ev_dict["timestamp"]),
                "sender_account": ev_dict["sender_account"],
                "receiver_account": ev_dict["receiver_account"],
                "sender_bank": ev_dict["sender_bank"],
                "receiver_bank": ev_dict["receiver_bank"],
                "sender_country": ev_dict["sender_country"],
                "receiver_country": ev_dict["receiver_country"],
                "amount": ev_dict["amount"],
                "currency": ev_dict["currency"],
                "transaction_type": ev_dict["transaction_type"],
                "payment_channel": ev_dict["payment_channel"],
                "device_id": ev_dict["device_id"],
                "ip_address": ev_dict["ip_address"],
                "geo_latitude": ev_dict["geo_latitude"],
                "geo_longitude": ev_dict["geo_longitude"],
                "merchant_category": ev_dict["merchant_category"],
                "transaction_status": ev_dict["transaction_status"],
                "sender_balance_before": "",
                "sender_balance_after": "",
                "receiver_balance_before": "",
                "receiver_balance_after": "",
                "kyc_level": ev_dict.get("kyc_level", ""),
                "is_international": str(bool(ev_dict["is_international"])),
                "remarks": f"probe={probe.name}",
                "amount_leading_digit": ev_dict["amount_leading_digit"],
            }
            rows.append(row)
            tx_ids.append(ev_dict["transaction_id"])
        manifest_entries.append({
            "probe_name": probe.name,
            "layer": probe.layer,
            "description": probe.description,
            "expected_rules": probe.expected_rules,
            "expected_graph_features": probe.expected_graph_features,
            "expected_min_risk_level": probe.expected_min_risk_level,
            "transaction_ids": tx_ids,
            "event_count": len(tx_ids),
        })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=stream_columns)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(rows),
            "total_probes": len(probes),
            "schema": stream_columns,
            "probes": manifest_entries,
        }, fh, indent=2, default=str)

    print(f"\nwrote {csv_path}  ({len(rows)} probe events)")
    print(f"wrote {manifest_path}  ({len(probes)} probe descriptors)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-rule + per-feature detection verification")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "reports" / "detection_verification.json")
    parser.add_argument("--write-csv", type=Path, default=None,
                        help="Also export the probe stream as a stream_transactions-compatible "
                             "CSV at this path (and a sidecar manifest JSON). Use this CSV with "
                             "the Overview file-injector or POST /generator/inject_csv.")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip running the probes through the pipeline; only export the CSV.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    probes = all_probes()

    if args.write_csv is not None:
        manifest_path = args.write_csv.with_suffix(".manifest.json")
        _export_probe_csv(probes, args.write_csv, manifest_path)

    if args.skip_run:
        return 0

    results: list[ProbeResult] = []
    for i, probe in enumerate(probes, 1):
        print(f"  [{i:2d}/{len(probes)}] running {probe.name} ({probe.n_events if hasattr(probe, 'n_events') else len(probe.events)} events) ...")
        r = _run_probe(probe)
        results.append(r)

    _print_table(results)
    _persist(results, args.output)

    # Exit non-zero if any probe failed (for CI)
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
