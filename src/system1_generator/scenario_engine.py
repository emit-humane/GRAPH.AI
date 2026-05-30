"""Scenario Studio — pattern-level controllable transaction generator.

One engine, two adapters: the same ``ScenarioEngine.build_plan(...)`` is the
source of truth for both the **live-inject** path (events queued into D0's
replay stream) and the **export** path (labeled stream CSV + ground truth).

Each typology declares its tunable parameter schema (matching
``docs/scenario_studio_reference.jsx`` exactly) and which Layer-1 rules the
generated edges target. The 10 typologies reuse the realism patches from
Session 1 wherever appropriate — particularly the per-ring shared infra
pool (one shared device + IP subnet + geo centroid) and the structured
timestamp spread that prevents back-to-back bursts.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np

from src.system2_detection.shared.schemas import TransactionEvent

# --------------------------------------------------------------------------- #
# Geographic + bank reference data (mirrors the System-1 generator)
# --------------------------------------------------------------------------- #

COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "IN": (20.59, 78.96),
    "US": (37.09, -95.71),
    "AE": (23.42, 53.85),
    "SG": (1.35, 103.81),
    "GB": (55.37, -3.43),
    "CN": (35.86, 104.19),
    "MU": (-20.35, 57.55),
    "NG": (9.08, 8.67),
    "PK": (30.37, 69.34),
    "CH": (46.81, 8.22),
}

BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "YES", "BOB"]
HIGH_RISK_COUNTRIES = ["AE", "MU", "CN", "NG", "PK"]

TX_TYPES = ("NEFT", "RTGS", "IMPS", "Wire", "UPI", "Card")
PAYMENT_CHANNELS = ("Mobile", "Web", "ATM", "Branch")


# --------------------------------------------------------------------------- #
# Param schemas — straight from docs/scenario_studio_reference.jsx TYPOLOGIES
# --------------------------------------------------------------------------- #


@dataclass
class ParamSpec:
    label: str
    min: float
    max: float
    default: float
    step: float
    integer: bool = True


@dataclass
class TypologySpec:
    key: str
    label: str
    pattern: str
    blurb: str
    target_rules: tuple[str, ...]
    params: dict[str, ParamSpec]
    color: str = "#d99a2b"


# Centralise the catalog so the API can expose it whole.
TYPOLOGIES: dict[str, TypologySpec] = {
    "structuring": TypologySpec(
        key="structuring",
        label="Structuring",
        pattern="Linear A→B, repeated",
        blurb="Repeated sub-threshold transfers to stay under reporting limits.",
        target_rules=("R02", "R11", "R13"),
        color="#d99a2b",
        params={
            "count":     ParamSpec("Sub-threshold transfers", 5, 15, 8, 1),
            "amount_lo": ParamSpec("Amount floor (₹)", 700_000, 950_000, 850_000, 10_000),
            "amount_hi": ParamSpec("Amount ceiling (₹)", 950_000, 999_999, 999_000, 1_000),
            "window_h":  ParamSpec("Window (hours)", 1, 48, 24, 1),
        },
    ),
    "circular_laundering": TypologySpec(
        key="circular_laundering",
        label="Circular Laundering",
        pattern="Directed cycle",
        blurb="Round-trip cycle A→B→C→A that returns funds to origin.",
        target_rules=("R10", "R03"),
        color="#e23d6e",
        params={
            "ring_size": ParamSpec("Ring size (accounts)", 3, 6, 4, 1),
            "amount":    ParamSpec("Cycle amount (₹)", 100_000, 2_000_000, 500_000, 50_000),
            "window_h":  ParamSpec("Cycle window (hours)", 6, 48, 48, 1),
        },
    ),
    "layering_chain": TypologySpec(
        key="layering_chain",
        label="Layering Chain",
        pattern="Linear chain",
        blurb="Multi-hop chain A→B→C→…→Z, each hop changing bank/country. Each hop forwards 85–100% within 2h — wins R15.",
        target_rules=("R03", "R06", "R15"),
        color="#e23d6e",
        params={
            "hops":   ParamSpec("Hops", 3, 8, 5, 1),
            "amount": ParamSpec("Initial amount (₹)", 100_000, 5_000_000, 1_000_000, 100_000),
            "decay":  ParamSpec("Value retained per hop (%)", 85, 100, 95, 1),
            "gap_h":  ParamSpec("Avg gap between hops (h)", 1, 2, 1, 1),
        },
    ),
    "fan_in": TypologySpec(
        key="fan_in",
        label="Fan-In",
        pattern="Star, target at center",
        blurb="Many sources funnel into one aggregator with low CV — wins R14.",
        target_rules=("R09", "R14"),
        color="#d99a2b",
        params={
            "sources":  ParamSpec("Source accounts", 5, 15, 8, 1),
            "amount":   ParamSpec("Per-source amount (₹)", 10_000, 500_000, 80_000, 10_000),
            "cv":       ParamSpec("Amount variance (CV)", 0.0, 1.5, 0.15, 0.05, integer=False),
            "window_h": ParamSpec("Collection window (h)", 1, 24, 12, 1),
        },
    ),
    "fan_out": TypologySpec(
        key="fan_out",
        label="Fan-Out",
        pattern="Star, source at center",
        blurb="One distributor splits funds across many targets.",
        target_rules=("R09", "R02"),
        color="#d99a2b",
        params={
            "targets":  ParamSpec("Target accounts", 5, 15, 8, 1),
            "amount":   ParamSpec("Per-target amount (₹)", 10_000, 500_000, 70_000, 10_000),
            "window_h": ParamSpec("Distribution window (h)", 1, 24, 6, 1),
        },
    ),
    "fraud_ring": TypologySpec(
        key="fraud_ring",
        label="Fraud Ring",
        pattern="Dense clique (density > 0.6)",
        blurb="Dense clique of mutually transacting accounts with shared infra.",
        target_rules=("R08", "R10"),
        color="#e23d6e",
        params={
            "members": ParamSpec("Ring members", 4, 10, 6, 1),
            "density": ParamSpec("Edge density", 0.6, 1.0, 0.7, 0.05, integer=False),
            "amount":  ParamSpec("Avg edge amount (₹)", 50_000, 1_000_000, 200_000, 50_000),
        },
    ),
    "dormant_activation": TypologySpec(
        key="dormant_activation",
        label="Dormant Activation",
        pattern="Single node burst",
        blurb="Long-silent account reactivates and bursts funds out.",
        target_rules=("R04", "R07", "R03"),
        color="#e23d6e",
        params={
            "silent_days": ParamSpec("Silent period (days)", 180, 720, 200, 10),
            "burst_count": ParamSpec("Burst transactions", 10, 30, 12, 1),
            "amount":      ParamSpec("Inbound amount (₹)", 100_000, 3_000_000, 800_000, 100_000),
            "disburse":    ParamSpec("Disbursement targets", 3, 10, 4, 1),
        },
    ),
    "velocity_burst": TypologySpec(
        key="velocity_burst",
        label="Velocity Burst",
        pattern="Temporal cluster",
        blurb="Intense cluster of transactions in a short window.",
        target_rules=("R03",),
        color="#d99a2b",
        params={
            "count":      ParamSpec("Transactions", 15, 60, 20, 1),
            "window_min": ParamSpec("Window (minutes)", 10, 60, 60, 5),
            "amount":     ParamSpec("Avg amount (₹)", 10_000, 500_000, 60_000, 10_000),
        },
    ),
    "cross_border_layering": TypologySpec(
        key="cross_border_layering",
        label="Cross-Border Layering",
        pattern="Chain via high-risk countries",
        blurb="Chain routed through high-risk jurisdictions.",
        target_rules=("R06", "R03"),
        color="#e23d6e",
        params={
            "hops":   ParamSpec("International hops", 2, 6, 3, 1),
            "amount": ParamSpec("Amount (₹)", 500_000, 5_000_000, 1_500_000, 100_000),
            "gap_h":  ParamSpec("Settlement gap (h)", 6, 24, 12, 1),
        },
    ),
    "round_tripping": TypologySpec(
        key="round_tripping",
        label="Round-Tripping",
        pattern="Directed cycle via international",
        blurb="International round-trip A(IN)→B(AE)→C(SG)→A(IN).",
        target_rules=("R10", "R06"),
        color="#e23d6e",
        params={
            "hops":   ParamSpec("International hops", 3, 5, 3, 1),
            "amount": ParamSpec("Amount (₹)", 500_000, 5_000_000, 2_000_000, 100_000),
            "gap_h":  ParamSpec("Settlement gap (h)", 12, 48, 24, 1),
        },
    ),
}


def typologies_catalog() -> list[dict[str, Any]]:
    """Return the catalog in the same JSON shape the studio frontend reads."""
    out: list[dict[str, Any]] = []
    for spec in TYPOLOGIES.values():
        out.append({
            "key": spec.key,
            "label": spec.label,
            "pattern": spec.pattern,
            "blurb": spec.blurb,
            "target_rules": list(spec.target_rules),
            "color": spec.color,
            "params": {
                name: {
                    "label": p.label,
                    "min": p.min, "max": p.max,
                    "default": p.default,
                    "step": p.step,
                    "integer": p.integer,
                }
                for name, p in spec.params.items()
            },
        })
    return out


def default_params(typology: str) -> dict[str, float]:
    spec = TYPOLOGIES.get(typology)
    if spec is None:
        raise KeyError(f"unknown typology {typology}")
    return {name: p.default for name, p in spec.params.items()}


# --------------------------------------------------------------------------- #
# Plan + engine
# --------------------------------------------------------------------------- #


@dataclass
class PlannedEdge:
    """A single edge in the planned scenario (pre-Event materialisation)."""
    sender: str
    receiver: str
    amount: float
    timestamp_offset_s: float
    sender_country: str = "IN"
    receiver_country: str = "IN"
    sender_bank: str = "HDFC"
    receiver_bank: str = "HDFC"
    transaction_type: str = "NEFT"
    payment_channel: str = "Web"
    device_id: str = ""
    ip_address: str = ""
    geo_lat: float = 20.59
    geo_lon: float = 78.96
    leading_digit: int = 1
    is_international: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender, "receiver": self.receiver,
            "amount": self.amount,
            "timestamp_offset_s": self.timestamp_offset_s,
            "sender_country": self.sender_country,
            "receiver_country": self.receiver_country,
            "is_international": self.is_international,
            "sender_bank": self.sender_bank,
            "receiver_bank": self.receiver_bank,
            "transaction_type": self.transaction_type,
            "note": self.note,
        }


@dataclass
class ScenarioPlan:
    typology: str
    ring_id: str
    members: list[str]
    edges: list[PlannedEdge]
    target_rules: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_value(self) -> float:
        return float(sum(e.amount for e in self.edges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "typology": self.typology,
            "ring_id": self.ring_id,
            "members": self.members,
            "target_rules": list(self.target_rules),
            "edges": [e.to_dict() for e in self.edges],
            "summary": {
                "edge_count": len(self.edges),
                "account_count": len(self.members),
                "total_value": self.total_value(),
            },
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class ScenarioEngine:
    """Builds ScenarioPlans + materialises them into TransactionEvents.

    The same engine drives:
      * Live-inject path (events queued into the running ReplayDriver stream)
      * Export path (labeled stream_transactions.csv + ground truth)
      * Preview path (returns the plan to the studio UI without injecting)
    """

    # ------------------------------ build ---------------------------------- #

    def build_plan(
        self,
        typology: str,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        count: int = 1,
    ) -> list[ScenarioPlan]:
        if typology not in TYPOLOGIES:
            raise KeyError(f"unknown typology {typology!r}")
        spec = TYPOLOGIES[typology]
        full_params = self._resolve_params(spec, params or {})

        plans: list[ScenarioPlan] = []
        for i in range(max(1, int(count))):
            rng = np.random.default_rng(int(seed) + i * 7919)
            builder = _BUILDERS[typology]
            plans.append(builder(rng, full_params, spec))
        return plans

    def _resolve_params(self, spec: TypologySpec, p: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, ps in spec.params.items():
            val = p.get(name, ps.default)
            val = max(ps.min, min(ps.max, val))
            if ps.integer:
                val = int(round(val))
            out[name] = val
        return out

    # ------------------------------ materialise ---------------------------- #

    def events_from_plans(
        self,
        plans: list[ScenarioPlan],
        *,
        base_time: datetime | None = None,
    ) -> list[TransactionEvent]:
        """Convert ScenarioPlans into TransactionEvents (the type D0 emits).

        ``base_time`` becomes the t0 of the first edge. Defaults to "now".
        """
        t0 = base_time or datetime.now(timezone.utc)
        events: list[TransactionEvent] = []
        for plan in plans:
            for edge in plan.edges:
                events.append(self._edge_to_event(edge, plan, t0))
        # Ensure timestamps strictly increase even across plans
        events.sort(key=lambda e: e.timestamp)
        return events

    @staticmethod
    def _edge_to_event(edge: PlannedEdge, plan: ScenarioPlan, t0: datetime) -> TransactionEvent:
        ts = t0 + timedelta(seconds=float(edge.timestamp_offset_s))
        return TransactionEvent(
            transaction_id=str(uuid.uuid4()),
            timestamp=ts,
            sender_account=edge.sender,
            receiver_account=edge.receiver,
            sender_bank=edge.sender_bank,
            receiver_bank=edge.receiver_bank,
            sender_country=edge.sender_country,
            receiver_country=edge.receiver_country,
            amount=float(edge.amount),
            currency="INR",
            transaction_type=edge.transaction_type,
            payment_channel=edge.payment_channel,
            device_id=edge.device_id or f"dev-{plan.ring_id[:8]}",
            ip_address=edge.ip_address or "203.0.113.10",
            geo_latitude=edge.geo_lat,
            geo_longitude=edge.geo_lon,
            merchant_category="Other",
            transaction_status="Success",
            kyc_level=2,
            is_international=bool(edge.is_international),
            amount_leading_digit=_leading_digit(edge.amount),
        )


def _leading_digit(amount: float) -> int:
    if amount <= 0:
        return 1
    s = f"{int(amount)}"
    return int(s[0]) if s and s[0].isdigit() and s[0] != "0" else 1


# --------------------------------------------------------------------------- #
# Per-typology plan builders
# --------------------------------------------------------------------------- #


def _new_account(rng: np.random.Generator, prefix: str = "STUDIO") -> str:
    """Deterministic-ish account id (still UUID-looking)."""
    return f"{prefix}-{rng.bytes(8).hex()}"


def _shared_infra(rng: np.random.Generator) -> tuple[str, str, float, float]:
    """One device + IP + geo for the whole ring (the Patch 4 D1 pool)."""
    device = f"dev-{rng.bytes(6).hex()}"
    ip = f"203.0.113.{int(rng.integers(2, 250))}"
    lat = 20.59 + float(rng.normal(0, 1.0))
    lon = 78.96 + float(rng.normal(0, 1.0))
    return device, ip, lat, lon


def _structuring(rng, p, spec) -> ScenarioPlan:
    n = int(p["count"])
    span_s = float(p["window_h"]) * 3600.0
    mean_gap = span_s / (n + 1)
    sigma_gap = 0.3 * mean_gap
    sender, receiver = _new_account(rng, "STR-S"), _new_account(rng, "STR-R")
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    for i in range(n):
        cursor += max(60.0, float(rng.normal(mean_gap, sigma_gap)))
        amt = float(rng.uniform(p["amount_lo"], p["amount_hi"]))
        edges.append(PlannedEdge(
            sender=sender, receiver=receiver, amount=amt, timestamp_offset_s=cursor,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="NEFT",
        ))
    return ScenarioPlan(
        typology="structuring", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=[sender, receiver], edges=edges, target_rules=list(spec.target_rules),
        metadata={"span_hours": p["window_h"], "amount_floor": p["amount_lo"]},
    )


def _circular_laundering(rng, p, spec) -> ScenarioPlan:
    ring = int(p["ring_size"])
    span_s = float(p["window_h"]) * 3600.0
    members = [_new_account(rng, f"CYC{i}") for i in range(ring)]
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    amt = float(p["amount"])
    for i in range(ring):
        cursor += max(300.0, float(rng.exponential(span_s / ring)))
        amt *= float(rng.uniform(0.95, 1.0))
        edges.append(PlannedEdge(
            sender=members[i], receiver=members[(i + 1) % ring], amount=amt,
            timestamp_offset_s=cursor,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="RTGS",
        ))
    return ScenarioPlan(
        typology="circular_laundering", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=members, edges=edges, target_rules=list(spec.target_rules),
        metadata={"ring_size": ring, "window_h": p["window_h"]},
    )


def _layering_chain(rng, p, spec) -> ScenarioPlan:
    hops = int(p["hops"])
    decay = float(p["decay"]) / 100.0
    members = [_new_account(rng, f"CH{i}") for i in range(hops + 1)]
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    amt = float(p["amount"])
    countries = ["IN", "AE", "SG", "MU", "GB", "CN", "PK"]
    banks = ["HDFC", "ICICI", "Axis", "SBI", "PNB", "Kotak"]
    for i in range(hops):
        # < 2h gap so R15 (relay) fires
        gap = max(300.0, float(rng.exponential(float(p["gap_h"]) * 3600.0)))
        gap = min(gap, 6900.0)  # cap under 7200 to ensure R15 window
        cursor += gap
        new_amt = amt * decay * float(rng.uniform(0.97, 1.0))
        edges.append(PlannedEdge(
            sender=members[i], receiver=members[i + 1], amount=new_amt,
            timestamp_offset_s=cursor,
            sender_country=countries[i % len(countries)],
            receiver_country=countries[(i + 1) % len(countries)],
            sender_bank=banks[i % len(banks)],
            receiver_bank=banks[(i + 1) % len(banks)],
            is_international=countries[i % len(countries)] != countries[(i + 1) % len(countries)],
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="Wire",
        ))
        amt = new_amt
    return ScenarioPlan(
        typology="layering_chain", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=members, edges=edges, target_rules=list(spec.target_rules),
        metadata={"hops": hops, "decay_pct": p["decay"]},
    )


def _fan_in(rng, p, spec) -> ScenarioPlan:
    n_src = int(p["sources"])
    span_s = float(p["window_h"]) * 3600.0
    aggregator = _new_account(rng, "FAN-IN-AGG")
    sources = [_new_account(rng, f"SRC{i}") for i in range(n_src)]
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    base_amt = float(p["amount"])
    cv = float(p.get("cv", 0.15))
    # Schedule everything within a 23h slot so R14's 24h window catches them
    target_span = min(span_s, 23 * 3600.0)
    for i, src in enumerate(sources):
        cursor = (i + 1) * (target_span / (n_src + 1))
        amt = max(1.0, float(rng.normal(base_amt, base_amt * cv)))
        edges.append(PlannedEdge(
            sender=src, receiver=aggregator, amount=amt,
            timestamp_offset_s=cursor,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="UPI",
        ))
    return ScenarioPlan(
        typology="fan_in", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=[aggregator] + sources, edges=edges, target_rules=list(spec.target_rules),
        metadata={"sources": n_src, "cv": cv, "aggregator": aggregator},
    )


def _fan_out(rng, p, spec) -> ScenarioPlan:
    n_tgt = int(p["targets"])
    span_s = float(p["window_h"]) * 3600.0
    distributor = _new_account(rng, "FAN-OUT-DIST")
    targets = [_new_account(rng, f"TGT{i}") for i in range(n_tgt)]
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    for i, tgt in enumerate(targets):
        cursor = (i + 1) * (span_s / (n_tgt + 1))
        amt = float(rng.uniform(p["amount"] * 0.9, p["amount"] * 1.1))
        edges.append(PlannedEdge(
            sender=distributor, receiver=tgt, amount=amt,
            timestamp_offset_s=cursor,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="IMPS",
        ))
    return ScenarioPlan(
        typology="fan_out", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=[distributor] + targets, edges=edges, target_rules=list(spec.target_rules),
        metadata={"targets": n_tgt, "distributor": distributor},
    )


def _fraud_ring(rng, p, spec) -> ScenarioPlan:
    n = int(p["members"])
    density = float(p["density"])
    members = [_new_account(rng, f"RING{i}") for i in range(n)]
    # 2-3 shared devices for the ring (the dense-clique infra share)
    devices = [f"dev-{rng.bytes(6).hex()}" for _ in range(min(3, max(1, n // 4)))]
    edges: list[PlannedEdge] = []
    max_edges = int(n * (n - 1) * density)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng.shuffle(pairs)
    for k, (i, j) in enumerate(pairs[:max_edges]):
        edges.append(PlannedEdge(
            sender=members[i], receiver=members[j],
            amount=float(rng.uniform(p["amount"] * 0.5, p["amount"] * 1.5)),
            timestamp_offset_s=600.0 + k * 60.0,
            device_id=devices[k % len(devices)],
            ip_address=f"203.0.113.{int(rng.integers(2, 250))}",
            transaction_type="NEFT",
        ))
    return ScenarioPlan(
        typology="fraud_ring", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=members, edges=edges, target_rules=list(spec.target_rules),
        metadata={"members": n, "density": density},
    )


def _dormant_activation(rng, p, spec) -> ScenarioPlan:
    dormant = _new_account(rng, "DORMANT")
    targets = [_new_account(rng, f"DBR{i}") for i in range(int(p["disburse"]))]
    new_device = f"dev-{rng.bytes(8).hex()}"   # the "stolen credentials" signal
    new_ip = f"198.51.100.{int(rng.integers(2, 250))}"
    edges: list[PlannedEdge] = []
    # Step 1: inbound credit (kicks R04 by closing a long-silent gap)
    edges.append(PlannedEdge(
        sender=f"EXT-{_new_account(rng, 'EXT')}", receiver=dormant,
        amount=float(p["amount"]), timestamp_offset_s=0.0,
        device_id="dev-employer", transaction_type="Wire",
        note=f"dormant_period_days={p['silent_days']}",
    ))
    # Step 2: rapid burst disbursements (R03 velocity + R07 new device)
    burst_n = int(p["burst_count"])
    per_target = max(1, burst_n // len(targets))
    base_amt = float(p["amount"]) / max(1, burst_n)
    for i in range(burst_n):
        tgt = targets[i % len(targets)]
        edges.append(PlannedEdge(
            sender=dormant, receiver=tgt, amount=float(rng.uniform(base_amt * 0.8, base_amt * 1.2)),
            timestamp_offset_s=1800.0 + i * 30.0,
            device_id=new_device, ip_address=new_ip,
            transaction_type="IMPS",
            note="burst_disbursement",
        ))
    return ScenarioPlan(
        typology="dormant_activation", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=[dormant] + targets, edges=edges, target_rules=list(spec.target_rules),
        metadata={"dormant": dormant, "silent_days": p["silent_days"]},
    )


def _velocity_burst(rng, p, spec) -> ScenarioPlan:
    n = int(p["count"])
    span_s = float(p["window_min"]) * 60.0
    sender = _new_account(rng, "VB")
    targets = [_new_account(rng, f"VBT{i}") for i in range(min(5, n))]
    device, ip, lat, lon = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    mean_gap = max(1.0, span_s / n)
    for i in range(n):
        cursor += max(2.0, float(rng.exponential(mean_gap)))
        edges.append(PlannedEdge(
            sender=sender, receiver=targets[i % len(targets)],
            amount=float(rng.uniform(p["amount"] * 0.5, p["amount"] * 1.5)),
            timestamp_offset_s=cursor,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lon,
            transaction_type="UPI",
        ))
    return ScenarioPlan(
        typology="velocity_burst", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=[sender] + targets, edges=edges, target_rules=list(spec.target_rules),
        metadata={"count": n, "window_min": p["window_min"]},
    )


def _cross_border_layering(rng, p, spec) -> ScenarioPlan:
    hops = int(p["hops"])
    members = [_new_account(rng, f"XB{i}") for i in range(hops + 1)]
    device, ip, _, _ = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    amt = float(p["amount"])
    countries = HIGH_RISK_COUNTRIES + ["IN"]
    banks = ["HDFC", "ICICI", "Axis", "SBI"]
    for i in range(hops):
        gap = float(p["gap_h"]) * 3600.0
        cursor += max(300.0, float(rng.exponential(gap)))
        sc, rc = countries[i % len(countries)], countries[(i + 1) % len(countries)]
        amt *= float(rng.uniform(0.92, 0.99))
        lat, lng = COUNTRY_CENTROIDS.get(sc, (20.0, 78.0))
        edges.append(PlannedEdge(
            sender=members[i], receiver=members[i + 1], amount=amt,
            timestamp_offset_s=cursor,
            sender_country=sc, receiver_country=rc,
            sender_bank=banks[i % len(banks)],
            receiver_bank=banks[(i + 1) % len(banks)],
            is_international=sc != rc,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lng,
            transaction_type="Wire",
        ))
    return ScenarioPlan(
        typology="cross_border_layering", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=members, edges=edges, target_rules=list(spec.target_rules),
        metadata={"hops": hops},
    )


def _round_tripping(rng, p, spec) -> ScenarioPlan:
    hops = int(p["hops"])
    countries = ["IN", "AE", "SG"][:max(2, min(3, hops))]
    members = [_new_account(rng, f"RT{i}") for i in range(hops)]
    device, ip, _, _ = _shared_infra(rng)
    edges: list[PlannedEdge] = []
    cursor = 0.0
    amt = float(p["amount"])
    for i in range(hops):
        gap = float(p["gap_h"]) * 3600.0
        cursor += max(300.0, float(rng.exponential(gap)))
        sender = members[i]
        receiver = members[(i + 1) % hops]      # closes the ring
        sc = countries[i % len(countries)]
        rc = countries[(i + 1) % len(countries)]
        amt *= float(rng.uniform(0.94, 1.0))
        lat, lng = COUNTRY_CENTROIDS.get(sc, (20.0, 78.0))
        edges.append(PlannedEdge(
            sender=sender, receiver=receiver, amount=amt,
            timestamp_offset_s=cursor,
            sender_country=sc, receiver_country=rc,
            is_international=sc != rc,
            device_id=device, ip_address=ip, geo_lat=lat, geo_lon=lng,
            transaction_type="Wire",
        ))
    return ScenarioPlan(
        typology="round_tripping", ring_id=str(uuid.UUID(bytes=rng.bytes(16))),
        members=members, edges=edges, target_rules=list(spec.target_rules),
        metadata={"hops": hops, "countries": countries},
    )


_BUILDERS: dict[str, Callable[[np.random.Generator, dict[str, Any], TypologySpec], ScenarioPlan]] = {
    "structuring": _structuring,
    "circular_laundering": _circular_laundering,
    "layering_chain": _layering_chain,
    "fan_in": _fan_in,
    "fan_out": _fan_out,
    "fraud_ring": _fraud_ring,
    "dormant_activation": _dormant_activation,
    "velocity_burst": _velocity_burst,
    "cross_border_layering": _cross_border_layering,
    "round_tripping": _round_tripping,
}


__all__ = [
    "ScenarioEngine",
    "ScenarioPlan",
    "PlannedEdge",
    "TypologySpec",
    "ParamSpec",
    "TYPOLOGIES",
    "typologies_catalog",
    "default_params",
    "COUNTRY_CENTROIDS",
]
