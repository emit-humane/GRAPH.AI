"""Scenario Studio engine tests.

Verify every typology produces a structurally-valid plan AND that the two new
typologies (fan_in, layering_chain) honour the R14 / R15 rule trigger windows.
The realism contract: shared infra pool per ring, monotonic timestamps,
positive amounts, and target_rules carried through.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.system1_generator.scenario_engine import (
    ScenarioEngine,
    ScenarioPlan,
    TYPOLOGIES,
    default_params,
    typologies_catalog,
)


# --------------------------------------------------------------------------- #
# Catalog / param shape
# --------------------------------------------------------------------------- #


def test_catalog_lists_all_ten_typologies():
    catalog = typologies_catalog()
    keys = {t["key"] for t in catalog}
    assert keys == {
        "structuring", "circular_laundering", "layering_chain",
        "fan_in", "fan_out", "fraud_ring",
        "dormant_activation", "velocity_burst",
        "cross_border_layering", "round_tripping",
    }
    # Every typology has at least one target rule + at least one param
    for t in catalog:
        assert t["target_rules"], f"{t['key']} has no target rules"
        assert t["params"], f"{t['key']} has no params"


def test_fan_in_targets_r14_and_layering_targets_r15():
    """R14/R15 schema additions must be wired into the typology metadata."""
    assert "R14" in TYPOLOGIES["fan_in"].target_rules
    assert "R15" in TYPOLOGIES["layering_chain"].target_rules


def test_default_params_round_trip():
    for key in TYPOLOGIES:
        p = default_params(key)
        spec = TYPOLOGIES[key]
        assert set(p.keys()) == set(spec.params.keys())
        for name, val in p.items():
            ps = spec.params[name]
            assert ps.min <= val <= ps.max, f"{key}.{name} default out of range"


# --------------------------------------------------------------------------- #
# Plan structure — every typology
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("typology", list(TYPOLOGIES.keys()))
def test_every_typology_builds_valid_plan(typology):
    engine = ScenarioEngine()
    plans = engine.build_plan(typology, params=default_params(typology), seed=42, count=1)
    assert len(plans) == 1
    plan = plans[0]
    assert isinstance(plan, ScenarioPlan)
    assert plan.typology == typology
    assert plan.target_rules, f"{typology} has no target rules carried into plan"
    assert plan.edges, f"{typology} produced no edges"
    assert plan.members, f"{typology} has no members"
    # Monotonically non-decreasing timestamps
    offsets = [e.timestamp_offset_s for e in plan.edges]
    assert offsets == sorted(offsets), f"{typology} edges not in chronological order"
    # All amounts positive
    assert all(e.amount > 0 for e in plan.edges)
    # Members are referenced by edges (sanity)
    referenced = {e.sender for e in plan.edges} | {e.receiver for e in plan.edges}
    in_members = set(plan.members)
    # External seed accounts (EXT-/STR-S/etc.) are allowed not to appear in members
    overlap = referenced & in_members
    assert overlap, f"{typology} edges reference NONE of its declared members"


@pytest.mark.parametrize("typology", list(TYPOLOGIES.keys()))
def test_events_from_plans_produces_transaction_events(typology):
    engine = ScenarioEngine()
    plans = engine.build_plan(typology, default_params(typology), seed=7, count=2)
    base = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
    events = engine.events_from_plans(plans, base_time=base)
    assert len(events) == sum(len(p.edges) for p in plans)
    # Timestamps strictly sorted globally
    ts = [e.timestamp for e in events]
    assert ts == sorted(ts)
    # Sample event has all the required Pydantic fields
    sample = events[0]
    assert sample.amount > 0
    assert sample.sender_account
    assert sample.receiver_account
    assert sample.amount_leading_digit >= 1


# --------------------------------------------------------------------------- #
# R14 / R15 honored at the PLAN level
# --------------------------------------------------------------------------- #


def test_fan_in_plan_meets_r14_window_and_cv():
    """R14 fires when receiver_in_degree_unique_24h >= 5 AND CV < 0.35.

    The plan should produce ≥5 distinct senders all hitting the same
    aggregator within a 24-hour span, AND the amounts should be uniform
    enough that the coefficient of variation is < 0.35.
    """
    engine = ScenarioEngine()
    params = default_params("fan_in")
    params["sources"] = 8
    params["cv"] = 0.10
    plans = engine.build_plan("fan_in", params, seed=11, count=1)
    plan = plans[0]
    aggregator = plan.metadata.get("aggregator")
    assert aggregator is not None
    # All edges land on the aggregator
    incoming = [e for e in plan.edges if e.receiver == aggregator]
    assert len(incoming) == params["sources"]
    distinct_senders = {e.sender for e in incoming}
    assert len(distinct_senders) >= 5

    # 24h window check
    span_s = max(e.timestamp_offset_s for e in incoming) - min(
        e.timestamp_offset_s for e in incoming
    )
    assert span_s <= 24 * 3600.0, "fan-in spread exceeds R14's 24h window"

    # CV check
    amounts = [float(e.amount) for e in incoming]
    mean = sum(amounts) / len(amounts)
    var = sum((a - mean) ** 2 for a in amounts) / len(amounts)
    cv = math.sqrt(var) / mean if mean > 0 else 0.0
    assert cv < 0.35, f"fan-in CV {cv:.3f} doesn't trigger R14"


def test_layering_chain_plan_meets_r15_relay_window():
    """R15 needs each hop to forward 85-100% of inflow within 2 hours.

    Confirm the engine generates edges that satisfy both windows.
    """
    engine = ScenarioEngine()
    params = default_params("layering_chain")
    params["decay"] = 95     # retain 95% per hop → in the 85-100% window
    params["hops"] = 5
    params["gap_h"] = 1      # nominal mean ≤ 1h; engine caps under 2h
    plans = engine.build_plan("layering_chain", params, seed=23, count=1)
    plan = plans[0]
    assert len(plan.edges) == params["hops"]

    # Walk consecutive hops and check both retain ratio + < 2h gap
    for i in range(1, len(plan.edges)):
        prev = plan.edges[i - 1]
        curr = plan.edges[i]
        gap_s = curr.timestamp_offset_s - prev.timestamp_offset_s
        assert gap_s < 7200, f"hop {i} gap {gap_s:.0f}s exceeds R15 2-hour window"
        ratio = curr.amount / prev.amount
        assert 0.85 <= ratio <= 1.0, f"hop {i} retain ratio {ratio:.2f} outside R15 band"


# --------------------------------------------------------------------------- #
# Realism: shared infra pool present where the typology asks for it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("typology", [
    "structuring", "circular_laundering", "layering_chain",
    "fan_in", "fan_out", "velocity_burst",
])
def test_shared_device_appears_in_ring_edges(typology):
    """For non-cross-border, non-fraud-ring typologies, the engine should put
    most edges on the same device (the Patch-4 shared infra pool)."""
    engine = ScenarioEngine()
    plans = engine.build_plan(typology, default_params(typology), seed=5, count=1)
    devices = [e.device_id for e in plans[0].edges if e.device_id]
    assert devices, f"{typology} produced edges with no device_id"
    most_common_count = max(devices.count(d) for d in set(devices))
    # The shared device should be on more than half the edges
    assert most_common_count >= len(devices) / 2


def test_dormant_activation_uses_new_device_for_burst():
    """Dormant-activation's R07 trigger is "previously unseen device" — the
    inbound credit must use a different device id than the disbursement
    burst."""
    engine = ScenarioEngine()
    plans = engine.build_plan("dormant_activation", default_params("dormant_activation"),
                              seed=9, count=1)
    plan = plans[0]
    # First edge is the inbound credit, the rest are the burst
    inbound = plan.edges[0]
    burst = plan.edges[1:]
    assert inbound.device_id == "dev-employer"
    burst_devices = {e.device_id for e in burst}
    assert "dev-employer" not in burst_devices


def test_cross_border_routes_through_high_risk_countries():
    engine = ScenarioEngine()
    plans = engine.build_plan("cross_border_layering",
                              default_params("cross_border_layering"),
                              seed=31, count=1)
    plan = plans[0]
    countries = {e.sender_country for e in plan.edges} | {e.receiver_country for e in plan.edges}
    # At least one high-risk country present (the whole point of the typology)
    assert countries & {"AE", "MU", "CN", "NG", "PK"}
    # And at least one international hop
    intl = [e for e in plan.edges if e.is_international]
    assert intl, "no international edges generated"


def test_round_tripping_returns_to_origin():
    engine = ScenarioEngine()
    plans = engine.build_plan("round_tripping", default_params("round_tripping"),
                              seed=17, count=1)
    plan = plans[0]
    # The first edge's sender should be the LAST edge's receiver
    assert plan.edges[0].sender == plan.edges[-1].receiver


# --------------------------------------------------------------------------- #
# Multi-instance, error paths
# --------------------------------------------------------------------------- #


def test_count_greater_than_one_produces_distinct_plans():
    engine = ScenarioEngine()
    plans = engine.build_plan("structuring", default_params("structuring"), seed=2, count=3)
    assert len(plans) == 3
    ring_ids = {p.ring_id for p in plans}
    assert len(ring_ids) == 3, "ring ids collided across count=3 plans"


def test_unknown_typology_raises():
    engine = ScenarioEngine()
    with pytest.raises(KeyError):
        engine.build_plan("not_a_real_typology", {}, seed=1, count=1)


def test_params_are_clipped_to_spec_bounds():
    """Send wildly out-of-range params; the engine should clamp them in."""
    engine = ScenarioEngine()
    # structuring count max is 15
    plans = engine.build_plan("structuring", {"count": 9999}, seed=1, count=1)
    assert len(plans[0].edges) <= 15
