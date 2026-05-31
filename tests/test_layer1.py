"""Layer 1 rule-engine tests.

Per-rule trigger / non-trigger tests for all 15 rules, plus the six R14/R15
tests called out in docs/rules_r14_r15_addendum.md. The R01-R13 tests build
each rule's specific firing condition and assert it lands in
``triggered_rules`` (and that a "baseline normal" event does not trigger it).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.system2_detection.layer1_rules.rule_definitions import RULES
from src.system2_detection.layer1_rules.rule_engine import RuleEngine
from src.system2_detection.shared.schemas import (
    LiveFeatureVector,
    LiveGraphFeatureVector,
    TransactionEvent,
)


# --------------------------------------------------------------------------- #
# Factories — produce "normal" instances that no rule should fire on
# --------------------------------------------------------------------------- #


def make_event(**overrides) -> TransactionEvent:
    defaults = dict(
        transaction_id="tx-test",
        timestamp=datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
        sender_account="A-normal",
        receiver_account="B-normal",
        sender_bank="HDFC",
        receiver_bank="HDFC",
        sender_country="IN",
        receiver_country="IN",
        amount=12_000.0,
        currency="INR",
        transaction_type="UPI",
        payment_channel="Mobile",
        device_id="dev-normal",
        ip_address="203.0.113.10",
        geo_latitude=19.07,
        geo_longitude=72.87,
        merchant_category="Grocery",
        transaction_status="Success",
        kyc_level=2,
        is_international=False,
        amount_leading_digit=1,
    )
    defaults.update(overrides)
    return TransactionEvent(**defaults)


def make_features(**overrides) -> LiveFeatureVector:
    defaults = dict(
        transaction_id="tx-test",
        sender_account="A-normal",
        # Temporal
        tx_velocity_1h=1.0,
        tx_velocity_6h=2.0,
        tx_velocity_24h=4.0,
        tx_velocity_7d=10.0,
        avg_amount_7d=15_000.0,
        std_amount_7d=4_000.0,
        tx_gap_seconds=3_600.0,
        night_tx_ratio=0.1,
        weekend_tx_ratio=0.2,
        hour_of_day=12,
        day_of_week=2,
        # Behavioral
        beneficiary_count_7d=5,
        beneficiary_count_30d=15,
        receiver_entropy=2.3,
        amount_zscore=0.1,
        avg_daily_volume_30d=20_000.0,
        round_amount_flag=False,
        sub_threshold_flag=False,
        amount_leading_digit=1,
        benford_chi2_score=1.0,
        # Geographic
        geo_distance_km=5.0,
        country_switch_count_7d=0,
        impossible_travel_flag=False,
        high_risk_country_flag=False,
        cross_border_ratio_30d=0.0,
        # Device
        device_change_count_7d=0,
        new_device_flag=False,
        shared_device_count=0,
        ip_change_count_24h=1,
        # Derived
        amount_raw=12_000.0,
        amount_log=9.39,
        hour_sin=0.0,
        hour_cos=-1.0,
        dow_sin=0.97,
        dow_cos=-0.22,
        tx_type_NEFT=False,
        tx_type_RTGS=False,
        tx_type_IMPS=False,
        tx_type_Wire=False,
        tx_type_UPI=True,
        tx_type_Card=False,
        channel_Mobile=True,
        channel_Web=False,
        channel_ATM=False,
        channel_Branch=False,
        scaled_feature_vector=[],
    )
    defaults.update(overrides)
    return LiveFeatureVector(**defaults)


def make_graph_vec(**overrides) -> LiveGraphFeatureVector:
    defaults = dict(
        transaction_id="tx-test",
        sender_account="A-normal",
        receiver_account="B-normal",
        sender_in_degree=3,
        sender_out_degree=4,
        sender_in_degree_unique=3,
        sender_out_degree_unique=4,
        sender_2hop_cycle_count=0,
        sender_3hop_cycle_count=0,
        sender_fan_in_score=0.1,
        sender_fan_out_score=0.1,
        sender_pagerank=0.001,
        sender_betweenness=0.001,
        sender_community_id=0,
        sender_community_size=20,
        sender_community_density=0.1,
        sender_community_risk_score=0.05,
        sender_benford_chi2_community=1.0,
        receiver_in_degree=2,
        receiver_out_degree=2,
        receiver_community_id=0,
        receiver_community_risk_score=0.05,
        edge_creates_cycle=False,
        cycle_length=0,
        shared_community=True,
        two_hop_neighborhood=[],
        two_hop_edge_list=[],
        # R14
        receiver_in_degree_unique_24h=1,
        receiver_inflow_amount_cv=1.5,
        # R15
        sender_is_relay_node=False,
        sender_last_inflow_amount=0.0,
        sender_last_inflow_gap_seconds=float("inf"),
    )
    defaults.update(overrides)
    return LiveGraphFeatureVector(**defaults)


@pytest.fixture
def engine() -> RuleEngine:
    return RuleEngine()


# --------------------------------------------------------------------------- #
# Sanity / structural
# --------------------------------------------------------------------------- #


def test_engine_has_15_rules(engine):
    assert len(engine.rules) == 15
    assert engine.rule_ids() == [f"R{i:02d}" for i in range(1, 16)]


def test_max_possible_raw_is_computed():
    """Guard against hardcoded denominators.

    The architecture's L1 rule weights as listed (R01..R13) sum to 10.5 and
    with R14 (0.8) + R15 (0.9) the 15-rule sum is 12.2. The addendum's prose
    saying 9.5 / 11.2 is a math error; the RULES table is the source of truth
    and the engine MUST compute the denominator from it, not hardcode.
    """
    expected = sum(r.weight for r in RULES)
    engine = RuleEngine()
    assert len(RULES) == 15
    assert abs(engine.max_possible_raw - expected) < 1e-9
    assert abs(expected - 12.2) < 1e-6


def test_baseline_normal_triggers_nothing(engine):
    out = engine.evaluate(make_features(), make_graph_vec(), make_event())
    assert out.triggered_rules == []
    assert out.rule_score == 0.0
    assert out.rule_count == 0


def test_score_is_normalised_in_range(engine):
    """Even if every rule fires, rule_score stays ≤ 100."""
    out = engine.evaluate(
        make_features(
            sub_threshold_flag=True,
            tx_velocity_1h=20,
            tx_velocity_24h=80,
            tx_velocity_7d=200,
            tx_gap_seconds=20_000_000,
            impossible_travel_flag=True,
            high_risk_country_flag=True,
            new_device_flag=True,
            shared_device_count=10,
            beneficiary_count_7d=50,
            round_amount_flag=True,
            benford_chi2_score=10.0,
            geo_distance_km=10_000,
        ),
        make_graph_vec(
            edge_creates_cycle=True,
            cycle_length=3,
            receiver_in_degree_unique_24h=6,
            receiver_inflow_amount_cv=0.2,
            sender_is_relay_node=True,
            sender_last_inflow_amount=1_000_000,
            sender_last_inflow_gap_seconds=1800,
        ),
        make_event(
            amount=950_000,
            is_international=True,
            sender_country="IN",
            receiver_country="AE",
            kyc_level=0,
        ),
    )
    assert 0.0 <= out.rule_score <= 100.0


# --------------------------------------------------------------------------- #
# Per-rule trigger tests (R01–R13)
# --------------------------------------------------------------------------- #


def test_r01_fires_above_threshold(engine):
    out = engine.evaluate(make_features(), make_graph_vec(), make_event(amount=2_000_000))
    assert "R01" in out.triggered_rules


def test_r01_does_not_fire_below(engine):
    out = engine.evaluate(make_features(), make_graph_vec(), make_event(amount=900_000))
    assert "R01" not in out.triggered_rules


def test_r02_fires_with_sub_threshold_and_burst(engine):
    out = engine.evaluate(
        make_features(sub_threshold_flag=True, tx_velocity_1h=4),
        make_graph_vec(),
        make_event(amount=900_000),
    )
    assert "R02" in out.triggered_rules


def test_r02_does_not_fire_without_velocity(engine):
    out = engine.evaluate(
        make_features(sub_threshold_flag=True, tx_velocity_1h=2),
        make_graph_vec(),
        make_event(),
    )
    assert "R02" not in out.triggered_rules


def test_r03_fires_on_hourly_burst(engine):
    out = engine.evaluate(make_features(tx_velocity_1h=15), make_graph_vec(), make_event())
    assert "R03" in out.triggered_rules


def test_r03_fires_on_daily_burst(engine):
    out = engine.evaluate(make_features(tx_velocity_24h=60), make_graph_vec(), make_event())
    assert "R03" in out.triggered_rules


def test_r04_fires_after_dormancy(engine):
    out = engine.evaluate(
        make_features(tx_gap_seconds=20_000_000),  # ~231 days
        make_graph_vec(),
        make_event(amount=200_000),
    )
    assert "R04" in out.triggered_rules


def test_r04_does_not_fire_low_amount(engine):
    out = engine.evaluate(
        make_features(tx_gap_seconds=20_000_000),
        make_graph_vec(),
        make_event(amount=50_000),
    )
    assert "R04" not in out.triggered_rules


def test_r05_fires_on_impossible_travel(engine):
    out = engine.evaluate(
        make_features(impossible_travel_flag=True), make_graph_vec(), make_event()
    )
    assert "R05" in out.triggered_rules


def test_r06_fires_on_high_risk_international(engine):
    out = engine.evaluate(
        make_features(high_risk_country_flag=True),
        make_graph_vec(),
        make_event(is_international=True, sender_country="IN", receiver_country="AE"),
    )
    assert "R06" in out.triggered_rules


def test_r06_does_not_fire_domestic(engine):
    out = engine.evaluate(
        make_features(high_risk_country_flag=True),
        make_graph_vec(),
        make_event(is_international=False),
    )
    assert "R06" not in out.triggered_rules


def test_r07_fires_on_new_device_highval(engine):
    out = engine.evaluate(
        make_features(new_device_flag=True),
        make_graph_vec(),
        make_event(amount=300_000),
    )
    assert "R07" in out.triggered_rules


def test_r07_does_not_fire_small_amount(engine):
    out = engine.evaluate(
        make_features(new_device_flag=True),
        make_graph_vec(),
        make_event(amount=50_000),
    )
    assert "R07" not in out.triggered_rules


def test_r08_fires_on_device_sharing(engine):
    out = engine.evaluate(
        make_features(shared_device_count=8), make_graph_vec(), make_event()
    )
    assert "R08" in out.triggered_rules


def test_r09_fires_on_excess_beneficiaries(engine):
    out = engine.evaluate(
        make_features(beneficiary_count_7d=30), make_graph_vec(), make_event()
    )
    assert "R09" in out.triggered_rules


def test_r10_fires_on_cycle_closure(engine):
    out = engine.evaluate(
        make_features(),
        make_graph_vec(edge_creates_cycle=True, cycle_length=3),
        make_event(),
    )
    assert "R10" in out.triggered_rules


def test_r10_does_not_fire_without_cycle(engine):
    out = engine.evaluate(make_features(), make_graph_vec(), make_event())
    assert "R10" not in out.triggered_rules


def test_r11_fires_on_round_amount_burst(engine):
    out = engine.evaluate(
        make_features(round_amount_flag=True, tx_velocity_24h=10),
        make_graph_vec(),
        make_event(amount=500_000),
    )
    assert "R11" in out.triggered_rules


def test_r12_fires_on_kyc0_highval(engine):
    out = engine.evaluate(
        make_features(),
        make_graph_vec(),
        make_event(kyc_level=0, amount=600_000),
    )
    assert "R12" in out.triggered_rules


def test_r12_does_not_fire_for_valid_kyc(engine):
    out = engine.evaluate(
        make_features(),
        make_graph_vec(),
        make_event(kyc_level=2, amount=600_000),
    )
    assert "R12" not in out.triggered_rules


def test_r13_fires_on_benford_anomaly(engine):
    out = engine.evaluate(
        make_features(benford_chi2_score=5.0), make_graph_vec(), make_event()
    )
    assert "R13" in out.triggered_rules


def test_r13_does_not_fire_near_benford(engine):
    out = engine.evaluate(
        make_features(benford_chi2_score=2.0), make_graph_vec(), make_event()
    )
    assert "R13" not in out.triggered_rules


# --------------------------------------------------------------------------- #
# R14 / R15 — verbatim from the addendum
# --------------------------------------------------------------------------- #


def test_r14_fan_in_fires(engine):
    graph = make_graph_vec(
        receiver_in_degree_unique_24h=6, receiver_inflow_amount_cv=0.20
    )
    out = engine.evaluate(make_features(), graph, make_event())
    assert "R14" in out.triggered_rules


def test_r14_ignores_busy_merchant(engine):
    graph = make_graph_vec(
        receiver_in_degree_unique_24h=12, receiver_inflow_amount_cv=1.4
    )
    out = engine.evaluate(make_features(), graph, make_event())
    assert "R14" not in out.triggered_rules


def test_r15_layering_relay_fires(engine):
    graph = make_graph_vec(
        sender_is_relay_node=True,
        sender_last_inflow_amount=1_000_000,
        sender_last_inflow_gap_seconds=1800,
    )
    out = engine.evaluate(make_features(), graph, make_event(amount=950_000))
    assert "R15" in out.triggered_rules


def test_r15_ignores_partial_spend(engine):
    graph = make_graph_vec(
        sender_is_relay_node=True,
        sender_last_inflow_amount=1_000_000,
        sender_last_inflow_gap_seconds=1800,
    )
    out = engine.evaluate(make_features(), graph, make_event(amount=100_000))
    assert "R15" not in out.triggered_rules


def test_r15_ignores_slow_forward(engine):
    graph = make_graph_vec(
        sender_is_relay_node=True,
        sender_last_inflow_amount=1_000_000,
        sender_last_inflow_gap_seconds=18_000,
    )
    out = engine.evaluate(make_features(), graph, make_event(amount=950_000))
    assert "R15" not in out.triggered_rules


# --------------------------------------------------------------------------- #
# Output shape
# --------------------------------------------------------------------------- #


def test_explanations_present_for_each_triggered_rule(engine):
    out = engine.evaluate(
        make_features(),
        make_graph_vec(edge_creates_cycle=True, cycle_length=2),
        make_event(amount=2_000_000),
    )
    assert out.rule_count == len(out.triggered_rules)
    assert len(out.rule_explanations) == len(out.triggered_rules)
    for ex in out.rule_explanations:
        assert isinstance(ex, str) and len(ex) > 0


def test_rule_score_v2_per_fire_severity(engine):
    """v2 aggregation: one weight-1.0 rule fires -> per_fire_score(1.0) = 70.

    The v1 formula gave ~8 for this case which is invisible to fusion. v2's
    per-fire base+slope (50 + 40 * (w - 0.5)) puts a single Critical-tier
    rule fire at 70 (solidly High) so Layer 1 can actually lead the fused
    decision instead of just decorating it.
    """
    out = engine.evaluate(
        make_features(),
        make_graph_vec(),
        make_event(amount=2_000_000),  # only R01 (weight 1.0) fires
    )
    assert out.triggered_rules == ["R01"]
    # _per_fire_score(1.0) = 50 + 40 * 0.5 = 70
    assert out.rule_score == pytest.approx(70.0, abs=1e-6)


def test_rule_score_single_low_weight_fire_lands_in_medium(engine):
    """A single weight-0.7 rule fire should land mid-Medium (~58), not
    invisible (the v1 8.2 territory).
    """
    out = engine.evaluate(
        make_features(high_risk_country_flag=True),
        make_graph_vec(),
        make_event(amount=300_000, sender_country="IN", receiver_country="AE",
                   is_international=True),  # only R06 (weight 0.7) fires
    )
    assert out.triggered_rules == ["R06"]
    # _per_fire_score(0.7) = 50 + 40 * 0.2 = 58
    assert out.rule_score == pytest.approx(58.0, abs=1e-6)


def test_rule_score_co_firing_bonus_pushes_toward_critical(engine):
    """Co-firing bonus: top per-fire score + 10 per additional rule, capped at +30.

    R01 (1.0) + R10 (1.0) together: top=70, +10 for one extra fire -> 80.
    """
    out = engine.evaluate(
        make_features(),
        make_graph_vec(edge_creates_cycle=True, cycle_length=2),
        make_event(amount=2_000_000),   # R01 + R10 both fire
    )
    assert set(out.triggered_rules) >= {"R01", "R10"}
    # top = 70, co_bonus = +10 per additional rule, capped at 30.
    n_fired = len(out.triggered_rules)
    expected = min(100.0, 70.0 + min(30.0, 10.0 * (n_fired - 1)))
    assert out.rule_score == pytest.approx(expected, abs=1e-6)
    # 2-rule co-fire MUST clear the High threshold (>60)
    assert out.rule_score >= 60.0


def test_rule_score_zero_when_nothing_fires(engine):
    out = engine.evaluate(make_features(), make_graph_vec(), make_event())
    assert out.triggered_rules == []
    assert out.rule_score == 0.0
