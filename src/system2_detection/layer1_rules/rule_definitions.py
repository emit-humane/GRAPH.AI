"""15 AML rule definitions (R01-R15).

R01-R13 transcribe the architecture's L1 rule table verbatim.
R14 (fan-in aggregation) and R15 (layering chain relay) implement the
``docs/rules_r14_r15_addendum.md`` spec exactly, reading the five new
LiveGraphFeatureVector fields D2 produces.

Note on the denominator: ``max_possible_raw = sum(r.weight for r in RULES)``
is COMPUTED in RuleEngine — never hardcoded. With the rule weights as listed
in the architecture, that sum is 12.2 (not 11.2 as the addendum's prose
incorrectly claims; the addendum lists the same 13 weights that we use and
they themselves sum to 10.5, not 9.5 — pure math error in the addendum text).
The RULES list is the source of truth; if either of those documents updates
a weight, change it here and the denominator updates automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Type aliases for the condition signature
ConditionFn = Callable[[Any, Any, Any], bool]
ExplanationFn = Callable[[Any, Any, Any], str]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    weight: float
    condition: ConditionFn
    explain: ExplanationFn

    def fires(self, features, graph, event) -> bool:
        return bool(self.condition(features, graph, event))

    def explanation(self, features, graph, event) -> str:
        return self.explain(features, graph, event)


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #


def _amount(event) -> float:
    return float(event.amount)


def _kyc(event) -> int | None:
    return getattr(event, "kyc_level", None)


# --- R01 ---
def _r01_cond(f, g, e):
    return _amount(e) > 1_000_000


def _r01_exp(f, g, e):
    return f"High-value transfer of {_amount(e):,.0f} INR exceeds the 1,000,000 threshold."


# --- R02 ---
def _r02_cond(f, g, e):
    return bool(f.sub_threshold_flag) and f.tx_velocity_1h >= 3


def _r02_exp(f, g, e):
    return (
        f"Structuring: sub-threshold amount of {_amount(e):,.0f} INR with "
        f"{int(f.tx_velocity_1h)} txns in the last 1h."
    )


# --- R03 ---
def _r03_cond(f, g, e):
    return f.tx_velocity_1h > 10 or f.tx_velocity_24h > 50


def _r03_exp(f, g, e):
    return (
        f"Velocity spike: {int(f.tx_velocity_1h)} txns/1h, "
        f"{int(f.tx_velocity_24h)} txns/24h exceeds (10 / 50)."
    )


# --- R04 ---
def _r04_cond(f, g, e):
    return f.tx_gap_seconds > 15_552_000 and _amount(e) > 100_000  # 180 days


def _r04_exp(f, g, e):
    days = f.tx_gap_seconds / 86_400.0
    return (
        f"Dormant activation: {days:.0f}-day gap since prior txn, "
        f"now moving {_amount(e):,.0f} INR (>100K threshold)."
    )


# --- R05 ---
def _r05_cond(f, g, e):
    return bool(f.impossible_travel_flag)


def _r05_exp(f, g, e):
    speed = (
        f.geo_distance_km / (f.tx_gap_seconds / 3600.0)
        if f.tx_gap_seconds > 0
        else float("inf")
    )
    return (
        f"Impossible travel: {f.geo_distance_km:.0f} km in "
        f"{f.tx_gap_seconds / 3600.0:.1f} h (~{speed:.0f} km/h > 900 km/h)."
    )


# --- R06 ---
def _r06_cond(f, g, e):
    return bool(f.high_risk_country_flag) and bool(e.is_international)


def _r06_exp(f, g, e):
    return (
        f"High-risk jurisdiction: international transfer "
        f"{e.sender_country} -> {e.receiver_country}."
    )


# --- R07 ---
def _r07_cond(f, g, e):
    return bool(f.new_device_flag) and _amount(e) > 200_000


def _r07_exp(f, g, e):
    return (
        f"Device anomaly: previously unseen device used for "
        f"{_amount(e):,.0f} INR (>200K)."
    )


# --- R08 ---
def _r08_cond(f, g, e):
    return f.shared_device_count > 5


def _r08_exp(f, g, e):
    return (
        f"Shared device: {int(f.shared_device_count)} other accounts have used "
        f"the same device id."
    )


# --- R09 ---
def _r09_cond(f, g, e):
    return f.beneficiary_count_7d > 20


def _r09_exp(f, g, e):
    return (
        f"Excessive beneficiaries: {int(f.beneficiary_count_7d)} distinct "
        f"receivers in 7d (> 20)."
    )


# --- R10 ---
def _r10_cond(f, g, e):
    return bool(g.edge_creates_cycle)

def _r10_exp(f, g, e):
    return (
        f"Cycle closure: this transaction closes a {int(g.cycle_length)}-hop "
        f"directed cycle through the multigraph."
    )


# --- R11 ---
def _r11_cond(f, g, e):
    return bool(f.round_amount_flag) and f.tx_velocity_24h > 5


def _r11_exp(f, g, e):
    return (
        f"Round-amount structuring: round amount of {_amount(e):,.0f} INR with "
        f"{int(f.tx_velocity_24h)} txns/24h."
    )


# --- R12 ---
def _r12_cond(f, g, e):
    kyc = _kyc(e)
    return kyc is not None and kyc == 0 and _amount(e) > 500_000


def _r12_exp(f, g, e):
    return (
        f"KYC mismatch: KYC-0 (none) account moving {_amount(e):,.0f} INR "
        f"(>500K)."
    )


# --- R13 ---
def _r13_cond(f, g, e):
    return f.benford_chi2_score > 3.84  # chi-sq critical at alpha=0.05, df=8 - but spec uses 3.84


def _r13_exp(f, g, e):
    return (
        f"Benford anomaly: chi-square deviation {f.benford_chi2_score:.2f} "
        f"exceeds 3.84 (alpha=0.05)."
    )


# --- R14 (addendum) ---
def _r14_cond(f, g, e):
    return g.receiver_in_degree_unique_24h >= 5 and g.receiver_inflow_amount_cv < 0.35


def _r14_exp(f, g, e):
    return (
        f"Fan-in aggregation: receiver collected from "
        f"{int(g.receiver_in_degree_unique_24h)} distinct sources in 24h with "
        f"near-uniform amounts (CV={g.receiver_inflow_amount_cv:.2f}), "
        f"consistent with a collection / mule account."
    )


# --- R15 (addendum) ---
def _r15_cond(f, g, e):
    if g.sender_last_inflow_amount <= 0:
        return False
    retain_ratio = _amount(e) / g.sender_last_inflow_amount
    return (
        bool(g.sender_is_relay_node)
        and 0.85 <= retain_ratio <= 1.0
        and g.sender_last_inflow_gap_seconds < 7200
    )


def _r15_exp(f, g, e):
    retain_ratio = _amount(e) / max(g.sender_last_inflow_amount, 1e-9)
    gap_minutes = g.sender_last_inflow_gap_seconds / 60.0
    return (
        f"Layering chain relay: account forwarded {retain_ratio:.0%} of a "
        f"recent inflow ({g.sender_last_inflow_amount:,.0f} INR) within "
        f"{gap_minutes:.0f} minutes as a single-in / single-out relay node "
        f"— a classic mid-chain layering hop."
    )


# --------------------------------------------------------------------------- #
# Public registry
# --------------------------------------------------------------------------- #

RULES: tuple[Rule, ...] = (
    Rule("R01", "High-value transfer", 1.0, _r01_cond, _r01_exp),
    Rule("R02", "Structuring", 1.0, _r02_cond, _r02_exp),
    Rule("R03", "Velocity spike", 0.8, _r03_cond, _r03_exp),
    Rule("R04", "Dormant activation", 0.9, _r04_cond, _r04_exp),
    Rule("R05", "Impossible travel", 0.9, _r05_cond, _r05_exp),
    Rule("R06", "High-risk jurisdiction", 0.7, _r06_cond, _r06_exp),
    Rule("R07", "Device anomaly", 0.6, _r07_cond, _r07_exp),
    Rule("R08", "Shared device", 0.7, _r08_cond, _r08_exp),
    Rule("R09", "Excessive beneficiaries", 0.8, _r09_cond, _r09_exp),
    Rule("R10", "Cycle closure", 1.0, _r10_cond, _r10_exp),
    Rule("R11", "Round-amount structuring", 0.6, _r11_cond, _r11_exp),
    Rule("R12", "KYC mismatch", 0.8, _r12_cond, _r12_exp),
    Rule("R13", "Benford anomaly", 0.7, _r13_cond, _r13_exp),
    Rule("R14", "Fan-in aggregation", 0.8, _r14_cond, _r14_exp),
    Rule("R15", "Layering chain relay", 0.9, _r15_cond, _r15_exp),
)


def rule_by_id(rule_id: str) -> Rule:
    for r in RULES:
        if r.rule_id == rule_id:
            return r
    raise KeyError(f"Unknown rule id: {rule_id}")
