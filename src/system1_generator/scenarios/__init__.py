"""Scenario typologies for G3 (one module per typology).

Each module exposes:
    PATTERN: str
    plan_instance(rng, config, accounts, candidate_ids, already_used) -> RingPlan | None

The G3 pipeline iterates over instances per typology until the per-typology
transaction quota implied by config.scenario_probabilities is met.
"""

from .base import RingPlan, ScenarioInfraPool, build_infra_pool, settlement_delay  # noqa: F401

from . import (
    circular_laundering,  # noqa: F401
    cross_border_layering,  # noqa: F401
    dormant_activation,  # noqa: F401
    fan_in,  # noqa: F401
    fan_out,  # noqa: F401
    fraud_ring,  # noqa: F401
    layering_chain,  # noqa: F401
    round_tripping,  # noqa: F401
    structuring,  # noqa: F401
    velocity_burst,  # noqa: F401
)

PATTERN_MODULES = {
    "structuring": structuring,
    "circular_laundering": circular_laundering,
    "layering_chain": layering_chain,
    "fan_in": fan_in,
    "fan_out": fan_out,
    "fraud_ring": fraud_ring,
    "dormant_activation": dormant_activation,
    "velocity_burst": velocity_burst,
    "cross_border_layering": cross_border_layering,
    "round_tripping": round_tripping,
}
