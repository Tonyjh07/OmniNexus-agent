"""Model registry and routing (v0.2).

Per-provider adapters were removed: model execution now lives in
:mod:`omninexus.agent.engine` (Pydantic AI). This package keeps the pure
config-level routing used by the orchestrator and the HTTP API.
"""

from omninexus.gateway.router import Gateway, RoutePlan

__all__ = ["Gateway", "RoutePlan"]
