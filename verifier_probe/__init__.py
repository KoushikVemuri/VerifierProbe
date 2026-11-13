"""Public API for verifier_probe.

Everything a user needs to run an attack or dump a trace is exported here.
"""
from .adapter import AlphaBetaCrownVerifier, Certificate, Region
from .attacker import Attacker
from .config import DEFAULT_CONFIG, load_config
from .prober import Prober
from .trace_schema import TRACE_FIELDS, validate_trace

__all__ = [
    "AlphaBetaCrownVerifier",  # talks to the verifier
    "Certificate",             # what the verifier returns
    "Region",                  # the L-infinity ball we ask about
    "Attacker",                # finds certified-but-wrong inputs
    "Prober",                  # dumps intermediate bounds
    "DEFAULT_CONFIG",
    "load_config",
    "TRACE_FIELDS",
    "validate_trace",
]