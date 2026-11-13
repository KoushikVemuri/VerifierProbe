"""Settings for the Attacker: defaults, loading, and validation.

Unknown keys raise ValueError on purpose — a typo like `verifer_calls`
should fail loudly instead of silently running with the default budget.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

# All keys the Attacker understands, with sensible defaults.
# Anything not in this dict is rejected by load_config().
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 0,                 # makes PGD + bisection reproducible
    "label": None,             # None = use the model's own prediction
    "tolerance": 1e-6,         # assumed float32 noise when judging a margin
    "pgd_steps": 100,          # gradient steps per restart
    "pgd_restarts": 20,        # random starting points for PGD
    "grow_factors": [1.1, 1.25, 1.5, 2.0, 3.0],  # how far to grow the radius
    "knife_margin": 1e-3,      # target margin at the probe point c
    "bisect_steps": 60,        # precision of the model-side bisections
    "verifier_calls": 25,      # hard cap on verify() calls per run
}


def load_config(config: Optional[Union[dict, str, Path]] = None) -> Dict[str, Any]:
    """Return the full settings dict.

    `config` may be None, a dict, or the path to a JSON file. Defaults
    are overlaid with whatever the caller passes in. Unknown keys raise
    ValueError so a typo can't silently be ignored.
    """
    if config is None:
        user: Dict[str, Any] = {}
    elif isinstance(config, (str, Path)):
        user = json.loads(Path(config).read_text())
    elif isinstance(config, dict):
        user = dict(config)
    else:
        raise TypeError("config must be None, a dict, or a path to a JSON file")

    # Catch typos early. This has saved us from a few silent-budget bugs.
    unknown = set(user) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    merged = {**DEFAULT_CONFIG, **user}
    _check(merged)
    return merged


def _check(cfg: Dict[str, Any]) -> None:
    """Raise ValueError if any setting is out of range."""
    for key in ("seed", "pgd_steps", "pgd_restarts", "bisect_steps", "verifier_calls"):
        # bool is a subclass of int, so reject it explicitly.
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be an integer")
    for key in ("pgd_steps", "pgd_restarts", "bisect_steps", "verifier_calls"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be at least 1")
    for key in ("tolerance", "knife_margin"):
        if not cfg[key] > 0:
            raise ValueError(f"{key} must be greater than 0")
    if cfg["label"] is not None and not isinstance(cfg["label"], int):
        raise ValueError("label must be an integer or null")
    factors = cfg["grow_factors"]
    if not factors or any(f <= 1 for f in factors):
        raise ValueError("grow_factors must be a non-empty list of numbers > 1")