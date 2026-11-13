"""Dump the verifier's intermediate bounds as a JSON-serializable trace.

We only READ numbers. The verifier's source is never touched, so its
decision cannot change because of us. Two sources of bounds:

  1. cert.reference — the verifier's own state after branch-and-bound.
     Present only if BaB actually ran.
  2. auto_LiRPA — if the verifier proved the property at its first
     bound it kept no state, so we recompute per-layer bounds with
     auto_LiRPA (the bound engine inside alpha-beta-CROWN) in a fresh
     BoundedModule. Fresh means it doesn't touch the verifier.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .adapter import Region
from .trace_schema import validate_trace

logger = logging.getLogger(__name__)

# (node name, lower bound, upper bound) for one layer
Layer = Tuple[str, torch.Tensor, torch.Tensor]


def _to_list(tensor: Optional[torch.Tensor]) -> Optional[list]:
    return None if tensor is None else tensor.detach().cpu().tolist()


class Prober:
    """Dump the verifier's per-layer bounds for one input."""

    def __init__(self, verifier: Any, eps: float = 0.03,
                 clip: Optional[Tuple[float, float]] = (0.0, 1.0)) -> None:
        self.verifier = verifier
        self.eps = eps      # radius we ask the verifier about
        self.clip = clip    # valid pixel range (None = don't clamp)

    def dump(self, model: nn.Module, x: torch.Tensor,
             label: Optional[int] = None) -> Dict[str, Any]:
        """Return the trace for input `x` (format documented in trace_schema.py)."""
        region = Region(x, self.eps, self.clip)
        cert = self.verifier.certify(model, region, label)
        data = self.verifier.internal_state(cert)
        lowers = data.get("lower_bounds", {})
        uppers = data.get("upper_bounds", {})

        if lowers:
            # Verifier ran BaB and gave us its own per-layer data.
            source = "verifier_reference"
            layers: List[Layer] = [(name, lowers[name], uppers[name])
                                   for name in lowers]
        else:
            # Verifier proved at the first bound and kept nothing. Recompute.
            source = "auto_lirpa_crown"
            layers = self._layers_from_auto_lirpa(model, region)
        logger.info("Prober: radius %.4g, verdict %s, %d layers from %s",
                    self.eps, cert.status, len(layers), source)

        trace = {
            "input": {"shape": list(x.shape), "eps": float(self.eps),
                      "label": cert.label},
            "source": source,
            "verdict": {"status": cert.status,
                        "bab_entries": cert.bab_entries,
                        "elapsed_sec": cert.elapsed},
            "global_lower_bound": _to_list(data.get("global_lb")),
            "states": self._make_states(layers),
        }
        validate_trace(trace)   # schema check
        json.dumps(trace)       # must be serializable
        return trace

    @staticmethod
    def _layers_from_auto_lirpa(model: nn.Module, region: Region) -> List[Layer]:
        """Per-layer CROWN bounds via auto_LiRPA."""
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        low, high = region.bounds()
        x_batched = region.center.unsqueeze(0)
        ptb = PerturbationLpNorm(norm=float("inf"),
                                 x_L=low.unsqueeze(0), x_U=high.unsqueeze(0))
        bounded = BoundedModule(model, x_batched, verbose=False)
        bounded.compute_bounds(x=(BoundedTensor(x_batched, ptb),), method="CROWN")

        # Only keep layers that actually hold bounds (linear / conv).
        layers: List[Layer] = []
        for name, node in bounded._modules.items():
            is_layer = type(node).__name__ in ("BoundLinear", "BoundConv")
            if is_layer and getattr(node, "lower", None) is not None:
                layers.append((name, node.lower, node.upper))
        return layers

    @staticmethod
    def _make_states(layers: List[Layer]) -> List[Dict[str, Any]]:
        """Turn (name, lower, upper) tuples into trace entries."""
        states = []
        for i, (name, low, up) in enumerate(layers):
            is_last = (i == len(layers) - 1)
            # Unstable = the ReLU could go either way inside the box.
            num_unstable = None
            if not is_last:
                num_unstable = int(((low < 0) & (up > 0)).sum().item())
            states.append({
                "node": name,
                "kind": "output" if is_last else "hidden_layer",
                "shape": list(low.shape),
                "lower": _to_list(low),
                "upper": _to_list(up),
                "num_unstable": num_unstable,
            })
        return states