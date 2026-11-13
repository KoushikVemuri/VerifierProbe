"""Search for a region the verifier certifies but the model actually breaks.

Idea (Jia & Rinard): float rounding error only matters where the true
margin is almost zero. So instead of sampling randomly, we build regions
that sit on the knife edge:

  1. find an adversarial point beyond X with PGD in float64
  2. walk from the center toward it to find a point c whose margin is
     just above `knife_margin`
  3. at c, move every pixel in the direction that shrinks the margin
     fastest — that corner of c +- r is where the model is most likely
     to flip
  4. bisect for the largest radius the verifier still certifies
  5. if the verifier certifies a radius whose corner has a non-positive
     real margin, we've got a counterexample inside a "verified" region
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .adapter import Region
from .config import load_config

logger = logging.getLogger(__name__)


def margin(logits: torch.Tensor, label: int) -> torch.Tensor:
    """logit[label] - max(other logits). Positive = correct prediction."""
    others = torch.cat([logits[:label], logits[label + 1:]])
    return logits[label] - others.max()


def pgd_minimize(model64: nn.Module, low: torch.Tensor, high: torch.Tensor,
                 label: int, steps: int, restarts: int,
                 gen: torch.Generator) -> Tuple[torch.Tensor, float]:
    """Gradient-search for the point in [low, high] with the smallest margin.

    Runs in float64 so the reference margin doesn't depend on float32
    execution. Restart 0 starts at the box center; the rest start at
    random corners, since ReLU networks have flat regions where the
    gradient is exactly zero.
    """
    width = high - low
    best_x, best_m = None, float("inf")
    for r in range(restarts):
        if r == 0:
            x = (low + high) / 2  # center
        else:
            coin = torch.rand(low.shape, generator=gen, dtype=torch.float64)
            x = torch.where(coin < 0.5, low, high)  # random corner
        for t in range(steps):
            x = x.clone().requires_grad_(True)
            grad, = torch.autograd.grad(
                margin(model64(x.unsqueeze(0))[0], label), x)
            step = 0.2 * width * (1 - t / steps)   # step size decays linearly
            x = torch.min(torch.max(x.detach() - step * grad.sign(), low), high)
        with torch.no_grad():
            m = margin(model64(x.unsqueeze(0))[0], label).item()
        if m < best_m:
            best_x, best_m = x, m
    return best_x, best_m


class Attacker:
    """Search for a region that the verifier certifies but the model violates."""

    def __init__(self, model: nn.Module, verifier: Any,
                 config: Optional[Union[dict, str, Path]] = None) -> None:
        self.model = model.eval()                             # float32: how it really runs
        self.model64 = copy.deepcopy(model).double().eval()   # float64: reference
        self.verifier = verifier
        self.config = load_config(config)

    # ---- helpers ----------------------------------------------------------
    def _m64(self, x: torch.Tensor, label: int) -> float:
        """Margin computed in float64. This is our ground truth."""
        with torch.no_grad():
            return margin(self.model64(x.double().unsqueeze(0))[0], label).item()

    def _real_margins(self, x: torch.Tensor, label: int) -> Dict[str, float]:
        """Return the margin at x, computed both ways."""
        with torch.no_grad():
            m32 = margin(self.model(x.unsqueeze(0))[0], label).item()
        return {"fp32": m32, "fp64": self._m64(x, label)}

    # ---- main entry point -------------------------------------------------
    def run(self, X: Region) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the attack on region X. Returns (adv_x, atk_logs)."""
        cfg = self.config
        gen = torch.Generator().manual_seed(cfg["seed"])
        center = X.center.detach().float()
        X = Region(center, X.eps, X.clip)
        logs: Dict[str, Any] = {"config": dict(cfg), "trajectory": []}
        logger.info("Attack: region X = point +- %.4g, seed %d", X.eps, cfg["seed"])

        # 1. Does the verifier certify X? If not, there's no claim to attack.
        cert = self.verifier.certify(self.model, X, cfg["label"])
        label = cert.label
        logs["verifier_claim"] = {
            "status": cert.status, "eps": float(X.eps), "seconds": cert.elapsed,
        }
        logger.info("Verifier's claim about X: %s (%.1f s)", cert.status, cert.elapsed)
        if not cert.verified:
            return self._finish(
                logs, center, label, False,
                f"The verifier did not certify X (status: {cert.status}), "
                f"so there is no claim to attack.")

        # 2. Find a real adversarial point beyond X by growing the radius.
        adv = None
        for factor in cfg["grow_factors"]:
            low, high = Region(center, X.eps * factor, X.clip).bounds()
            point, m = pgd_minimize(
                self.model64, low.double(), high.double(), label,
                cfg["pgd_steps"], cfg["pgd_restarts"], gen)
            if m < 0:
                adv = point
                logs["adversarial_point_found_at_eps"] = float(X.eps * factor)
                logger.info("Found a real adversarial point at radius %.4g",
                            X.eps * factor)
                break
        if adv is None:
            return self._finish(
                logs, center, label, False,
                "No real adversarial point was found, so there is no "
                "decision boundary nearby to probe.")

        # 3. Walk from the center toward `adv` and stop just above knife_margin.
        c = self._point_with_margin(center, adv, label, cfg["knife_margin"])
        direction = self._worst_direction(c, adv, label)

        def corner(r: float) -> torch.Tensor:
            """The lowest-margin corner of the box c +- r."""
            low, high = Region(c, r, X.clip).bounds()
            return torch.where(direction > 0, high,
                               torch.where(direction < 0, low, c))

        # 4. Find r0: the radius where the real margin at that corner hits 0.
        r_hi = 1e-6
        while self._m64(corner(r_hi), label) > 0:
            r_hi *= 2
            if r_hi > 1.0:
                return self._finish(logs, c, label, False,
                                    "The margin never reached zero near c.")
        r_lo = 0.0
        for _ in range(cfg["bisect_steps"]):
            mid = (r_lo + r_hi) / 2
            if self._m64(corner(mid), label) > 0:
                r_lo = mid
            else:
                r_hi = mid
        r0 = r_hi
        logger.info("Probe point c has margin %.3e; its worst corner stops "
                    "being positive at radius r0 = %.6e", self._m64(c, label), r0)

        # 5. Ask the verifier about boxes of different sizes. Remember the
        #    certified corner with the lowest real margin.
        best_margin = min(self._real_margins(c, label).values())
        best_point, best_radius = c, 0.0
        lo, hi = 0.0, 2 * r0          # lo = certified, hi = assumed not
        for call in range(1, cfg["verifier_calls"] + 1):
            r = (lo + hi) / 2
            cert = self.verifier.certify(self.model, Region(c, r, X.clip), label)
            z = corner(r)
            real = self._real_margins(z, label)
            logs["trajectory"].append({
                "radius": r, "verifier_status": cert.status,
                "real_margin_fp32": real["fp32"],
                "real_margin_fp64": real["fp64"],
            })
            logger.info(
                "  verifier call %2d/%d: radius %.9e -> %-9s "
                "real margin at the corner: fp32 %.2e, fp64 %.2e",
                call, cfg["verifier_calls"], r, cert.status,
                real["fp32"], real["fp64"])
            if cert.verified:
                lo = r
                if min(real.values()) < best_margin:
                    best_margin, best_point, best_radius = min(real.values()), z, r
            else:
                hi = r

        at_lo = self._real_margins(corner(lo), label)
        logs["knife_edge"] = {
            "margin_at_c": self._real_margins(c, label)["fp64"],
            "r_real_threshold": r0,
            "r_verifier_threshold": lo,
            "real_margin_at_largest_certified": min(at_lo.values()),
        }
        # The certified region that actually contains adv_x.
        logs["probe_region"] = {
            "center": c.tolist(), "radius": best_radius, "clip": X.clip,
            "verifier_status": "verified" if best_radius > 0 else None,
        }

        # The property is strict (>0). A tie counts as a violation.
        violation = best_margin <= 0
        return self._finish(
            logs, best_point, label, violation,
            self._explain(violation, best_point, label, r0, lo, best_radius))

    # ---- helper methods ---------------------------------------------------
    def _point_with_margin(self, center: torch.Tensor, adv: torch.Tensor,
                           label: int, target: float) -> torch.Tensor:
        """Bisect along center -> adv until the real margin is just above target."""
        c64 = center.double()
        lo, hi = 0.0, 1.0
        for _ in range(self.config["bisect_steps"]):
            mid = (lo + hi) / 2
            if self._m64((c64 + mid * (adv - c64)).float(), label) > target:
                lo = mid
            else:
                hi = mid
        return (c64 + lo * (adv - c64)).float()

    def _worst_direction(self, c: torch.Tensor, adv: torch.Tensor,
                         label: int) -> torch.Tensor:
        """+1/-1 per pixel: the direction that shrinks the margin fastest.

        Where the gradient is exactly zero (flat ReLU regions) we move
        toward the adversarial point instead.
        """
        x = c.double().clone().requires_grad_(True)
        grad, = torch.autograd.grad(margin(self.model64(x.unsqueeze(0))[0], label), x)
        direction = -grad.sign()
        return torch.where(direction == 0, (adv - c.double()).sign(), direction)

    def _explain(self, violation: bool, point: torch.Tensor, label: int,
                 r0: float, lo: float, radius: float) -> str:
        """Build the human-readable summary string for atk_logs."""
        real = self._real_margins(point, label)
        numbers = f"fp32 {real['fp32']:.3e}, fp64 {real['fp64']:.3e}"
        if violation:
            if real["fp64"] <= 0:
                kind = ("in float64 as well, so the certificate is wrong even "
                        "in near-exact arithmetic")
            elif real["fp32"] < 0:
                kind = ("in float32 only: the float32 margin is negative while "
                        "the float64 margin is positive")
            else:
                kind = ("in float32 only: a TIE (float32 margin exactly 0) while "
                        "the float64 margin is positive; the strict property "
                        "fails and argmax picks the lowest class index")
            return (f"SOUNDNESS VIOLATION: the verifier certified radius "
                    f"{radius:.6e} around the probe point, i.e. it claims the "
                    f"label wins strictly everywhere in that box. At adv_x, a "
                    f"corner of that box, the real margin is {numbers}: {kind}. "
                    f"Re-run with the same seed to check it reproduces.")
        return (f"No soundness violation found. The verifier certified radius "
                f"{lo:.6e} around a decision-boundary point; the model's real "
                f"margin stops being positive at radius {r0:.6e}. Closest "
                f"near-miss: real margin {numbers} (assumed float32 noise: "
                f"{self.config['tolerance']:.0e}).")

    def _finish(self, logs: Dict[str, Any], adv_x: torch.Tensor, label: int,
                violation: bool, text: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Fill in the final fields of atk_logs and return (adv_x, logs)."""
        real = self._real_margins(adv_x, label)
        with torch.no_grad():
            logits = self.model(adv_x.unsqueeze(0))[0]
            logits64 = self.model64(adv_x.double().unsqueeze(0))[0]
        logs["near_miss"] = {
            "real_margin_fp32": real["fp32"],
            "real_margin_fp64": real["fp64"],
            "within_tolerance": abs(min(real.values())) < self.config["tolerance"],
        }
        logs["model_real_output"] = {
            "label": int(label),
            "logits_fp32": logits.tolist(),
            "logits_fp64": logits64.tolist(),
            "predicted": int(logits.argmax()),
            "mispredicted": bool(int(logits.argmax()) != label),
            "tie_with_label": bool((logits == logits[label]).sum() > 1),
        }
        logs["violation_found"] = bool(violation)
        logs["explanation"] = text
        logger.info("Attack finished: violation_found = %s", bool(violation))
        return adv_x, logs