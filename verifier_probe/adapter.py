"""Thin wrapper around the alpha,beta-CROWN verifier.

The verifier answers one question: "does this model predict the same
label for every input inside an L-infinity ball?" We wrap its Python
API so the rest of the project only needs certify().
"""
import contextlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """An L-infinity ball: each pixel may move by at most `eps`."""

    center: torch.Tensor           # the image we start from
    eps: float                     # how far each pixel can move
    clip: Optional[Tuple[float, float]] = (0.0, 1.0)  # valid pixel range

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) corners of the box, clamped to `clip`."""
        low = self.center - self.eps
        high = self.center + self.eps
        if self.clip is not None:
            low = low.clamp(*self.clip)
            high = high.clamp(*self.clip)
        return low, high


@dataclass
class Certificate:
    """The verifier's answer for one region."""

    status: str                            # "verified" | "falsified" | "unknown"
    label: int                             # class we asked about
    region: Region
    elapsed: float                         # seconds spent in verify()
    bab_entries: int                       # branch-and-bound iterations (0 = CROWN was enough)
    reference: Optional[dict] = None       # internal per-layer state (may be {})
    log: str = ""                          # verifier's stdout, captured

    @property
    def verified(self) -> bool:
        """True if the property was proved for the whole region."""
        return self.status == "verified"


def _import_abcrown() -> Any:
    """Import the verifier, or fail with a message that tells the user what to run."""
    try:
        import abcrown
    except ImportError as error:
        raise ImportError(
            "The alpha,beta-CROWN verifier is not installed. "
            "Run once: python scripts/install_verifier.py"
        ) from error
    return abcrown


class AlphaBetaCrownVerifier:
    """Wrapper that exposes certify() plus access to the verifier's internal state."""

    def __init__(self, bab_timeout: float = 60, quiet: bool = True,
                 use_attack: bool = True) -> None:
        self.bab_timeout = bab_timeout
        self.quiet = quiet           # hide the verifier's noisy stdout
        self.use_attack = use_attack # False = disable the verifier's own PGD step

    def _config(self) -> Any:
        """Build the verifier config: CPU, branch-and-bound, given timeout."""
        # Import here, not at module level: on Windows the verifier spawns
        # worker processes that re-import this module, and a top-level
        # import would pull in the wrong abcrown.
        abcrown = _import_abcrown()
        config = (
            abcrown.ConfigBuilder.from_defaults()
            .set("general/device", "cpu")
            .set("general/complete_verifier", "bab")
            .set("bab/timeout", self.bab_timeout)
        )
        if not self.use_attack:
            # Skip the verifier's own PGD attack so only bounds + BaB decide.
            config = config.set("attack/pgd_order", "skip")
        return config

    def _solve(self, model: nn.Module, X: Region, label: int, num_classes: int) -> Any:
        """Build the multi-class spec and ask the verifier to prove it."""
        abcrown = _import_abcrown()
        low, high = X.bounds()

        # Symbolic input/output tensors. Shape matches the model's own input.
        x_vars = abcrown.input_vars(tuple(X.center.shape))
        y_vars = abcrown.output_vars(num_classes)

        # Spec: logit[label] > logit[j] for every j != label.
        # If the verifier proves this everywhere in X, the model never
        # predicts anything other than `label` inside the ball.
        spec = None
        for j in range(num_classes):
            if j != label:
                clause = y_vars[label] > y_vars[j]
                spec = clause if spec is None else (spec & clause)

        constraints = abcrown.IOConstraints(
            input_vars=x_vars,
            output_vars=y_vars,
            input_constraint=(x_vars >= low) & (x_vars <= high),
            output_constraint=spec,
        )
        solver = abcrown.ABCrownSolver(model, x_vars, y_vars, config=self._config())
        return solver.verify(constraints=constraints)

    def certify(self, model: nn.Module, X: Region,
                label: Optional[int] = None) -> Certificate:
        """Ask: does `model` predict `label` everywhere in X?

        `label` defaults to the model's prediction at the center of X.
        """
        model = model.eval()
        with torch.no_grad():
            logits = model(X.center.detach().float().unsqueeze(0))
        if label is None:
            label = int(logits.argmax(1).item())

        # Capture stdout so the verifier's long DNF prints don't pollute
        # our logs. We still keep the text in Certificate.log for debugging.
        buffer = io.StringIO()
        start = time.time()
        if self.quiet:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = self._solve(model, X, label, logits.shape[-1])
        else:
            result = self._solve(model, X, label, logits.shape[-1])
        elapsed = time.time() - start

        stats = result.stats or {}
        logger.debug("certify radius %.6e -> %s in %.1fs", X.eps, result.status, elapsed)
        return Certificate(
            status=str(result.status),
            label=label,
            region=X,
            elapsed=elapsed,
            bab_entries=len(stats.get("bab", [])),
            reference=result.reference,
            log=buffer.getvalue(),
        )

    @staticmethod
    def internal_state(cert: Certificate) -> dict:
        """Return the verifier's own internal dict (empty if it kept nothing)."""
        return cert.reference or {}