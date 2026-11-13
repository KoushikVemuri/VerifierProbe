"""Tests for Attacker on the toy network and on a deliberately-unsound verifier."""
import json

import torch

from verifier_probe import AlphaBetaCrownVerifier, Attacker, Certificate, Region


def toy_model():
    """Same tiny model as in test_adapter.py (see comment there)."""
    layer1, layer2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        layer1.weight.copy_(torch.eye(2))
        layer1.bias.zero_()
        layer2.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        layer2.bias.zero_()
    return torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2).eval()


POINT = torch.tensor([0.5, -0.5])
# The toy model flips at radius 0.5, so growing to 60x (0.6) reaches it.
CONFIG = {
    "seed": 0,
    "pgd_steps": 30,
    "pgd_restarts": 40,
    "grow_factors": [10.0, 60.0],
    "knife_margin": 0.1,
    "bisect_steps": 40,
    "verifier_calls": 8,
}


class AlwaysVerified:
    """An unsound 'verifier' that certifies everything.

    If the attacker can't catch this, the attack logic is broken.
    """

    def certify(self, model, X, label=None):
        with torch.no_grad():
            pred = int(model(X.center.unsqueeze(0)).argmax(1))
        return Certificate(
            status="verified",
            label=pred if label is None else label,
            region=X, elapsed=0.0, bab_entries=0,
        )


def run_attack(verifier):
    region = Region(POINT, 0.01, clip=None)
    return Attacker(toy_model(), verifier, CONFIG).run(region)


def test_logs_have_the_required_parts():
    adv_x, logs = run_attack(AlphaBetaCrownVerifier())
    # Every key a caller might rely on must be present.
    for key in ("config", "verifier_claim", "trajectory", "knife_edge",
                "probe_region", "near_miss", "model_real_output",
                "violation_found", "explanation"):
        assert key in logs
    assert adv_x.shape == POINT.shape
    assert len(logs["trajectory"]) == CONFIG["verifier_calls"]
    json.dumps(logs)  # must be JSON-serializable


def test_same_seed_same_result():
    # Same seed and config → identical knife-edge table.
    _, logs1 = run_attack(AlphaBetaCrownVerifier())
    _, logs2 = run_attack(AlphaBetaCrownVerifier())
    assert logs1["knife_edge"] == logs2["knife_edge"]


def test_attack_catches_an_unsound_verifier():
    # The AlwaysVerified mock should be flagged as a violation.
    adv_x, logs = run_attack(AlwaysVerified())
    assert logs["violation_found"] is True
    with torch.no_grad():
        predicted = int(toy_model()(adv_x.unsqueeze(0)).argmax())
    assert predicted != logs["model_real_output"]["label"]