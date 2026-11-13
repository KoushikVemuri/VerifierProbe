"""Tests for AlphaBetaCrownVerifier.certify() on a toy network."""
import torch
from verifier_probe import AlphaBetaCrownVerifier, Region


def toy_model():
    """Tiny 2-in 2-out network with hand-set weights.

    With x = [0.5, -0.5], class 0 wins by a margin of 1.0. That gives
    us an analytically known answer to compare against.
    """
    layer1, layer2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        layer1.weight.copy_(torch.eye(2))
        layer1.bias.zero_()
        layer2.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        layer2.bias.zero_()
    return torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2).eval()


POINT = torch.tensor([0.5, -0.5])  # class 0 wins by margin 1.0


def test_known_safe():
    # Radius 0.01 is far smaller than the margin, so the property must hold.
    region = Region(POINT, 0.01, clip=None)
    cert = AlphaBetaCrownVerifier().certify(toy_model(), region, label=0)
    assert cert.status == "verified"


def test_known_unsafe():
    # Radius 0.8 reaches inputs where class 1 wins, so the property is false.
    region = Region(POINT, 0.8, clip=None)
    cert = AlphaBetaCrownVerifier().certify(toy_model(), region, label=0)
    assert cert.status == "falsified"