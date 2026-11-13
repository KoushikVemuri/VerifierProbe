"""Tests for Prober.dump(): the trace must match trace_schema.py."""
import json
import os

import pytest
import torch

from verifier_probe import AlphaBetaCrownVerifier, Prober, validate_trace


def toy_model():
    """Same tiny model used in the other tests."""
    layer1, layer2 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        layer1.weight.copy_(torch.eye(2))
        layer1.bias.zero_()
        layer2.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        layer2.bias.zero_()
    return torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2).eval()


def test_trace_is_valid_on_toy():
    # The toy is proved at the first bound, so the trace uses auto_LiRPA.
    prober = Prober(AlphaBetaCrownVerifier(), eps=0.01, clip=None)
    trace = prober.dump(toy_model(), torch.tensor([0.5, -0.5]))
    validate_trace(trace)
    json.dumps(trace)  # must be serializable


def test_validate_rejects_a_broken_trace():
    # A trace missing required fields must raise.
    with pytest.raises(ValueError):
        validate_trace({"input": {}})


@pytest.mark.skipif(not os.path.exists("mnist_fc.pt"), reason="train the model first")
def test_trace_on_mnist_has_three_layers():
    from torchvision import datasets, transforms
    from train_mnist import build_model

    model = build_model()
    model.load_state_dict(torch.load("mnist_fc.pt"))
    model.eval()
    test_set = datasets.MNIST("data", train=False, download=True,
                              transform=transforms.ToTensor())
    x, _ = test_set[0]
    trace = Prober(AlphaBetaCrownVerifier(bab_timeout=30), eps=0.0391).dump(model, x)
    validate_trace(trace)
    assert len(trace["states"]) == 3  # two hidden layers + output
    assert trace["states"][-1]["kind"] == "output"