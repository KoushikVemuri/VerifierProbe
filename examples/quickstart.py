"""examples/quickstart.py -- the whole VerifierProbe demo in one command.

    python examples/quickstart.py

What happens:
  1. load the MNIST model (train it first if mnist_fc.pt is missing)
  2. run the Attacker against the alpha,beta-CROWN verifier
  3. run the Prober on the same input
  4. print a summary and save the logs to examples/output/

Optional arguments:
  --image N       which MNIST test image to use (default 0)
  --eps R         L-infinity radius of the region X (default 0.034)
  --config FILE   JSON settings for the attack (default examples/config.json)
"""
import argparse
import json
import logging
import os
import pathlib
import sys
import warnings

# torch.jit emits a DeprecationWarning on import; harmless, just noisy.
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so train_mnist.py can be imported

import torch
from torchvision import datasets, transforms

from train_mnist import build_model, main as train_model
from verifier_probe import AlphaBetaCrownVerifier, Attacker, Prober, Region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VerifierProbe quickstart")
    parser.add_argument("--image", type=int, default=0,
                        help="MNIST test image index")
    parser.add_argument("--eps", type=float, default=0.034,
                        help="L-infinity radius")
    parser.add_argument("--config", default=str(ROOT / "examples" / "config.json"),
                        help="JSON file with the attack settings")
    return parser.parse_args()


def check_verifier_installed() -> None:
    """Stop early with a helpful message if abcrown can't be imported."""
    try:
        import abcrown  # noqa: F401
    except ImportError as error:
        sys.exit("The alpha,beta-CROWN verifier is not installed "
                 f"({error}).\nRun once:  python scripts/install_verifier.py")


def load_model() -> torch.nn.Module:
    """Load the trained network; train it first if the weights file is missing."""
    weights = ROOT / "mnist_fc.pt"
    if not weights.exists():
        print("mnist_fc.pt not found: training the model first (about a minute)...")
        train_model()  # saves mnist_fc.pt in the current folder
    model = build_model()
    model.load_state_dict(torch.load(weights))
    return model.eval()


def report_attack(logs: dict) -> None:
    """Print the attacker's summary in a human-readable form."""
    claim = logs["verifier_claim"]
    print(f"verifier's claim about X (radius {claim['eps']}): {claim['status']}")
    print("soundness violation found:", logs["violation_found"])
    print(logs["explanation"])

    real = logs["model_real_output"]
    print(f"\nat adv_x: true label {real['label']}, the float32 model predicts "
          f"{real['predicted']} (tie with the label: {real['tie_with_label']})")
    print("float32 logits:", [round(v, 9) for v in real["logits_fp32"]])
    print("float64 logits:", [round(v, 9) for v in real["logits_fp64"]])

    edge = logs.get("knife_edge")
    if edge:
        print("\nradius where the real margin stops being positive:",
              edge["r_real_threshold"])
        print("largest radius the verifier certifies           :",
              edge["r_verifier_threshold"])


def report_trace(trace: dict) -> None:
    """Print the prober's trace summary."""
    print("source of the numbers:", trace["source"])
    print("verdict:", trace["verdict"]["status"],
          "| branch-and-bound runs:", trace["verdict"]["bab_entries"])
    for state in trace["states"]:
        unstable = ("(output layer)" if state["num_unstable"] is None
                    else f"unstable ReLUs: {state['num_unstable']}")
        print(f"  layer {state['node']:<10} shape {state['shape']}  {unstable}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(ROOT)  # train_mnist and the MNIST download use relative paths
    check_verifier_installed()

    model = load_model()
    test_set = datasets.MNIST("data", train=False, download=True,
                              transform=transforms.ToTensor())
    x, true_label = test_set[args.image]
    print(f"MNIST test image {args.image} (true label {true_label}), "
          f"region X = image +- {args.eps}")

    # use_attack=False: the verifier's own PGD attack is off. Only its
    # bounds and branch-and-bound decide the answer — those are what we probe.
    verifier = AlphaBetaCrownVerifier(bab_timeout=30, use_attack=False)

    print("\n[1/2] Attacker: the log below shows every question asked to the "
          "verifier (a few minutes).")
    attacker = Attacker(model, verifier, args.config)
    adv_x, atk_logs = attacker.run(Region(x, args.eps))
    print()
    report_attack(atk_logs)

    print("\n[2/2] Prober: the verifier's per-layer bounds for the same input.")
    trace = Prober(verifier, eps=args.eps).dump(model, x)
    report_trace(trace)

    out = ROOT / "examples" / "output"
    out.mkdir(exist_ok=True)
    (out / "atk_logs.json").write_text(json.dumps(atk_logs, indent=2))
    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    torch.save(adv_x, out / "adv_x.pt")
    print("\nSaved atk_logs.json, trace.json and adv_x.pt to", out)


if __name__ == "__main__":
    main()