"""Documents the trace returned by Prober.dump() and checks its shape."""
from typing import Any, Dict

TRACE_FIELDS: Dict[str, str] = {
    "input": "dict. shape (list of int), eps (float, L-infinity radius), "
             "label (int, the class asked about).",
    "source": "str. 'verifier_reference' = the verifier's own state after "
              "branch-and-bound; 'auto_lirpa_crown' = CROWN bounds recomputed "
              "by auto_LiRPA because the verifier kept no state.",
    "verdict": "dict. status ('verified' | 'falsified' | 'unknown'), "
               "bab_entries (int, 0 = no branch-and-bound was needed), "
               "elapsed_sec (float, seconds).",
    "global_lower_bound": "list of list of float with shape [1, num_classes - 1], "
                          "or None. The verifier's lower bound on the margin "
                          "(logit of the label minus the best other logit) "
                          "BEFORE branch-and-bound. Can be negative even when "
                          "the verdict is 'verified', because branch-and-bound "
                          "then closes the gap. Use verdict.status, not this "
                          "number, for the final answer.",
    "states": "list of layer states, in network order.",
    "states[].node": "str. The verifier's layer name, e.g. '/input'.",
    "states[].kind": "'hidden_layer' (bounds on a layer's values before its "
                     "ReLU) or 'output' (the last entry).",
    "states[].shape": "list of int, [batch, number of neurons]; it is the "
                      "shape of lower and upper.",
    "states[].lower": "list of list of float, shape [batch, neurons]. "
                      "Certified lower bound per neuron (same unit as the "
                      "layer's values).",
    "states[].upper": "list of list of float, shape [batch, neurons]. "
                      "Certified upper bound per neuron.",
    "states[].num_unstable": "int or None. Counted by us, not by the "
                             "verifier: neurons with lower < 0 < upper (the "
                             "ReLU could go either way). None for the output.",
}

SOURCES = ("verifier_reference", "auto_lirpa_crown")


def validate_trace(trace: Dict[str, Any]) -> None:
    """Raise ValueError if the trace doesn't match the documented shape."""
    for key in ("input", "source", "verdict", "global_lower_bound", "states"):
        if key not in trace:
            raise ValueError(f"trace is missing '{key}'")
    if trace["source"] not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if trace["verdict"]["status"] not in ("verified", "falsified", "unknown"):
        raise ValueError("verdict.status must be verified / falsified / unknown")
    for i, state in enumerate(trace["states"]):
        for key in ("node", "kind", "shape", "lower", "upper", "num_unstable"):
            if key not in state:
                raise ValueError(f"states[{i}] is missing '{key}'")
        if state["kind"] not in ("hidden_layer", "output"):
            raise ValueError(f"states[{i}].kind is invalid")