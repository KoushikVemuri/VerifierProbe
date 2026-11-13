# VerifierProbe

A small toolbox for testing a neural-network verifier called **α,β-CROWN**.

- **Attack:** it looks for an input that the verifier says is "safe" but that
  the real model gets wrong.
- **Prober:** it saves what the verifier computes inside, layer by layer, in a
  documented file.

Everything runs on **CPU**. No GPU is needed.

**At a glance:** on a small MNIST network I found one case where the verifier
says "verified" and the real float32 model gives a different answer. It is a
tiny rounding tie, not a clear mistake, and it only happens in float32 (details
in [Results](#5-results)).

---

## 1. Install

**You need:** Python 3.11, `git`, and an internet connection (the installer
downloads the verifier and MNIST). PyTorch is a big download; on Linux the
default build is several GB because it includes GPU libraries, but it runs
fine on CPU.

**Step 1: create a clean Python 3.11 environment.** Pick **one** option, in the
project folder.

| Option | Commands |
|---|---|
| **A. conda** (Windows / Linux / macOS) | `conda env create -n verifierprobe -f environment.yml`<br>`conda activate verifierprobe` |
| **B. plain Python, Linux / macOS** | `python3.11 -m venv .venv`<br>`source .venv/bin/activate` |
| **C. plain Python, Windows (PowerShell)** | `py -3.11 -m venv .venv`<br>`.venv\Scripts\Activate.ps1` |

Notes: in option C, if PowerShell blocks the script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. If you do
not have Python 3.11, install it from python.org, or use option A (conda
installs it for you).

Check that `python --version` prints `3.11.x` and that your prompt shows
`(verifierprobe)` or `(.venv)`.

**Step 2: install the package and the verifier** (same for every option):

```
pip install -e .
python scripts/install_verifier.py
```

The second command downloads α,β-CROWN at a fixed version (release 0.8.0) into
`third_party/`, installs it, and prints `alpha-beta-CROWN import OK` when it
works.

> **Tested on:** Windows 11, Python 3.11, conda. The pip/venv and Linux/macOS
> steps use the same commands but I have not run them.

## 2. Run

```
python examples/quickstart.py     # the whole demo, a few minutes
python -m pytest tests -q         # 12 tests, about a minute
```

The quickstart loads the small network (`mnist_fc.pt`, included), runs the
Attacker, runs the Prober, prints a summary and saves `atk_logs.json`,
`trace.json` and `adv_x.pt` into `examples/output/`. Optional arguments:
`--image N` (which MNIST test image), `--eps R` (size of the region X),
`--config FILE` (attack settings, default `examples/config.json`).

The files already in `examples/output/` are from the run I report below.
Running the quickstart overwrites them with your own run.

**On another computer:** the finding depends on how the processor rounds
numbers, so the last digits may differ and the tie may not appear. The
quickstart still finishes and reports the closest near-miss, which the task
accepts as a valid result.

## 3. Usage

```python
from verifier_probe import AlphaBetaCrownVerifier, Attacker, Prober, Region

verifier = AlphaBetaCrownVerifier(bab_timeout=30, use_attack=False)

atk = Attacker(model, verifier, config)          # config: a dict or a JSON file path
adv_x, atk_logs = atk.run(Region(x, 0.034))      # X = image x, each pixel +- 0.034

prober = Prober(verifier, eps=0.034)             # eps is optional (default 0.03)
trace = prober.dump(model, x)
```

`model` is a PyTorch network, `x` is one image tensor (for MNIST: shape
`[1, 28, 28]`). On Windows, put your script's main code under
`if __name__ == "__main__":`, because the verifier starts extra processes that
re-run the script.

### Config (dict or JSON, all keys optional, unknown keys are rejected)

| key | meaning | default |
|---|---|---|
| `seed` | makes the search repeatable | 0 |
| `label` | class to certify (`null` = the model's own prediction) | null |
| `tolerance` | assumed float32 noise on a margin | 1e-6 |
| `pgd_steps`, `pgd_restarts` | gradient search: steps per start, number of starts | 100, 20 |
| `grow_factors` | how much to grow the radius when looking for an adversarial image | 1.1 ... 3.0 |
| `knife_margin` | margin of the probe point next to the decision edge | 1e-3 |
| `bisect_steps` | precision of the model-side bisections | 60 |
| `verifier_calls` | budget: how many times the attack may ask the verifier | 25 (15 in `examples/config.json`) |

### What `atk_logs` contains

`config`, `verifier_claim` (the verifier's answer for X), `trajectory` (every
question asked to the verifier: radius, answer, the model's real float32 and
float64 margin), `knife_edge` (real threshold radius vs. the largest radius the
verifier certifies), `probe_region`, `near_miss` (the closest approach to a
violation), `model_real_output` (float32 and float64 scores at `adv_x`, the
predicted class, whether there is a tie), `violation_found`, and a plain-text
`explanation`.

### What `trace` contains (the documented schema)

It is a JSON-serializable dict. The same text lives in
`verifier_probe/trace_schema.py`, and `validate_trace` checks every trace
against it.

| field | meaning | type / shape |
|---|---|---|
| `input` | `shape` (list of int), `eps` (float, L-infinity radius), `label` (int, class asked about) | dict |
| `source` | where the numbers come from: `verifier_reference` = the verifier's own state after branch-and-bound; `auto_lirpa_crown` = recomputed with auto_LiRPA because the verifier kept no state | str |
| `verdict` | `status` (`verified` / `falsified` / `unknown`), `bab_entries` (0 = no branch-and-bound needed), `elapsed_sec` (seconds) | dict |
| `global_lower_bound` | verifier's lower bound on the margin *before* branch-and-bound (can be negative even when the verdict is `verified`) | `[[float]]`, shape `[1, 1]`, or null |
| `states` | one entry per layer, in network order | list |
| `states[].node` | the verifier's name for the layer, e.g. `/input` | str |
| `states[].kind` | `hidden_layer` (values before the ReLU) or `output` (last entry) | str |
| `states[].shape` | shape of `lower` and `upper` | `[batch, neurons]` |
| `states[].lower`, `.upper` | certified lowest and highest value of each neuron inside the box (same scale as the network's scores) | `[[float]]`, shape `[batch, neurons]` |
| `states[].num_unstable` | neurons whose ReLU could go either way (lower < 0 < upper); counted by me, not the verifier; null for the output | int or null |

For the MNIST network the trace has 3 states: `/input` `[1, 64]` (21 unstable
ReLUs), `/input-3` `[1, 32]` (7 unstable) and `/12` `[1, 1]` (output).

---

## 4. Method

### Which verifier, and why
**α,β-CROWN.** It takes an image and an allowed change (for example every
pixel may move by ±0.034) and tries to *prove* that the model gives the same
answer for every possible change. If it succeeds it says `verified`. I chose it
because it is a well-known winning verifier, it runs on CPU, and it has a Python
interface. I run it with its branch-and-bound mode (the other refinement mode
crashed on Windows in a multiprocessing step) and a 30-second limit per
question.

### Sound and complete
- **Sound:** if the verifier says `verified`, no adversarial example exists in
  the region. This holds for exact maths.
- **Complete:** given enough time, branch-and-bound always reaches an answer:
  `verified`, or `falsified` with a counterexample. Because I limit the time, it
  can also answer `unknown`.

### What I hooked (instrumentation), and why it does not change the verifier
**I did not modify the verifier's source code.** The verifier already returns
its internal numbers in its result object, so no hook was needed.

- `adapter.py`, function `certify()`: the only place that calls the verifier.
  It builds the question ("the label's score beats every other score for all
  inputs in the box"), runs `ABCrownSolver.verify()`, and keeps the returned
  `result.reference` and the number of branch-and-bound runs in a `Certificate`.
- `prober.py`, function `dump()`: reads `lower_bounds`, `upper_bounds` and
  `global_lb` from that `reference`. This happens *after* the verifier has
  finished and only reads numbers, so it cannot change what the verifier decides.
- Fallback: when the verifier proves the property at its very first step it
  keeps no internal numbers. Then `Prober` computes the same kind of per-layer
  bounds with auto_LiRPA (the library inside the verifier) in a separate
  object, and the trace says so in its `source` field. The verifier's own run
  is not touched.
- Settings I give the verifier (its own configuration options): run on CPU, use
  branch-and-bound, set the time limit, and optionally (`use_attack=False`)
  skip its built-in gradient attack. Skipping that attack only removes the
  search for counterexamples; the proof part (bounds and branch-and-bound) is
  unchanged. I do this so that only the proof decides the answer.

The exact verifier version is pinned in `scripts/install_verifier.py`.

### The numerical gap I try to exploit, and why the attack looks where it does
The paper behind this task (Jia & Rinard) shows that a verifier can be wrong
because of rounding. For α,β-CROWN:

- The verifier answers `verified` when its computed lower bound on the margin
  is positive. I believe it computes in float32 without directed rounding (I
  did not check its source).
- The real model also runs in float32 and rounds after every layer. *float32*
  keeps about 7 digits; *float64* keeps about 16 and is my "almost exact"
  reference.
- The two can disagree by a few float32 steps. That only changes an answer
  when the model's top two scores are almost equal. So random images are
  useless; the attack goes to the place where two scores are nearly equal: the
  decision edge.

**The attack, step by step**
1. Ask the verifier about X. If it does not say `verified`, stop.
2. Find a real adversarial image a bit further away (gradient search, float64).
3. Walk from the original image toward it and stop just before the answer
   flips. Call this point `c`. It sits on the edge.
4. In a small box around `c`, the corner where the model is most likely to be
   wrong is known (every pixel moves against the gradient).
5. Ask the verifier about boxes of different sizes (bisection) to find the
   largest box it still calls `verified`. At each box's corner compute the
   model's scores in float32 and in float64.
6. If the verifier said `verified` but the real float32 scores no longer make
   the right class win strictly, that is a violation. A tie counts, because the
   verifier proved a strict "greater than".

If no violation is found, the attack returns the closest near-miss and
explains why (`atk_logs["near_miss"]`, `["explanation"]`).

### Reproducibility
`seed` controls the only random part (the random starting points of the
gradient search); the bisections are deterministic. The same seed and config
gave the same result on repeated runs, and in a freshly created environment
after retraining the model. A test checks this.

---

## 5. Results

MNIST test image 0 (a handwritten 7), small network 784-64-32-10, 96.3%
accuracy, region X = image ±0.034. The saved logs are in `examples/output/`
(`atk_logs.json`, `trace.json`, `adv_x.pt`), and the printed output of the run
is at the end of this section.

1. The verifier proves the network answers 7 for every image within ±0.034
   (about 14 seconds). A real adversarial image exists at ±0.0425.
2. Near the decision edge the verifier is almost exact. It certifies boxes up
   to radius 5.52986e-06; the real threshold (where the worst corner's float64
   margin reaches zero) is 5.53019e-06, a gap of about 0.006%.
3. **One case where the verifier and the real model disagree.** The verifier
   says `verified` for a box of radius 5.527493e-06. At one corner of that box:

   | | score of class 7 (true) | score of class 3 | winner |
   |---|---|---|---|
   | float64 (precise) | 5.279089183 | 5.279088586 | 7, by 0.0000006 |
   | float32 (how it really runs) | 5.27908802 | 5.27908802 | exact tie, the computer picks the lower number: **3** |

4. The end of the search, as the verifier is asked about boxes just below the
   real threshold:

   | call | radius | verifier says | float32 margin | float64 margin |
   |---|---|---|---|---|
   | 1 | 5.530193e-06 | unknown | -4.77e-07 | -9.25e-08 |
   | 11 | 5.524793e-06 | verified | 1.91e-06 | 1.18e-06 |
   | **12** | **5.527493e-06** | **verified** | **0.00** | **5.97e-07** |
   | 13 | 5.528843e-06 | verified | 4.77e-07 | 4.84e-07 |
   | 15 | 5.529856e-06 | verified | 4.77e-07 | 4.81e-07 |

   (4.77e-07 is the gap between two neighbouring float32 numbers at this size.)

**Did I reproduce a genuine soundness violation? Yes, but only a marginal one.**
The verifier certified a box in which the exact (float64) margin is positive,
but the real float32 model computes a tie at one corner and therefore answers
3. The float32 scores are off from the float64 ones by about 1.2e-6 and
5.7e-7, which is one to two float32 steps. It is a rounding tie, not a clear
mistake, and it is not a violation in float64. At the neighbouring radii the
float32 margin sits one step above zero, so this is a rounding-level event at
one radius, not a wide gap. It does not mean the verifier's mathematics is
wrong: its promise holds on paper and breaks by a hair because real computers
round numbers.

### Sample output

Output of `python examples/quickstart.py` (the PyTorch deprecation warning at
the top is left out):

```
MNIST test image 0 (true label 7), region X = image +- 0.034

[1/2] Attacker: the log below shows every question asked to the verifier (a few minutes).
Attack: region X = point +- 0.034, seed 0
Verifier's claim about X: verified (14.3 s)
Found a real adversarial point at radius 0.0425
Probe point c has margin 1.000e-03; its worst corner stops being positive at radius r0 = 5.530193e-06
  verifier call  1/15: radius 5.530193221e-06 -> unknown   real margin at the corner: fp32 -4.77e-07, fp64 -9.25e-08
  verifier call  2/15: radius 2.765096610e-06 -> verified  real margin at the corner: fp32 5.02e-04, fp64 5.01e-04
  verifier call  3/15: radius 4.147644916e-06 -> verified  real margin at the corner: fp32 2.50e-04, fp64 2.50e-04
  verifier call  4/15: radius 4.838919068e-06 -> verified  real margin at the corner: fp32 1.25e-04, fp64 1.26e-04
  verifier call  5/15: radius 5.184556144e-06 -> verified  real margin at the corner: fp32 6.29e-05, fp64 6.27e-05
  verifier call  6/15: radius 5.357374683e-06 -> verified  real margin at the corner: fp32 3.10e-05, fp64 3.16e-05
  verifier call  7/15: radius 5.443783952e-06 -> verified  real margin at the corner: fp32 1.76e-05, fp64 1.66e-05
  verifier call  8/15: radius 5.486988586e-06 -> verified  real margin at the corner: fp32 1.00e-05, fp64 8.29e-06
  verifier call  9/15: radius 5.508590903e-06 -> verified  real margin at the corner: fp32 4.29e-06, fp64 4.57e-06
  verifier call 10/15: radius 5.519392062e-06 -> verified  real margin at the corner: fp32 2.38e-06, fp64 1.82e-06
  verifier call 11/15: radius 5.524792641e-06 -> verified  real margin at the corner: fp32 1.91e-06, fp64 1.18e-06
  verifier call 12/15: radius 5.527492931e-06 -> verified  real margin at the corner: fp32 0.00e+00, fp64 5.97e-07
  verifier call 13/15: radius 5.528843076e-06 -> verified  real margin at the corner: fp32 4.77e-07, fp64 4.84e-07
  verifier call 14/15: radius 5.529518148e-06 -> verified  real margin at the corner: fp32 4.77e-07, fp64 4.82e-07
  verifier call 15/15: radius 5.529855684e-06 -> verified  real margin at the corner: fp32 4.77e-07, fp64 4.81e-07
Attack finished: violation_found = True

verifier's claim about X (radius 0.034): verified
soundness violation found: True
SOUNDNESS VIOLATION: the verifier certified radius 5.527493e-06 around the probe point, i.e. it claims the label wins strictly everywhere in that box. At adv_x, a corner of that box, the real margin is fp32 0.000e+00, fp64 5.971e-07: in float32 only: a TIE (float32 margin exactly 0) while the float64 margin is positive; the strict property fails and argmax picks the lowest class index. Re-run with the same seed to check it reproduces.

at adv_x: true label 7, the float32 model predicts 3 (tie with the label: True)
float32 logits: [-3.940839767, -5.324691772, 1.476958752, 5.27908802, -10.487765312, 0.169752151, -14.058846474, 5.27908802, -1.456673265, -2.296215534]
float64 logits: [-3.940840154, -5.324691971, 1.476959865, 5.279088586, -10.487766197, 0.16975235, -14.058844689, 5.279089183, -1.456673388, -2.296216151]

radius where the real margin stops being positive: 5.530193220693036e-06
largest radius the verifier certifies           : 5.529855684485718e-06

[2/2] Prober: the verifier's per-layer bounds for the same input.
Prober: radius 0.034, verdict verified, 3 layers from verifier_reference
source of the numbers: verifier_reference
verdict: verified | branch-and-bound runs: 1
  layer /input     shape [1, 64]  unstable ReLUs: 21
  layer /input-3   shape [1, 32]  unstable ReLUs: 7
  layer /12        shape [1, 1]  (output layer)

Saved atk_logs.json, trace.json and adv_x.pt to <project folder>/examples/output
```

Output of `python -m pytest tests -q`:

```
............                                                  [100%]
12 passed, 1 warning in 61.92s
```

## 6. Limitations

- One model and one image (the task asks for one region). I did not measure how
  often this happens.
- It is a tie, not a sign flip, and it exists only in float32. Another computer
  may round differently, so it may not repeat there.
- The box where it appears is one I built near the edge
  (`atk_logs["probe_region"]`), not inside the original X.
- I switched off the verifier's own gradient attack (`use_attack=False`).
- I only compared float32 (batch size 1) and float64.
- The verifier does not expose its internal search steps (the "leaf nodes"
  mentioned in the task), so I aimed the attack at the decision edge instead.

## 7. Files and tests

- `verifier_probe/adapter.py`: talks to the verifier (`certify`).
- `verifier_probe/attacker.py`: the attack.
- `verifier_probe/prober.py`, `trace_schema.py`: build and check the trace.
- `verifier_probe/config.py`: settings, loading from dict or JSON, checking.
- `examples/quickstart.py`, `examples/config.json`: the demo and its settings.
- `examples/output/`: the logs of the run reported above (overwritten when you
  run the quickstart).
- `scripts/install_verifier.py`: downloads and installs α,β-CROWN.
- `train_mnist.py`, `mnist_fc.pt`: the model and its training script. The result
  depends on these exact weights.
- `tests/` (12 tests): on a tiny hand-made network, one case that must be
  `verified` and one that must be `falsified`; trace-format checks; config
  checks; and attack tests, including one where the attack runs against a fake
  verifier that always says "safe" and must catch it.

## 8. Troubleshooting

- *"The alpha,beta-CROWN verifier is not installed":* run
  `python scripts/install_verifier.py`.
- A pytest warning about `asyncio_default_fixture_loop_scope` is harmless; it
  comes from a plugin installed with the verifier.
