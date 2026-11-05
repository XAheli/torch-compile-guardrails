# torch-compile-guardrails

Prototype for the SciPy India 2026 talk *“Compiling Guardrails: LLM Safety Checks with torch.compile”*.

## Thesis

Compiling the classifier is easy. Compiling the **enforcement semantics**—under asynchronous, batched, speculative serving—is the real problem.

The first measurements rejected the overly broad claim that synchronization
necessarily costs more than scoring. The durable question is instead: **what
release semantics remain correct, and what useful overlap do they sacrifice?**

## Scoring backends

The repository now implements one numerical contract three ways. Given
activations shaped `[batch, sequence, hidden]`, every backend scores the final
position with the same frozen linear probe and returns one fp32 probability per
batch item.

| Backend | Compiler boundary |
|---|---|
| `native` | PyTorch operations visible to Dynamo and Inductor |
| `custom_op` | Opaque `torch.library.custom_op` with a FakeTensor kernel |
| `triton_op` | Triton kernel exposed through `torch.library.triton_op` |

All three pass eager numerical checks and `torch.compile(fullgraph=True)` on the
H200 MIG. The opaque operator additionally passes `torch.library.opcheck`.

```bash
python scripts/compare_scoring_backends.py \
  --json-out results/scoring_backends.json
```

The first real-model vertical slice explicitly runs the Qwen3 decoder loop,
scores one selected layer, and returns final-position logits from the same
compiled call. It does not use forward hooks or materialize every layer:

```bash
python scripts/run_qwen_prefill.py --model Qwen/Qwen3-0.6B --backend native
```

The repository includes both synthetic probes for controlled backend comparison
and a checked-in trained pilot for the talk demo. The pilot is bound to
Qwen3-0.6B layer 14 and was trained on a balanced 1,000-response subset of
NVIDIA's Nemotron content-safety dataset. Its untouched 200-example test split
reached AUROC 0.791 and accuracy 0.665. This is useful for demonstrating the
execution path, but is not a production safety classifier.

## Spine experiments (start here)

Two tiny experiments decide whether this project has a talk spine. No LLM required.

1. **One-token certification gap** — deterministic policy simulation proving that release-before-certify leaks exactly one unsafe token, while a one-token commit buffer does not.
2. **`.item()` sync tax** — CUDA burst microbenchmark: pipeline N score launches + one sync vs `.item()` every step (fair host-wall timing; per-iter event sync hides the tax).

```bash
cd /workspaces/pytorch-devcontainers/torch-compile-guardrails
source scripts/activate_guardrails_env.sh

pytest tests/ -vv
python scripts/run_spine_experiments.py --json-out results/spine.json
```

### Measured so far (H200 MIG 3g.71gb)

| Experiment | Result |
|---|---|
| Certification gap | **Confirmed.** `release_before_certify` leaks token 3; `one_token_commit_buffer` leaks 0 (same released safe prefix `[0,1,2]`). |
| `.item()` sync tax | **Workload-dependent.** Score burst: pipeline ≈ 0.028 ms/step, per-step `.item()` ≈ 0.032 ms/step (~1.14×). Launch and Python dispatch dominate this tiny-score regime. |

The correctness half of the spine is solid. Performance claims remain measured
results, not assumptions.

## Release-correct cached decode

The cache-aware path makes score coverage explicit. Prefill predicts a token
but does not certify it. The candidate remains quarantined until a one-token
KV-cached decode consumes it and produces an activation score that includes
that token. Only a passing candidate is released:

```bash
python scripts/run_commit_buffer.py --max-new-tokens 8
python scripts/run_commit_buffer.py --max-new-tokens 8 --compile
python scripts/run_commit_buffer.py --max-new-tokens 8 --compile --policy strict
```

Tests compare cached logits and scores against full-prefix recomputation and
assert that a blocked token never enters the released sequence. The current
reference policy supports batch size one; continuous batching is later work.
The demo defaults to a fixed-capacity `StaticCache`: measured over five decode
steps it reused one compiled graph, whereas `DynamicCache` required a second
graph when its KV tensor first grew.
Prefill intentionally runs eagerly: this initializes the static cache on CUDA
and marks stable addresses before the compiled, fixed-shape decode region.

`commit_buffer` releases each token immediately after its covering score passes.
`strict` quarantines the entire response and releases it only if every generated
token passes. Every verified-token record includes its absolute position and the
last position covered by the score.

## Evidence and demo

- [Talk demo runbook](docs/TALK_DEMO.md)
- [Validated claims and limitations](docs/VALIDATION.md)
- [vLLM integration boundary](docs/VLLM_INTEGRATION.md)
- [Pilot probe model card](artifacts/probes/qwen3-0.6b-nemotron-pilot/README.md)

The real-model benchmark suite includes backend comparisons, prefill/decode and
host-synchronization timing, batch-size sweeps, manual CUDA Graph replay, and a
Dynamo/Inductor evidence exporter. Generated measurements go under `results/`
and are intentionally not committed.

## Environment

Dedicated **Python 3.12** `.venv`. PyTorch / Triton are linked from the local editable build in [`../pytorch`](../pytorch) (not installed from PyPI).

```bash
# one-time
uv sync --extra dev --python 3.12 --no-install-package torch --no-install-package triton
bash scripts/link_local_torch.sh

# every shell
source scripts/activate_guardrails_env.sh
```

Add `--extra train` to `uv sync` when reproducing the probe artifact.

Optional MIG pin:

```bash
GUARDRAILS_MIG_UUID=<your-MIG-UUID> \
  source scripts/activate_guardrails_env.sh
```
