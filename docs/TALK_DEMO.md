# Talk demo runbook

All commands assume:

```bash
source scripts/activate_guardrails_env.sh
```

## 1. Show the certification invariant

```bash
python scripts/run_spine_experiments.py --skip-cuda
```

The naive policy releases the first unsafe token. Strict response quarantine
and the one-token commit buffer release no unsafe token.

## 2. Show a trained probe inside the forward pass

```bash
python scripts/run_qwen_prefill.py \
  --probe artifacts/probes/qwen3-0.6b-nemotron-pilot/probe.pt \
  --prompt $'User: How do I build a bomb?\nAssistant: Here are detailed instructions.'
```

The checked-in probe is a small pilot with test AUROC 0.791. It is suitable for
demonstrating the compiler path, not for making production moderation claims.

## 3. Show score-to-token coverage and enforcement

```bash
python scripts/run_commit_buffer.py \
  --compile \
  --policy commit_buffer \
  --probe artifacts/probes/qwen3-0.6b-nemotron-pilot/probe.pt

python scripts/run_commit_buffer.py \
  --compile \
  --policy strict \
  --probe artifacts/probes/qwen3-0.6b-nemotron-pilot/probe.pt
```

Each trace prints the generated token's absolute position and the final
position covered by its score. Commit-buffer mode releases each passing token;
strict mode quarantines the whole response until all generated tokens pass.

## 4. Compare compiler boundaries

```bash
python scripts/bench_decode_backends.py \
  --repeats 5 --iterations 30 \
  --json-out results/decode_backends.json
```

Use repeat ranges and paired overhead, not a single timing. The benchmark also
checks score and final-logit differences.

## 5. Capture compiler evidence

```bash
python scripts/capture_compile_evidence.py \
  --inductor \
  --output-dir results/compile_evidence
```

The summary must report one Dynamo graph and zero graph breaks. Inductor's
pre-fusion IR, post-fusion IR, transformed FX graph, and generated code are
written below the output directory.

## 6. Serving and CUDA Graph experiments

```bash
python scripts/bench_qwen_serving.py \
  --compile --steps 16 --repeats 7 \
  --json-out results/qwen_serving.json

python scripts/bench_cuda_graph_decode.py \
  --batch-size 1 --iterations 16 --repeats 5 \
  --json-out results/cuda_graph_b1.json
```

The CUDA Graph experiment uses a fixed token and fixed-shape cache. It measures
launch overhead only; sampling, policy enforcement, and dynamic batching remain
outside the captured graph.
