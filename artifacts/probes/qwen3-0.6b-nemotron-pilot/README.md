# Qwen3-0.6B Nemotron linear probe pilot

This is a research pilot, not a production safety classifier.

- Backbone: `Qwen/Qwen3-0.6B`
- Activation: decoder layer 14, final non-padding prompt token
- Dataset: 1,000 labeled English responses from `nvidia/Nemotron-3.5-Content-Safety-Dataset`
- Input format: `User: {prompt}\nAssistant: {response}`
- Class balance: 500 safe / 500 unsafe
- Split: stratified 600 train / 200 validation / 200 test, seed 0
- Classifier: one fp32 linear logit with sigmoid at inference
- Test accuracy at threshold 0.5: 0.665
- Test AUROC: 0.791
- Artifact SHA-256: `53cd9414681ada28d5948fb9be60ff701c1bad1aafeef95761084c2f48b18ab3`

The validation set selects the training epoch. Metrics are reported once on the
separate test split. This small, single-seed, in-distribution evaluation does
not establish jailbreak robustness, calibration, or deployment readiness. The
training labels apply to complete responses, so token-prefix calibration during
streaming is also unvalidated; a production probe needs a rolling objective or
token-level labels.

Reproduce with:

```bash
python scripts/prepare_nemotron_probe_data.py \
  --target response \
  --output results/data/nemotron_response_probe_1000.jsonl \
  --examples-per-class 500

python scripts/train_linear_probe.py \
  --data results/data/nemotron_response_probe_1000.jsonl \
  --dataset-name nvidia/Nemotron-3.5-Content-Safety-Dataset:balanced-en-response-1000 \
  --output artifacts/probes/qwen3-0.6b-nemotron-pilot/probe.pt \
  --metrics-out artifacts/probes/qwen3-0.6b-nemotron-pilot/metrics.json
```
