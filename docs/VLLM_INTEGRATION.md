# vLLM integration boundary

The repository does not claim a completed vLLM integration yet.

The local vLLM source checkout is at commit `5ecd8e9c70`, but its existing
environment is internally inconsistent: its executable resolves Python 3.14
while installed packages live under Python 3.10, and no usable Torch or vLLM
extension is present.

A custom logits processor is insufficient because it sees logits, not the
selected decoder-layer activation. A correct integration needs all of:

1. An out-of-tree registered Qwen3 model that computes the probe score at the selected decoder layer.
2. A typed model-runner output carrying `(request slot, token position, score)` alongside logits.
3. Scheduler state that preserves this coverage mapping under continuous batching, slot reuse, preemption, and speculative decode.
4. A serving release gate that quarantines text until the matching score is available.
5. Tests for request reordering, cancellation, cache preemption, and a blocked token at every batch slot.

This is intentionally not implemented as a side-channel Python attribute: that
would be unsafe under `torch.compile`, CUDA Graph replay, tensor parallelism,
and multi-process workers. vLLM supports out-of-tree model registration through
its plugin system, but the score still needs an explicit runner/scheduler
contract before the integration can be called complete.
