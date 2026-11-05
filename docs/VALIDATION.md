# Validation status

## Confirmed

| Claim | Evidence |
|---|---|
| Cached token decode matches full-prefix recomputation | Unit tests compare logits and scores |
| Score covers the pending token before release | Coverage ledger records equal token and covered-through positions |
| Blocked tokens do not cross the serving boundary | Strict and commit-buffer policy tests |
| StaticCache supports one fixed-shape compiled decode graph | Dynamo counters and real Qwen benchmark |
| Native, opaque custom op, and Triton op compile full-graph | CUDA tests and full-model benchmark |
| Real cached decode is one Dynamo graph with no break | 1 graph, 0 breaks, 1,315 operations on Qwen3-0.6B |
| Manual CUDA Graph replay preserves outputs | Exact logits and score agreement at batch 1 and 4 |

## H200 MIG measurements

These are measurements from this machine, not general performance claims.

- Compiled batch-1 decode: approximately 6.0 ms/token.
- Per-step score `.item()` added approximately 0.047 ms/token in the seven-repeat full-model run.
- Batch-4 compiled scoring overhead was within measurement noise: native 0.22%, opaque custom op 0.66%, Triton op 0.33% paired median overhead versus no scorer.
- Manual fixed-token CUDA Graph replay: 4.05 ms/token at batch 1 versus 21.27 ms eager; 4.26 ms/token at batch 4 versus 21.40 ms eager.
- Compiled backend score differences reached about 4.5e-4 while final logits remained equal. Different reduction/fusion orders therefore require tolerance-aware policy calibration.

## Not yet confirmed

- Robustness of the pilot probe to jailbreaks or distribution shift.
- Calibration of a deployment threshold and false-positive rate.
- Continuous-batching release semantics.
- End-to-end vLLM integration. The available vLLM environment is currently unusable and activation scores require a model-runner output path, not a logits-only plugin.
- CUDA Graph capture of sampling and enforcement. Current capture intentionally covers only fixed-token decode.
