#!/usr/bin/env python
"""Capture Dynamo graph statistics and optional TorchInductor debug artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch._dynamo import explain
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.models import GuardedQwen3ForCausalLM
from torch_compile_guardrails.scoring import ScoringBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--backend", choices=[b.value for b in ScoringBackend], default="native")
    parser.add_argument("--output-dir", type=Path, default=Path("results/compile_evidence"))
    parser.add_argument("--inductor", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    if not isinstance(base, Qwen3ForCausalLM):
        raise TypeError(f"expected Qwen3ForCausalLM, got {type(base).__name__}")
    torch.manual_seed(0)
    guarded = GuardedQwen3ForCausalLM(
        base,
        score_layer=base.config.num_hidden_layers // 2,
        scoring_backend=args.backend,
    ).eval()
    prompt = tokenizer("Compiler evidence:", return_tensors="pt").input_ids.to("cuda")
    token = torch.tensor([[tokenizer.eos_token_id - 1]], device="cuda")

    with torch.no_grad():
        cache = StaticCache(config=base.config, max_cache_len=32)
        guarded.forward_with_cache(prompt, None, cache)
        explanation = explain(guarded.forward_with_cache)(token, None, cache)

    operator_names = [
        str(operation)
        for operations in (explanation.ops_per_graph or [])
        for operation in operations
    ]
    summary = {
        "model": args.model,
        "backend": args.backend,
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "op_count": explanation.op_count,
        "break_reasons": [reason.reason for reason in explanation.break_reasons],
        "operator_histogram": dict(Counter(operator_names).most_common()),
        "ops_per_graph": [
            [str(operation) for operation in operations]
            for operations in (explanation.ops_per_graph or [])
        ],
    }
    (args.output_dir / "dynamo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for index, graph in enumerate(explanation.graphs):
        (args.output_dir / f"dynamo_graph_{index}.py").write_text(graph.print_readable(False))

    if args.inductor:
        import torch._inductor.config as inductor_config

        inductor_config.trace.enabled = True
        inductor_config.trace.debug_dir = str(args.output_dir / "inductor")
        torch.compiler.config.force_disable_caches = True
        torch._dynamo.reset()
        cache = StaticCache(config=base.config, max_cache_len=32)
        with torch.no_grad():
            guarded.forward_with_cache(prompt, None, cache)
            compiled = torch.compile(guarded.forward_with_cache, fullgraph=True)
            compiled(token, None, cache)
            torch.cuda.synchronize()
        required_artifacts = {
            "fx_graph_readable.py",
            "ir_pre_fusion.txt",
            "ir_post_fusion.txt",
            "output_code.py",
        }
        emitted = {path.name for path in (args.output_dir / "inductor").rglob("*")}
        missing = required_artifacts - emitted
        if missing:
            raise RuntimeError(f"Inductor trace did not emit required artifacts: {sorted(missing)}")

    console_summary = {
        "model": summary["model"],
        "backend": summary["backend"],
        "graph_count": summary["graph_count"],
        "graph_break_count": summary["graph_break_count"],
        "op_count": summary["op_count"],
        "break_reasons": summary["break_reasons"],
    }
    print(json.dumps(console_summary, indent=2))
    print(f"wrote evidence under {args.output_dir}")


if __name__ == "__main__":
    main()
