"""Qwen3 forward pass with an explicit, compiler-visible activation tap."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch_compile_guardrails.scoring import SafetyScorer, ScoringBackend, load_probe


class GuardedQwen3ForCausalLM(nn.Module):
    """Run Qwen3 and score one decoder layer without materializing all layers."""

    def __init__(
        self,
        causal_lm: Qwen3ForCausalLM,
        *,
        score_layer: int | None,
        scoring_backend: ScoringBackend | str = ScoringBackend.NATIVE,
        probe_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        layer_count = causal_lm.config.num_hidden_layers
        if score_layer is not None and not 0 <= score_layer < layer_count:
            raise ValueError(f"score_layer must be in [0, {layer_count}), got {score_layer}")
        self.causal_lm = causal_lm
        self.score_layer = score_layer
        self.scorer: SafetyScorer | None = None
        if score_layer is not None:
            self.scorer = SafetyScorer(causal_lm.config.hidden_size, backend=scoring_backend)
            model_parameter = next(causal_lm.parameters())
            self.scorer.to(device=model_parameter.device, dtype=model_parameter.dtype)
            if probe_path is not None:
                weight, bias, _ = load_probe(
                    probe_path,
                    expected_hidden_size=causal_lm.config.hidden_size,
                    expected_layer=score_layer,
                    expected_model_id=causal_lm.config.name_or_path or None,
                )
                with torch.no_grad():
                    self.scorer.weight.copy_(weight)
                    self.scorer.bias.copy_(bias)
        elif probe_path is not None:
            raise ValueError("probe_path requires score_layer")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final-position next-token logits and the input-prefix score."""
        logits, score, _ = self._forward_impl(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )
        return logits, score

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Cache]:
        """Return logits, prefix score, and the updated KV cache.

        If ``input_ids`` is a pending generated token, this score is the first
        score whose activation contains that token. It is therefore the score
        that must be observed before releasing the token.
        """
        logits, score, updated_cache = self._forward_impl(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if updated_cache is None:
            raise AssertionError("cache-enabled forward did not return a cache")
        return logits, score, updated_cache

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Cache | None]:
        model = self.causal_lm.model
        config = self.causal_lm.config
        hidden_states = model.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=config)
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = (
            torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_length
        ).unsqueeze(0)

        mask_kwargs = {
            "config": config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if model.has_sliding_layers:
            causal_masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        safety_score: torch.Tensor | None = None
        for layer_index, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_masks[config.layer_types[layer_index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if layer_index == self.score_layer:
                if self.scorer is None:
                    raise AssertionError("score layer configured without a scorer")
                safety_score = self.scorer(hidden_states)

        if safety_score is None:
            if self.score_layer is not None:
                raise AssertionError("configured score layer was not executed")
            safety_score = hidden_states.new_zeros((hidden_states.shape[0],), dtype=torch.float32)
        hidden_states = model.norm(hidden_states)
        next_token_logits = self.causal_lm.lm_head(hidden_states[:, -1:, :])
        return next_token_logits, safety_score, past_key_values if use_cache else None
