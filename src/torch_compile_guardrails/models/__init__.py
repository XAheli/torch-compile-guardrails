"""Model integrations that expose a compiler-visible activation tap."""

from .qwen3 import GuardedQwen3ForCausalLM

__all__ = ["GuardedQwen3ForCausalLM"]
