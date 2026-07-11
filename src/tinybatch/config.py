"""Engine configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Knobs that govern memory layout and scheduling.

    The defaults are tuned for Qwen2.5-0.5B-Instruct on an Apple-Silicon Mac.
    """

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"                # "auto" | "cpu" | "mps"
    dtype: str = "auto"                 # "auto" | "float32" | "float16"

    # --- paged KV cache ---
    block_size: int = 16                # tokens per KV block (vLLM default)
    num_blocks: int = 512               # total KV blocks in the pool
    enable_prefix_caching: bool = True  # hash-match full blocks across requests

    # --- scheduler ---
    max_batch_tokens: int = 2048        # token budget per engine step (prefill+decode)
    max_running: int = 32               # max concurrently running sequences
    max_model_len: int = 4096           # hard cap on prompt+output length

    # --- sampling defaults ---
    max_new_tokens: int = 256
    temperature: float = 0.0            # 0 => greedy
    top_p: float = 1.0
    stop_token_ids: tuple = field(default_factory=tuple)

    def kv_pool_tokens(self) -> int:
        return self.block_size * self.num_blocks
