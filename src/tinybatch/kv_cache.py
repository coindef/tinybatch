"""Physical paged KV storage.

One pool per layer: ``[num_blocks, 2, n_kv_heads, block_size, head_dim]``
(the ``2`` axis is K/V). A sequence's logical token position ``t`` lives in
physical block ``table[t // block_size]`` at slot ``t % block_size`` — the
same virtual-memory-style indirection as vLLM's PagedAttention.
"""
from __future__ import annotations

import torch


class PagedKVCache:
    def __init__(self, n_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.block_size = block_size
        self.pools = [
            torch.zeros(num_blocks, 2, n_kv_heads, block_size, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]

    def memory_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.pools)

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor,
              positions: torch.Tensor, seq_slices: list[tuple[int, int]],
              block_tables: list[list[int]]) -> None:
        """Scatter each new token's K/V into its sequence's blocks.

        k, v: [n_tokens, n_kv_heads, head_dim] for the flat mixed batch.
        """
        pool = self.pools[layer]
        for (s, e), table in zip(seq_slices, block_tables):
            pos = positions[s:e]
            blocks = torch.tensor(table, device=k.device)[torch.div(pos, self.block_size, rounding_mode="floor")]
            slots = pos % self.block_size
            pool[blocks, 0, :, slots] = k[s:e]
            pool[blocks, 1, :, slots] = v[s:e]

    def gather(self, layer: int, table: list[int], context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a sequence's contiguous K/V views: [context_len, n_kv_heads, head_dim]."""
        pool = self.pools[layer]
        blocks = pool[torch.tensor(table, device=pool.device)]          # [n_blk, 2, kvh, bs, hd]
        kv = blocks.permute(1, 0, 3, 2, 4).reshape(2, -1, blocks.shape[2], blocks.shape[4])
        return kv[0, :context_len], kv[1, :context_len]
