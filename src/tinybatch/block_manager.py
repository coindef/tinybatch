"""Paged KV-cache block manager.

This is the memory-management core of the engine, modeled on vLLM's
PagedAttention block tables (Kwon et al., SOSP 2023):

- KV memory is carved into fixed-size *blocks* of ``block_size`` tokens.
- Each sequence owns a *block table*: a list of physical block ids, one per
  logical block of its context. Logical position -> physical block is fully
  scattered, so sequences never need contiguous KV memory and fragmentation
  is bounded by ``block_size - 1`` tokens per sequence.
- Blocks are reference-counted so multiple sequences can share them.
- *Prefix caching*: a full (completely written) block is content-addressed by
  the hash of all token ids from the start of the sequence through the end of
  that block. A new request whose prompt begins with an already-cached prefix
  reuses those blocks instead of recomputing their KV (a "prefix hit").

Eviction policy for cached-but-unreferenced blocks is LRU, like vLLM v1.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


class OutOfBlocks(Exception):
    """Raised when an allocation cannot be satisfied even after eviction."""


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    # content hash for prefix caching (None until the block is full)
    prefix_hash: int | None = None
    token_ids: list[int] = field(default_factory=list)


def hash_prefix(token_ids: list[int]) -> int:
    """Content hash of a whole prefix (start of sequence .. end of block)."""
    return hash(tuple(token_ids))


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_ids: list[int] = list(range(num_blocks))
        # prefix_hash -> block_id for full, cached blocks
        self.cache: dict[int, int] = {}
        # cached blocks with ref_count == 0, in LRU order (evictable)
        self.evictable: OrderedDict[int, None] = OrderedDict()

        # stats
        self.prefix_hits = 0
        self.prefix_queries = 0

    # ------------------------------------------------------------- alloc

    def num_free(self) -> int:
        return len(self.free_ids) + len(self.evictable)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _take_free_block(self) -> Block:
        if self.free_ids:
            return self.blocks[self.free_ids.pop()]
        if self.evictable:
            # LRU-evict a cached, unreferenced block
            victim_id, _ = self.evictable.popitem(last=False)
            victim = self.blocks[victim_id]
            if victim.prefix_hash is not None:
                self.cache.pop(victim.prefix_hash, None)
            victim.prefix_hash = None
            victim.token_ids = []
            return victim
        raise OutOfBlocks(f"KV pool exhausted ({self.num_blocks} blocks)")

    def allocate_prompt(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Allocate a block table for a prompt.

        Returns ``(block_table, num_cached_tokens)`` where the first
        ``num_cached_tokens`` positions are served from the prefix cache and
        need no recomputation.
        """
        table: list[int] = []
        cached_tokens = 0

        n_full = len(token_ids) // self.block_size
        matching = self.enable_prefix_caching
        for b in range(n_full):
            self.prefix_queries += 1
            h = hash_prefix(token_ids[: (b + 1) * self.block_size])
            hit = self.cache.get(h) if matching else None
            if hit is not None:
                blk = self.blocks[hit]
                blk.ref_count += 1
                self.evictable.pop(hit, None)
                table.append(hit)
                cached_tokens += self.block_size
                self.prefix_hits += 1
            else:
                matching = False  # prefix broken; later blocks cannot match
                table.append(self._alloc_written_block(token_ids, b, h).block_id)

        # trailing partial block (never cached)
        if len(token_ids) % self.block_size:
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.token_ids = token_ids[n_full * self.block_size :]
            table.append(blk.block_id)

        return table, cached_tokens

    def _alloc_written_block(self, token_ids: list[int], block_idx: int, h: int) -> Block:
        blk = self._take_free_block()
        blk.ref_count = 1
        blk.token_ids = token_ids[block_idx * self.block_size : (block_idx + 1) * self.block_size]
        if self.enable_prefix_caching:
            blk.prefix_hash = h
            self.cache[h] = blk.block_id
        return blk

    def append_slot(self, table: list[int], seq_token_ids: list[int]) -> bool:
        """Reserve room for one more token; grow the table on block boundary.

        Returns True if a new block was allocated. When a block *fills up*
        as a result of the append, it becomes eligible for the prefix cache.
        """
        pos = len(seq_token_ids)  # index the new token will occupy
        if pos % self.block_size == 0:
            blk = self._take_free_block()
            blk.ref_count = 1
            blk.token_ids = []
            table.append(blk.block_id)
            # the previous block just became full -> publish to prefix cache
            if self.enable_prefix_caching and len(table) >= 2:
                self._publish_full_block(table, len(table) - 2, seq_token_ids)
            return True
        last = self.blocks[table[-1]]
        last.token_ids = seq_token_ids[(len(table) - 1) * self.block_size : pos]
        return False

    def _publish_full_block(self, table: list[int], block_idx: int, seq_token_ids: list[int]) -> None:
        blk = self.blocks[table[block_idx]]
        if blk.prefix_hash is not None or blk.ref_count != 1:
            return  # already cached, or shared (shared blocks are published by their first owner)
        h = hash_prefix(seq_token_ids[: (block_idx + 1) * self.block_size])
        if h in self.cache:
            return  # identical content already cached under another block
        blk.prefix_hash = h
        blk.token_ids = seq_token_ids[block_idx * self.block_size : (block_idx + 1) * self.block_size]
        self.cache[h] = blk.block_id

    # -------------------------------------------------------------- free

    def free_table(self, table: list[int]) -> None:
        for bid in table:
            blk = self.blocks[bid]
            blk.ref_count -= 1
            assert blk.ref_count >= 0, "double free"
            if blk.ref_count == 0:
                if blk.prefix_hash is not None:
                    self.evictable[bid] = None  # keep cached, evict lazily (LRU)
                else:
                    blk.token_ids = []
                    self.free_ids.append(bid)

    # ------------------------------------------------------------- stats

    def hit_rate(self) -> float:
        return self.prefix_hits / self.prefix_queries if self.prefix_queries else 0.0
