"""Unit tests for the paged block manager — no model required."""
import pytest

from tinybatch.block_manager import BlockManager, OutOfBlocks


def make(num_blocks=8, block_size=4, prefix=True):
    return BlockManager(num_blocks, block_size, enable_prefix_caching=prefix)


def test_prompt_allocation_block_math():
    bm = make()
    table, cached = bm.allocate_prompt(list(range(10)))   # 10 tokens, bs=4 -> 3 blocks
    assert len(table) == 3
    assert cached == 0
    assert bm.num_free() == 5


def test_free_returns_blocks():
    bm = make(prefix=False)
    table, _ = bm.allocate_prompt(list(range(10)))
    bm.free_table(table)
    assert bm.num_free() == 8


def test_out_of_blocks_raises():
    bm = make(num_blocks=2, block_size=4, prefix=False)
    with pytest.raises(OutOfBlocks):
        bm.allocate_prompt(list(range(100)))


def test_append_grows_on_boundary_only():
    # contract: append_slot is called BEFORE the new token is appended to seq
    bm = make()
    seq = list(range(4))
    table, _ = bm.allocate_prompt(seq)                     # exactly 1 full block
    assert len(table) == 1
    grew = bm.append_slot(table, seq)                      # incoming position 4 -> new block
    assert grew and len(table) == 2
    seq.append(99)
    grew = bm.append_slot(table, seq)                      # incoming position 5 -> same block
    assert not grew and len(table) == 2


def test_prefix_cache_hit_shares_blocks():
    bm = make()
    prompt = list(range(8))                                # 2 full blocks
    t1, cached1 = bm.allocate_prompt(prompt)
    assert cached1 == 0
    t2, cached2 = bm.allocate_prompt(prompt + [42, 43])    # same 8-token prefix
    assert cached2 == 8                                    # both full blocks reused
    assert t2[:2] == t1[:2]                                # physically shared
    assert bm.blocks[t1[0]].ref_count == 2


def test_prefix_cache_partial_match_stops_at_divergence():
    bm = make()
    bm.allocate_prompt([1, 2, 3, 4, 5, 6, 7, 8])           # blocks [1-4],[5-8] cached
    _, cached = bm.allocate_prompt([1, 2, 3, 4, 9, 9, 9, 9])
    assert cached == 4                                     # only the first block matches


def test_shared_block_not_freed_until_refcount_zero():
    bm = make()
    prompt = list(range(8))
    t1, _ = bm.allocate_prompt(prompt)
    t2, _ = bm.allocate_prompt(prompt)
    shared = t1[0]
    bm.free_table(t1)
    assert bm.blocks[shared].ref_count == 1                # t2 still owns it
    bm.free_table(t2)
    assert bm.blocks[shared].ref_count == 0


def test_cached_blocks_evicted_lru_when_pool_pressured():
    bm = make(num_blocks=4, block_size=4)
    t1, _ = bm.allocate_prompt(list(range(8)))             # 2 cached-able blocks
    bm.free_table(t1)                                      # rc=0 but stays cached
    assert bm.num_free() == 4                              # evictable counts as free
    t2, cached = bm.allocate_prompt(list(range(100, 116)))  # needs all 4 -> evicts both
    assert len(t2) == 4 and cached == 0
    bm.free_table(t2)
    _, cached_again = bm.allocate_prompt(list(range(8)))   # old prefix was evicted
    assert cached_again == 0                               # no stale hits after eviction


def test_no_prefix_caching_when_disabled():
    bm = make(prefix=False)
    prompt = list(range(8))
    bm.allocate_prompt(prompt)
    _, cached = bm.allocate_prompt(prompt)
    assert cached == 0


def test_decode_filled_block_published_to_cache():
    bm = make()
    seq = [1, 2, 3]
    table, _ = bm.allocate_prompt(seq)                     # partial block
    for tok in [4, 5]:                                     # fill to 4 then start block 2
        bm.append_slot(table, seq)                         # reserve slot, THEN append
        seq.append(tok)
    # block 0 ([1,2,3,4]) should now be prefix-cached
    _, cached = bm.allocate_prompt([1, 2, 3, 4, 9])
    assert cached == 4
