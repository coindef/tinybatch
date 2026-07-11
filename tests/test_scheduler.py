"""Scheduler behavior tests using the real BlockManager, no model."""
from tinybatch.block_manager import BlockManager
from tinybatch.config import EngineConfig
from tinybatch.scheduler import Request, Scheduler, Status


def make(max_batch_tokens=64, num_blocks=16, block_size=4, max_running=8):
    cfg = EngineConfig(max_batch_tokens=max_batch_tokens, num_blocks=num_blocks,
                       block_size=block_size, max_running=max_running,
                       enable_prefix_caching=False)
    return Scheduler(cfg, BlockManager(num_blocks, block_size, False))


def req(n_prompt, max_new=32):
    return Request(prompt_token_ids=list(range(n_prompt)), max_new_tokens=max_new)


def test_admission_respects_token_budget():
    s = make(max_batch_tokens=20)
    for _ in range(3):
        s.add(req(12))
    batch = s.schedule()
    assert len(batch.prefill) == 1          # 12 fits, 24 would not
    batch2 = s.schedule()                    # decodes (1 tok) + next prefill
    assert len(batch2.decode) == 1 and len(batch2.prefill) == 1


def test_iteration_level_admission():
    """A new request joins while others are mid-decode — the Orca property."""
    s = make()
    a = req(8); s.add(a)
    s.schedule()                             # a prefills
    b = req(8); s.add(b)
    batch = s.schedule()
    assert a in batch.decode and b in batch.prefill


def test_finish_frees_blocks():
    s = make()
    a = req(8)
    s.add(a)
    s.schedule()
    free_before = s.bm.num_free()
    a.output_token_ids = [1]
    s.finish(a)
    assert s.bm.num_free() > free_before
    assert a.status is Status.FINISHED


def test_preemption_on_memory_pressure():
    # tiny pool: 4 blocks of 4 tokens
    s = make(num_blocks=4, block_size=4, max_batch_tokens=64)
    a = req(8, max_new=64); b = req(7, max_new=64)
    s.add(a); s.add(b)
    s.schedule()                             # both admitted: 2 + 2 blocks
    assert len(s.running) == 2
    # decode until the pool can't grow: b (youngest) must be preempted, not crash
    for _ in range(12):
        for r in s.schedule().decode + []:
            r.output_token_ids.append(0)
        if b.status is Status.WAITING:
            break
    assert b.status is Status.WAITING and b.preemptions >= 1
    assert a.status is Status.RUNNING        # oldest survives
