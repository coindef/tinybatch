"""Engine correctness vs the HuggingFace reference implementation.

These are the tests that make the project real: the paged engine must
produce (numerically) the same logits and the same greedy generations as
``transformers`` running the same weights contiguously.

Run on CPU/float32 for determinism. Marked slow; they load the model twice.
"""
import pytest
import torch

from tinybatch import EngineConfig, LLMEngine

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def engine():
    return LLMEngine(EngineConfig(model_id=MODEL, device="cpu", dtype="float32",
                                  num_blocks=256, enable_prefix_caching=True))


@pytest.fixture(scope="module")
def reference():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    return tok, model


def test_prefill_logits_match_reference(engine, reference):
    tok, ref = reference
    ids = tok.apply_chat_template([{"role": "user", "content": "What is a GPU?"}],
                                  add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    with torch.inference_mode():
        ref_logits = ref(torch.tensor([ids])).logits[0, -1]

    req = engine.add_request("What is a GPU?", max_new_tokens=1)
    batch = engine.scheduler.schedule()
    assert req in batch.prefill
    import tinybatch.engine as _e
    logits = engine.model.forward(
        torch.tensor(req.prompt_token_ids, device=engine.device),
        torch.tensor(list(range(len(req.prompt_token_ids))), device=engine.device),
        [(0, len(req.prompt_token_ids))], [req.block_table],
        [len(req.prompt_token_ids)], engine.kv_cache,
    )[0]
    engine.scheduler.finish(req)

    torch.testing.assert_close(logits, ref_logits, atol=2e-3, rtol=2e-3)
    assert int(logits.argmax()) == int(ref_logits.argmax())


def test_decode_logits_match_reference_teacher_forced(engine, reference):
    """Every decode step through the paged KV cache must reproduce the
    reference logits for the same prefix (teacher-forced on HF's greedy
    tokens). Exact greedy *text* equality across implementations is a
    non-goal: ~1e-5 op-ordering differences legitimately flip near-tie
    tokens (the batch-invariance problem real engines have too).
    """
    tok, ref = reference
    ids = tok.apply_chat_template([{"role": "user", "content": "List three colors."}],
                                  add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    with torch.inference_mode():
        ref_out = ref.generate(torch.tensor([ids]), max_new_tokens=16, do_sample=False,
                               pad_token_id=tok.eos_token_id)
    full = ref_out[0].tolist()
    with torch.inference_mode():
        ref_logits_all = ref(torch.tensor([full[:-1]])).logits[0]   # [L-1, vocab]

    P = len(ids)
    bm, kv = engine.block_manager, engine.kv_cache
    table, _ = bm.allocate_prompt(full)   # prefix cache may reuse earlier test blocks
    # teacher-forced prefill of the prompt
    logits = engine.model.forward(
        torch.tensor(full[:P], device=engine.device),
        torch.tensor(list(range(P)), device=engine.device),
        [(0, P)], [table], [P], kv,
    )[0]
    torch.testing.assert_close(logits, ref_logits_all[P - 1], atol=5e-3, rtol=5e-3)

    # teacher-forced decode steps, one token at a time through the paged cache
    for k in range(len(full) - P - 1):
        pos = P + k
        step_logits = engine.model.forward(
            torch.tensor([full[pos]], device=engine.device),
            torch.tensor([pos], device=engine.device),
            [(0, 1)], [table], [pos + 1], kv,
        )[0]
        ref_step = ref_logits_all[pos]
        torch.testing.assert_close(step_logits, ref_step, atol=5e-3, rtol=5e-3)
        assert int(ref_step.argmax()) in step_logits.topk(2).indices.tolist()
    bm.free_table(table)


def test_batched_generation_matches_sequential(engine):
    """Continuous batching must not change results (greedy, fp32)."""
    prompts = ["Name a planet.", "What is 2+2?", "Say hello in French."]
    batched = engine.generate(prompts, max_new_tokens=16, temperature=0.0)
    sequential = [engine.generate([p], max_new_tokens=16, temperature=0.0)[0] for p in prompts]
    assert batched == sequential


def test_prefix_cache_does_not_change_output(engine):
    long_shared = ("You are a helpful assistant. Answer briefly and precisely. "
                   "Context: we are testing an inference engine. ")
    q = long_shared + "What is a KV cache?"
    first = engine.generate([q], max_new_tokens=16, temperature=0.0)[0]
    hits_before = engine.block_manager.prefix_hits
    second = engine.generate([q], max_new_tokens=16, temperature=0.0)[0]
    assert engine.block_manager.prefix_hits > hits_before   # cache actually used
    assert first == second                                  # ...without changing output
