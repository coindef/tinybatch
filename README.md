# tinybatch

A miniature vLLM-style LLM inference engine — small enough to read in an afternoon, real enough to serve [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on a laptop.

Implements, from scratch, the three ideas that define modern LLM serving:

- **Paged KV cache** ([PagedAttention](https://arxiv.org/abs/2309.06180), SOSP '23) — KV memory in fixed-size blocks with per-sequence block tables, reference counting, and LRU eviction. No contiguous allocation, fragmentation bounded by `block_size − 1` tokens per sequence.
- **Continuous batching** ([Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI '22) — iteration-level scheduling: every step, finished sequences leave, waiting requests join under a token budget, and decode never waits for the slowest member of a batch. Includes vLLM-style recompute-preemption under memory pressure.
- **Prefix caching** — full blocks are content-addressed by prefix hash; requests sharing a system prompt reuse KV instead of recomputing it (measured below: ~78% of prefill eliminated on a shared-prefix workload).

The model forward pass (Qwen2 architecture: RMSNorm, GQA attention with RoPE, SwiGLU) is also implemented from scratch — because HuggingFace's `past_key_values` requires contiguous KV, which is exactly the design paged attention replaces.

## Correctness

Not vibes — tests (`pytest`, 18 passing):

- **Logits parity with HuggingFace `transformers`** at prefill *and* at every teacher-forced decode step through the paged cache (`atol=5e-3`, fp32).
- Batched generation ≡ sequential generation (greedy).
- Prefix-cache hits change performance, never outputs.
- Block-manager unit tests: refcounts, copy-on-write-free sharing, LRU eviction, boundary growth, no stale hits after eviction.
- Scheduler tests: token-budget admission, iteration-level joining, preemption under memory pressure (the tests caught a real mid-pass preemption bug during development — see `Scheduler.schedule`).

## Benchmarks

`python benchmarks/bench.py` — results on this machine in [benchmarks/results.json](benchmarks/results.json). See the numbers table there for gang-scheduled (static) vs continuous batching throughput/p99, and prefix-caching prefill reduction and TTFT gains.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[server,dev]"

pytest                              # correctness suite (downloads the 0.5B model)
python benchmarks/bench.py          # scheduling + prefix-cache benchmarks

python -m tinybatch.server          # OpenAI-compatible SSE server on :8321
curl -N localhost:8321/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is a KV cache?"}], "max_tokens":100}'
```

## Architecture

```
add_request ──> Scheduler ──────────────┐  waiting ⇄ running (preemption)
                   │ schedule()         │
                   ▼                    │
            ScheduledBatch              │ token budget, FCFS admission
            (prefill + decode)          │
                   │                    ▼
LLMEngine.step ──> Qwen2Paged.forward   BlockManager
                   │   QKV/MLP batched  │  free list · refcounts
                   │   attention reads  │  prefix hash → block id
                   ▼   through ────────>│  LRU eviction
            PagedKVCache                │
            [blocks, 2, kv_heads,       │
             block_size, head_dim]      │
```

Honest simplifications (each is a named real-world technique this project *doesn't* do): attention gathers per-sequence in Python instead of a fused paged-attention kernel; prefill is unchunked; no speculative decoding; no tensor parallelism; greedy/top-p sampling only.

## Why this exists

Built as the capstone of an AI-infrastructure self-study curriculum: the KV-cache math, block-allocator, scheduler, and attention pieces were each first built as isolated exercises, then assembled here into a system that actually serves tokens.

## License

MIT
