# Study Guide — own every line of this project

If this repo is on your resume, you must be able to defend it under interview probing. This guide is the reading order, the why behind each design decision, and the questions an NVIDIA/Meta interviewer would actually ask. Budget ~2 evenings.

## Reading order (with the question each file answers)

1. **`src/tinybatch/block_manager.py`** — *why doesn't KV memory fragment?*
   Understand: blocks, block tables, refcounts, the prefix hash (why hash the *whole prefix*, not the block's own tokens — because attention makes a block's KV depend on everything before it), LRU eviction of unreferenced cached blocks.
2. **`src/tinybatch/kv_cache.py`** — *where do the bytes actually live?*
   One tensor per layer `[num_blocks, 2, kv_heads, block_size, head_dim]`. Do the math: with fp16 and Qwen2.5-0.5B (24 layers, 2 KV heads, head_dim 64), one block of 16 tokens = 24·2·2·16·64·2 bytes ≈ 197 KB across layers.
3. **`src/tinybatch/scheduler.py`** — *why doesn't a short request wait behind a long one?*
   Understand: iteration-level scheduling, the token budget (why prefill and decode share it), FCFS admission, and recompute-preemption (why the *youngest* is the victim; why preempting resets output_token_ids).
4. **`src/tinybatch/model.py`** — *what does a transformer forward actually do?*
   Understand: why QKV/MLP projections are batched across all sequences but attention is per-sequence; what RoPE does and why positions must be explicit; where GQA saves memory; why prefill uses `is_causal=True` but decode builds an explicit mask.
5. **`src/tinybatch/engine.py`** — *how does one step tie it together?*
   The flat ragged batch (`token_ids` + `seq_slices`), sampling, EOS handling, streaming.
6. **`tests/test_correctness.py`** — *how do you PROVE an engine is correct?*
   Teacher-forced stepwise logits parity — and why exact greedy text equality across implementations is the wrong test (near-tie tokens flip on ~1e-5 op-ordering noise; this is the batch-invariance problem production engines have too).
7. **`benchmarks/bench.py`** — *what do the techniques buy, measured how?*

## Interview questions you must now be able to answer

**Memory / PagedAttention**
- Why does contiguous KV allocation waste 60–80% of memory in naive serving? (Reserved-but-unused tail + external fragmentation; the PagedAttention paper's motivating measurement.)
- What's the KV-cache size formula? (`2 · layers · kv_heads · head_dim · dtype_bytes` per token — derive it, then compute it for Llama-3-8B: 128 KB/token.)
- Why block_size 16 and not 1 or 1024? (1 → block-table overhead and scattered reads; 1024 → internal fragmentation of `block_size − 1` per sequence amortizes badly.)
- How does prefix caching interact with refcounts? When can a cached block be evicted?

**Scheduling / continuous batching**
- What is head-of-line blocking in gang scheduling, numerically? (In the benchmark: a 16-token request gang-batched with a 256-token one inherits its ~4-min completion — check `benchmarks/results.json` for actuals on your machine.)
- Why schedule decodes before admissions? (Running sequences hold memory; starving them of slots deadlocks the pool.)
- Preemption: why recompute instead of swap-to-CPU? When would swapping win? (Long contexts where recompute costs more than PCIe transfer; vLLM supports both.)
- What limits batch size — compute or memory? (Decode is memory-bandwidth-bound: bigger batches amortize weight reads until KV reads dominate.)

**Model / numerics**
- Why do prefill and decode have different bottlenecks? (Prefill: compute-bound matmuls over the whole prompt. Decode: one token per step — weight/KV memory bandwidth.)
- Why fp32 for the correctness tests but fp16 for benchmarks?
- Why can two correct engines emit different greedy text? (Near-tie argmax + op-ordering ≈1e-5 noise.)

**Systems**
- Walk through one `engine.step()` end to end.
- How would you add chunked prefill? Speculative decoding? Where would tensor parallelism split this model?
- What would a fused paged-attention kernel do that the Python gather loop doesn't? (One kernel reading via block tables — no materialized contiguous copy, no per-sequence launch overhead.)

## Honest talking points (know the limits)

- Throughput/latency numbers are on Apple-Silicon CPU/MPS with a 0.5B model — the *phenomena* (HOL blocking, prefill reuse) are real and measured, but absolute numbers don't transfer to H100s. Say this proactively; it reads as maturity.
- The attention loop is Python-per-sequence; a production engine fuses it. You know exactly where the kernel boundary is — that's the point of the exercise.
- If asked "did you write this alone?" — answer honestly about your process. What you must be able to do is explain and extend every component: that's what these two evenings are for. Extending it yourself (pick one: chunked prefill, swap-based preemption, a Triton kernel for the gather) converts this from a studied project into unambiguously your engineering.

## Suggested extensions (each is a resume-strengthening PR to your own repo)

1. Chunked prefill (split long prompts across steps; bounds TTFT jitter for decodes).
2. Per-step metrics endpoint + a latency histogram (ties to your Prometheus/Grafana experience).
3. Swap-based preemption with a CPU block pool, benchmarked against recompute.
4. A `torch.compile` pass over the forward; measure and explain the speedup.
