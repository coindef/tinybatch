"""Benchmark: continuous batching vs gang-scheduled batching, and prefix caching.

Run:  python benchmarks/bench.py [--device mps|cpu] [--n 24]

Two experiments:

1. SCHEDULING — N requests with heterogeneous prompt/output lengths, all
   submitted at t=0.
   - continuous: the engine as designed (iteration-level admission).
   - gang (static baseline): requests are grouped into fixed batches of
     ``--gang-size``; a group must fully finish before the next is admitted —
     emulating naive HF-``generate``-per-batch serving. (Sequences that hit
     EOS do exit their gang early; the head-of-line blocking between gangs is
     what static batching costs you.)

2. PREFIX CACHING — N requests sharing one long system prompt, cache on/off;
   measures prefill tokens actually computed and TTFT.

Results are written to benchmarks/results.json and printed as markdown.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path

try:
    from tinybatch import EngineConfig, LLMEngine
except ModuleNotFoundError:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinybatch import EngineConfig, LLMEngine

QUESTIONS = [
    "Explain what a KV cache is in two sentences.",
    "Why is LLM decoding memory-bandwidth-bound?",
    "Name three GPU memory tiers.",
    "What does an all-reduce do?",
    "Give one reason to quantize a model to INT8.",
    "What is speculative decoding?",
    "Why do inference engines batch requests?",
    "What is tensor parallelism?",
    "Explain TTFT vs inter-token latency.",
    "What problem does paged attention solve?",
    "How does pipeline parallelism create bubbles?",
    "What is a roofline model?",
]
SYSTEM_PREFIX = (
    "You are an expert AI infrastructure tutor. Answer concisely and precisely, "
    "using correct systems terminology. Assume the reader knows Python and basic "
    "ML but is new to GPU systems, distributed training, and inference serving. "
    "Prefer concrete numbers and named real-world systems in every answer. "
)


def make_requests(n: int, seed: int = 0) -> list[tuple[str, int]]:
    """Realistic serving mix: mostly short answers, a tail of long generations.

    High output-length variance is what makes gang scheduling hurt — short
    requests stuck in a gang behind a 256-token generation inherit its
    completion time (head-of-line blocking).
    """
    rng = random.Random(seed)
    lengths = rng.choices([16, 32, 64, 256], weights=[30, 30, 25, 15], k=n)
    return [(QUESTIONS[i % len(QUESTIONS)], lengths[i]) for i in range(n)]


def drain(engine: LLMEngine) -> None:
    while engine.scheduler.has_work():
        engine.step()


def summarize(reqs, t0, wall) -> dict:
    completion = [r.finish_time - t0 for r in reqs]
    ttft = [r.first_token_time - t0 for r in reqs]
    out_toks = sum(len(r.output_token_ids) for r in reqs)
    return {
        "wall_s": round(wall, 2),
        "output_tokens": out_toks,
        "throughput_tok_s": round(out_toks / wall, 1),
        "mean_ttft_s": round(stats.mean(ttft), 2),
        "mean_completion_s": round(stats.mean(completion), 2),
        "p99_completion_s": round(sorted(completion)[int(0.99 * (len(completion) - 1))], 2),
    }


def bench_continuous(cfg, workload) -> dict:
    engine = LLMEngine(cfg)
    t0 = time.perf_counter()
    reqs = [engine.add_request(p, max_new_tokens=m) for p, m in workload]
    for r in reqs:
        r.arrival_time = t0
    drain(engine)
    wall = time.perf_counter() - t0
    out = summarize(reqs, t0, wall)
    out["preemptions"] = sum(r.preemptions for r in reqs)
    return out


def bench_gang(cfg, workload, gang_size: int) -> dict:
    engine = LLMEngine(cfg)
    t0 = time.perf_counter()
    all_reqs = []
    for g in range(0, len(workload), gang_size):
        gang = [engine.add_request(p, max_new_tokens=m) for p, m in workload[g:g + gang_size]]
        for r in gang:
            r.arrival_time = t0
        drain(engine)                      # gang must finish before the next is admitted
        all_reqs.extend(gang)
    wall = time.perf_counter() - t0
    return summarize(all_reqs, t0, wall)


def bench_prefix(cfg_on: EngineConfig, n: int) -> dict:
    res = {}
    for label, enabled in [("prefix_cache_on", True), ("prefix_cache_off", False)]:
        cfg = EngineConfig(**{**cfg_on.__dict__, "enable_prefix_caching": enabled})
        engine = LLMEngine(cfg)
        prompts = [SYSTEM_PREFIX + q for q in QUESTIONS[: max(1, n // 2)]] * 2
        t0 = time.perf_counter()
        reqs = [engine.add_request(p, max_new_tokens=32) for p in prompts]
        drain(engine)
        wall = time.perf_counter() - t0
        prompt_toks = sum(len(r.prompt_token_ids) for r in reqs)
        cached_toks = sum(r.num_cached_tokens for r in reqs)
        res[label] = {
            **summarize(reqs, t0, wall),
            "prompt_tokens": prompt_toks,
            "prefill_tokens_computed": prompt_toks - cached_toks,
            "prefix_hit_rate": round(engine.block_manager.hit_rate(), 3),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gang-size", type=int, default=8)
    args = ap.parse_args()

    cfg = EngineConfig(device=args.device, num_blocks=768, max_batch_tokens=2048)
    workload = make_requests(args.n)

    print(f"== scheduling benchmark: {args.n} requests, gang size {args.gang_size} ==")
    gang = bench_gang(cfg, workload, args.gang_size)
    cont = bench_continuous(cfg, workload)

    print(f"== prefix caching benchmark ==")
    prefix = bench_prefix(cfg, args.n)

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    results = {
        "hardware": {"chip": chip, "platform": platform.platform()},
        "model": cfg.model_id,
        "config": {"n_requests": args.n, "gang_size": args.gang_size,
                   "block_size": cfg.block_size, "num_blocks": cfg.num_blocks},
        "gang_static": gang,
        "continuous": cont,
        "prefix": prefix,
        "speedup": {
            "throughput_x": round(cont["throughput_tok_s"] / gang["throughput_tok_s"], 2),
            "mean_completion_x": round(gang["mean_completion_s"] / cont["mean_completion_s"], 2),
            "p99_completion_x": round(gang["p99_completion_s"] / cont["p99_completion_s"], 2),
            "prefill_reduction_pct": round(100 * (1 - prefix["prefix_cache_on"]["prefill_tokens_computed"]
                                                  / prefix["prefix_cache_off"]["prefill_tokens_computed"]), 1),
            "ttft_x": round(prefix["prefix_cache_off"]["mean_ttft_s"]
                            / prefix["prefix_cache_on"]["mean_ttft_s"], 2),
        },
    }
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))

    s = results["speedup"]
    print(f"\n| metric | gang (static) | continuous | gain |")
    print(f"|---|---|---|---|")
    print(f"| throughput (tok/s) | {gang['throughput_tok_s']} | {cont['throughput_tok_s']} | {s['throughput_x']}x |")
    print(f"| mean completion (s) | {gang['mean_completion_s']} | {cont['mean_completion_s']} | {s['mean_completion_x']}x |")
    print(f"| p99 completion (s) | {gang['p99_completion_s']} | {cont['p99_completion_s']} | {s['p99_completion_x']}x |")
    print(f"\nprefix caching: {s['prefill_reduction_pct']}% fewer prefill tokens, "
          f"TTFT {s['ttft_x']}x better, hit rate {prefix['prefix_cache_on']['prefix_hit_rate']}")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
