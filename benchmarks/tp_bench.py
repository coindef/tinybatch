"""Tensor-parallel forward on real Qwen2.5-0.5B weights: verify parity with
the single-process forward and measure communication cost.

    python benchmarks/tp_bench.py [--tp 2] [--tokens 256]

Honest framing: on a single machine's CPU there is no speedup to be had —
both "ranks" share the same silicon. What this measures is the thing that
matters on a real multi-GPU node: how many bytes cross the interconnect per
layer, and what fraction of wall time communication takes when the compute
is this small. (2 all-reduces/layer x tokens x d_model x 4 bytes.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

try:
    from tinybatch.model import Qwen2Paged
    from tinybatch.tensor_parallel import run_tp_forward, tp_forward
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinybatch.model import Qwen2Paged
    from tinybatch.tensor_parallel import run_tp_forward, tp_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    m = Qwen2Paged("Qwen/Qwen2.5-0.5B-Instruct", torch.device("cpu"), torch.float32)
    ids = torch.randint(0, m.cfg.vocab_size, (args.tokens,),
                        generator=torch.Generator().manual_seed(0))

    t0 = time.perf_counter()
    ref = tp_forward(ids, m.w, m.cfg, world=1)
    single_s = time.perf_counter() - t0

    tp_logits, stats = run_tp_forward(args.tp, m.w, m.cfg, ids)
    # fp32 summation-order noise across ranks is ~1e-5 relative; anything
    # beyond that (or argmax flips) would indicate a real sharding bug
    parity = bool(torch.allclose(tp_logits, ref, atol=5e-4, rtol=1e-4))
    argmax_agree = float((tp_logits.argmax(-1) == ref.argmax(-1)).float().mean())

    results = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "tp_degree": args.tp,
        "prefill_tokens": args.tokens,
        "logits_match_single_process": parity,
        "argmax_agreement": argmax_agree,
        "max_abs_logit_diff": round(float((tp_logits - ref).abs().max()), 6),
        "single_process_forward_s": round(single_s, 3),
        "tp_rank0_wall_s": round(stats["wall_seconds"], 3),
        "allreduce_calls": stats["calls"],
        "allreduce_mb_total": round(stats["bytes"] / 2**20, 2),
        "comm_seconds": round(stats["seconds"], 3),
        "comm_fraction_of_wall": round(stats["seconds"] / stats["wall_seconds"], 3),
    }
    out = Path(__file__).parent / "results_tp.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not parity:
        sys.exit(1)


if __name__ == "__main__":
    main()
