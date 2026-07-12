"""Smallest possible tour of the engine.

    python examples/generate.py

Submits three chat requests at once; the scheduler batches them through a
single paged KV cache, and each streams out as it finishes.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from tinybatch import EngineConfig, LLMEngine
except ModuleNotFoundError:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tinybatch import EngineConfig, LLMEngine


def main() -> None:
    engine = LLMEngine(EngineConfig(max_new_tokens=64))
    prompts = [
        "In one sentence, what does a KV cache store?",
        "Why do inference servers batch requests?",
        "Name two LLM quantization formats.",
    ]
    reqs = {engine.add_request(p).req_id: p for p in prompts}

    print(f"engine: {engine.device.type}/{str(engine.dtype).split('.')[-1]}, "
          f"KV pool {engine.stats()['kv_pool_mb']} MB\n")
    while engine.scheduler.has_work():
        for out in engine.step():
            if out.finished:
                text = engine.tokenizer.decode(out.request.output_token_ids,
                                               skip_special_tokens=True)
                print(f"Q: {reqs[out.req_id]}\nA: {text.strip()}\n")
    print("stats:", engine.stats())


if __name__ == "__main__":
    main()
