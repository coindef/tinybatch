"""Minimal OpenAI-compatible streaming server on top of the engine.

    python -m tinybatch.server            # serves on :8321

    curl -N localhost:8321/v1/chat/completions -d '{
      "messages": [{"role": "user", "content": "Explain KV caching"}],
      "stream": true, "max_tokens": 128}'

One background thread drives ``engine.step()`` continuously; HTTP handlers
just enqueue requests and read from per-request queues — the same
decoupling vLLM's AsyncLLMEngine uses.
"""
from __future__ import annotations

import json
import queue
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .config import EngineConfig
from .engine import LLMEngine

app = FastAPI(title="tinybatch")
engine: LLMEngine | None = None
_lock = threading.Lock()
_streams: dict[int, queue.Queue] = {}


def _engine_loop() -> None:
    while True:
        with _lock:
            outputs = engine.step() if engine.scheduler.has_work() else []
        if not outputs:
            time.sleep(0.005)
            continue
        for out in outputs:
            q = _streams.get(out.req_id)
            if q is not None:
                q.put(out)


@app.post("/v1/chat/completions")
async def chat(body: dict):
    prompt = " ".join(m["content"] for m in body.get("messages", []) if m.get("role") == "user")
    q: queue.Queue = queue.Queue()
    with _lock:
        req = engine.add_request(
            prompt,
            max_new_tokens=body.get("max_tokens"),
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
        )
        _streams[req.req_id] = q

    def sse():
        try:
            while True:
                out = q.get(timeout=120)
                chunk = {"id": f"tb-{req.req_id}", "object": "chat.completion.chunk",
                         "choices": [{"delta": {"content": out.text_delta},
                                      "finish_reason": "stop" if out.finished else None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                if out.finished:
                    yield "data: [DONE]\n\n"
                    return
        finally:
            _streams.pop(req.req_id, None)

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/stats")
async def stats():
    with _lock:
        return engine.stats()


def main() -> None:
    global engine
    engine = LLMEngine(EngineConfig())
    threading.Thread(target=_engine_loop, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8321, log_level="warning")


if __name__ == "__main__":
    main()
