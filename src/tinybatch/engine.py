"""The engine: admit -> schedule -> batched forward -> sample -> stream.

``LLMEngine.step()`` runs ONE iteration of continuous batching:
every running sequence decodes one token, and newly admitted requests
prefill their (un-cached) prompt in the same flat batch. The model sees a
single ragged batch of tokens described by ``seq_slices``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer

from .block_manager import BlockManager
from .config import EngineConfig
from .kv_cache import PagedKVCache
from .model import Qwen2Paged
from .scheduler import Request, Scheduler, Status


@dataclass
class StepOutput:
    req_id: int
    token_id: int
    text_delta: str
    finished: bool
    request: Request = field(repr=False)


class LLMEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        if self.cfg.device == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        if self.cfg.dtype == "auto":
            self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        else:
            self.dtype = getattr(torch, self.cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.model = Qwen2Paged(self.cfg.model_id, self.device, self.dtype)
        mc = self.model.cfg
        self.kv_cache = PagedKVCache(
            mc.n_layers, self.cfg.num_blocks, self.cfg.block_size,
            mc.n_kv_heads, mc.head_dim, self.device, self.dtype,
        )
        self.block_manager = BlockManager(
            self.cfg.num_blocks, self.cfg.block_size, self.cfg.enable_prefix_caching
        )
        self.scheduler = Scheduler(self.cfg, self.block_manager)
        self.eos_ids = set(self.cfg.stop_token_ids) | {self.tokenizer.eos_token_id}
        # Qwen chat models terminate turns with <|im_end|>
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            self.eos_ids.add(im_end)

    # ------------------------------------------------------------ API

    def add_request(self, prompt: str, *, chat: bool = True, max_new_tokens: int | None = None,
                    temperature: float | None = None, top_p: float | None = None) -> Request:
        if chat:
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
            if hasattr(ids, "keys"):          # transformers >= 5 returns BatchEncoding
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = self.tokenizer.encode(prompt)
        ids = ids[: self.cfg.max_model_len - 1]
        req = Request(
            prompt_token_ids=list(ids),
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens,
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=self.cfg.top_p if top_p is None else top_p,
        )
        self.scheduler.add(req)
        return req

    def generate(self, prompts: list[str], **kw) -> list[str]:
        """Blocking batch API: submit all, run to completion, return texts."""
        reqs = [self.add_request(p, **kw) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        return [self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True) for r in reqs]

    # ----------------------------------------------------------- step

    def step(self) -> list[StepOutput]:
        batch = self.scheduler.schedule()
        if batch.is_empty():
            return []

        token_ids: list[int] = []
        positions: list[int] = []
        seq_slices: list[tuple[int, int]] = []
        block_tables: list[list[int]] = []
        context_lens: list[int] = []
        ordered = batch.prefill + batch.decode

        for req in batch.prefill:
            start = len(token_ids)
            new = req.prompt_token_ids[req.num_cached_tokens:]
            token_ids.extend(new)
            positions.extend(range(req.num_cached_tokens, len(req.prompt_token_ids)))
            seq_slices.append((start, len(token_ids)))
            block_tables.append(req.block_table)
            context_lens.append(len(req.prompt_token_ids))
        for req in batch.decode:
            start = len(token_ids)
            token_ids.append(req.all_token_ids()[-1])
            positions.append(req.context_len() - 1)
            seq_slices.append((start, len(token_ids)))
            block_tables.append(req.block_table)
            context_lens.append(req.context_len())

        logits = self.model.forward(
            torch.tensor(token_ids, device=self.device),
            torch.tensor(positions, device=self.device),
            seq_slices, block_tables, context_lens, self.kv_cache,
        )

        outputs: list[StepOutput] = []
        next_ids = self._sample(logits, ordered)
        for req, tok in zip(ordered, next_ids):
            if req.first_token_time is None:
                req.first_token_time = time.perf_counter()
            req.output_token_ids.append(tok)
            finished = (
                tok in self.eos_ids
                or len(req.output_token_ids) >= req.max_new_tokens
                or req.context_len() >= self.cfg.max_model_len
            )
            delta = self.tokenizer.decode([tok], skip_special_tokens=True)
            if finished:
                self.scheduler.finish(req)
            outputs.append(StepOutput(req.req_id, tok, delta, finished, req))
        return outputs

    def _sample(self, logits: torch.Tensor, reqs: list[Request]) -> list[int]:
        out: list[int] = []
        for i, req in enumerate(reqs):
            row = logits[i]
            if req.temperature <= 0.0:
                out.append(int(row.argmax()))
                continue
            probs = torch.softmax(row / req.temperature, dim=-1)
            if req.top_p < 1.0:
                sorted_p, idx = probs.sort(descending=True)
                keep = sorted_p.cumsum(0) - sorted_p < req.top_p
                probs = torch.zeros_like(probs).scatter(0, idx[keep], sorted_p[keep])
                probs = probs / probs.sum()
            out.append(int(torch.multinomial(probs, 1)))
        return out

    # ---------------------------------------------------------- stats

    def stats(self) -> dict:
        bm = self.block_manager
        return {
            "kv_pool_mb": round(self.kv_cache.memory_bytes() / 2**20, 1),
            "free_blocks": bm.num_free(),
            "prefix_hit_rate": round(bm.hit_rate(), 3),
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
        }
