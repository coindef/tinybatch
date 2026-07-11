"""Iteration-level (continuous-batching) scheduler.

Each engine step assembles a fresh batch from whatever is runnable *right
now* — the Orca/vLLM insight that removes head-of-line blocking:

- Running sequences each contribute 1 decode token.
- Waiting requests are admitted while the per-step token budget
  (``max_batch_tokens``) and the KV pool allow; admission triggers prefill of
  their un-cached prompt tokens in the same step.
- If a decode step cannot allocate a KV slot, the *youngest* running sequence
  is preempted: its blocks are freed and it re-enters the wait queue for full
  recomputation (vLLM's recompute-preemption policy — cheap for short
  contexts, and prefix caching often makes the recompute nearly free).

A ``static`` mode is included purely as the benchmark baseline: it fills a
batch, then runs it until every member finishes.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from .block_manager import BlockManager, OutOfBlocks
from .config import EngineConfig

_req_counter = itertools.count()


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    prompt_token_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    req_id: int = field(default_factory=lambda: next(_req_counter))
    arrival_time: float = field(default_factory=time.perf_counter)

    # mutable state
    status: Status = Status.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def context_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


@dataclass
class ScheduledBatch:
    """What the model should run this step."""
    prefill: list[Request]
    decode: list[Request]

    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.cfg = config
        self.bm = block_manager
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------ step

    def schedule(self) -> ScheduledBatch:
        decode: list[Request] = []
        prefill: list[Request] = []
        budget = self.cfg.max_batch_tokens

        # 1) decodes first: every running sequence needs 1 slot + 1 budget
        for req in list(self.running):
            if req.status is not Status.RUNNING:
                continue  # preempted earlier in this same pass
            try:
                self.bm.append_slot(req.block_table, req.all_token_ids())
            except OutOfBlocks:
                victim = self._preempt_youngest()
                if victim is req:
                    continue
                try:
                    self.bm.append_slot(req.block_table, req.all_token_ids())
                except OutOfBlocks:
                    self._preempt(req)
                    continue
            decode.append(req)
            budget -= 1
        # a later request's OutOfBlocks may have preempted one already in `decode`
        decode = [r for r in decode if r.status is Status.RUNNING]

        # 2) admit waiting requests FCFS while budget + memory allow
        while self.waiting and len(self.running) < self.cfg.max_running:
            req = self.waiting[0]
            new_tokens = len(req.prompt_token_ids)  # upper bound; prefix cache may shrink it
            if new_tokens > budget:
                break
            if self.bm.blocks_needed(new_tokens) > self.bm.num_free():
                break
            self.waiting.pop(0)
            req.block_table, req.num_cached_tokens = self.bm.allocate_prompt(req.prompt_token_ids)
            # a fully-cached prompt still must compute its last token's logits
            if req.num_cached_tokens >= len(req.prompt_token_ids):
                req.num_cached_tokens = len(req.prompt_token_ids) - 1
            req.status = Status.RUNNING
            self.running.append(req)
            prefill.append(req)
            budget -= new_tokens - req.num_cached_tokens

        return ScheduledBatch(prefill=prefill, decode=decode)

    # ------------------------------------------------------- lifecycle

    def finish(self, req: Request) -> None:
        req.status = Status.FINISHED
        req.finish_time = time.perf_counter()
        self.running.remove(req)
        self.bm.free_table(req.block_table)
        req.block_table = []

    def _preempt(self, req: Request) -> None:
        req.preemptions += 1
        req.status = Status.WAITING
        self.running.remove(req)
        self.bm.free_table(req.block_table)
        req.block_table = []
        req.output_token_ids = []          # recompute policy: restart generation
        req.num_cached_tokens = 0
        self.waiting.insert(0, req)

    def _preempt_youngest(self) -> Request:
        victim = self.running[-1]
        self._preempt(victim)
        return victim
