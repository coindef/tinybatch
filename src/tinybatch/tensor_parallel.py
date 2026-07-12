"""Megatron-style tensor parallelism for the Qwen2 forward pass.

The classic recipe (Shoeybi et al., Megatron-LM 2019), applied per layer:

- Attention: q/k/v projections are COLUMN-sharded (each rank owns a slice of
  the heads), the output projection is ROW-sharded, and one all-reduce(SUM)
  reassembles the residual contribution.
- MLP: gate/up are COLUMN-sharded, down is ROW-sharded, one all-reduce.
  Column-then-row ordering is the whole trick: the nonlinearity sits between
  two shardings that compose without any mid-block communication, so the
  price is exactly TWO all-reduces per transformer layer.

Embeddings, norms, and the LM head are replicated (their cost is small at
this scale; production systems shard the vocab dimension too).

This module is deliberately cache-free (single prefill forward): it exists to
demonstrate and *verify* the parallelism pattern — the test proves TP output
matches the single-process forward — and to measure communication cost. It
runs on CPU processes with the gloo backend, the same collective semantics
NCCL provides on GPUs.
"""
from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from .model import ModelConfig, _rms_norm, _rope


def shard_weights(w: dict, cfg: ModelConfig, rank: int, world: int) -> dict:
    """Slice a full Qwen2 weight dict down to this rank's tensor-parallel shard.

    F.linear weights are [out_features, in_features]:
    column-parallel => split dim 0 (and bias), row-parallel => split dim 1.
    """
    assert cfg.n_heads % world == 0, "q heads must divide tp degree"
    assert cfg.n_kv_heads % world == 0, "kv heads must divide tp degree"
    assert cfg.d_ff % world == 0, "d_ff must divide tp degree"

    def col(t: torch.Tensor) -> torch.Tensor:  # split output features
        return t.chunk(world, dim=0)[rank].contiguous()

    def row(t: torch.Tensor) -> torch.Tensor:  # split input features
        return t.chunk(world, dim=1)[rank].contiguous()

    out = {}
    for k, v in w.items():
        if any(k.endswith(s) for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                       "q_proj.bias", "k_proj.bias", "v_proj.bias",
                                       "gate_proj.weight", "up_proj.weight")):
            out[k] = col(v)
        elif any(k.endswith(s) for s in ("o_proj.weight", "down_proj.weight")):
            out[k] = row(v)
        else:
            out[k] = v  # embeddings, norms, lm_head: replicated
    return out


@torch.inference_mode()
def tp_forward(token_ids: torch.Tensor, w: dict, cfg: ModelConfig, world: int,
               comm_stats: dict | None = None) -> torch.Tensor:
    """Single prefill forward over this rank's shard. Returns full logits
    (identical on every rank after the all-reduces). world=1 degenerates to
    a plain single-process forward with zero communication.
    """
    n_heads = cfg.n_heads // world
    n_kv = cfg.n_kv_heads // world
    T = token_ids.shape[0]
    positions = torch.arange(T)

    def allreduce(x: torch.Tensor) -> torch.Tensor:
        if world > 1:
            t0 = time.perf_counter()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            if comm_stats is not None:
                comm_stats["calls"] += 1
                comm_stats["bytes"] += x.numel() * x.element_size()
                comm_stats["seconds"] += time.perf_counter() - t0
        return x

    h = F.embedding(token_ids, w["model.embed_tokens.weight"])
    for layer in range(cfg.n_layers):
        p = f"model.layers.{layer}."
        x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_eps)

        # column-parallel QKV: each rank computes only its own heads
        q = F.linear(x, w[p + "self_attn.q_proj.weight"], w.get(p + "self_attn.q_proj.bias"))
        k = F.linear(x, w[p + "self_attn.k_proj.weight"], w.get(p + "self_attn.k_proj.bias"))
        v = F.linear(x, w[p + "self_attn.v_proj.weight"], w.get(p + "self_attn.v_proj.bias"))
        q = _rope(q.view(T, n_heads, cfg.head_dim), positions, cfg.rope_theta)
        k = _rope(k.view(T, n_kv, cfg.head_dim), positions, cfg.rope_theta)
        v = v.view(T, n_kv, cfg.head_dim)

        rep = n_heads // n_kv
        attn = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.repeat_interleave(rep, dim=1).transpose(0, 1),
            v.repeat_interleave(rep, dim=1).transpose(0, 1),
            is_causal=True,
        ).transpose(0, 1).reshape(T, n_heads * cfg.head_dim)

        # row-parallel output projection -> partial sums -> all-reduce #1
        h = h + allreduce(F.linear(attn, w[p + "self_attn.o_proj.weight"]))

        x = _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
        gate = F.linear(x, w[p + "mlp.gate_proj.weight"])     # column-parallel
        up = F.linear(x, w[p + "mlp.up_proj.weight"])          # column-parallel
        # row-parallel down projection -> all-reduce #2
        h = h + allreduce(F.linear(F.silu(gate) * up, w[p + "mlp.down_proj.weight"]))

    h = _rms_norm(h, w["model.norm.weight"], cfg.rms_eps)
    return F.linear(h, w["lm_head.weight"]).float()


# --------------------------------------------------------------- runner

def _worker(rank: int, world: int, w: dict, cfg: ModelConfig,
            token_ids: torch.Tensor, port: int, queue) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world)
    try:
        shard = shard_weights(w, cfg, rank, world)
        stats = {"calls": 0, "bytes": 0, "seconds": 0.0}
        t0 = time.perf_counter()
        logits = tp_forward(token_ids, shard, cfg, world, stats)
        stats["wall_seconds"] = time.perf_counter() - t0
        if rank == 0:
            queue.put((logits, stats))
    finally:
        dist.destroy_process_group()


def run_tp_forward(world: int, w: dict, cfg: ModelConfig, token_ids: torch.Tensor,
                   port: int = 29517) -> tuple[torch.Tensor, dict]:
    """Spawn `world` CPU processes, run the sharded forward, return rank-0's
    (logits, comm_stats). Every rank's logits are identical by construction."""
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    procs = [ctx.Process(target=_worker, args=(r, world, w, cfg, token_ids, port, queue))
             for r in range(world)]
    for p in procs:
        p.start()
    result = queue.get()
    for p in procs:
        p.join()
    return result
