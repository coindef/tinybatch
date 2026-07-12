"""Tensor parallelism must be an exact refactoring of the math: the sharded
multi-process forward has to reproduce the single-process forward.

Uses a small synthetic Qwen2-shaped model with random weights — no downloads,
CI-safe — so the equality check covers the sharding arithmetic itself.
"""
from __future__ import annotations

import torch

from tinybatch.model import ModelConfig
from tinybatch.tensor_parallel import run_tp_forward, shard_weights, tp_forward

CFG = ModelConfig(
    n_layers=4, n_heads=4, n_kv_heads=2, d_model=64, head_dim=16,
    d_ff=128, vocab_size=97, rms_eps=1e-6, rope_theta=10000.0,
    tie_embeddings=False, max_position=512,
)


def synthetic_weights(cfg: ModelConfig, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def t(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    w = {"model.embed_tokens.weight": t(cfg.vocab_size, cfg.d_model),
         "model.norm.weight": torch.ones(cfg.d_model),
         "lm_head.weight": t(cfg.vocab_size, cfg.d_model)}
    qdim, kvdim = cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.ones(cfg.d_model)
        w[p + "post_attention_layernorm.weight"] = torch.ones(cfg.d_model)
        w[p + "self_attn.q_proj.weight"] = t(qdim, cfg.d_model)
        w[p + "self_attn.q_proj.bias"] = t(qdim)
        w[p + "self_attn.k_proj.weight"] = t(kvdim, cfg.d_model)
        w[p + "self_attn.k_proj.bias"] = t(kvdim)
        w[p + "self_attn.v_proj.weight"] = t(kvdim, cfg.d_model)
        w[p + "self_attn.v_proj.bias"] = t(kvdim)
        w[p + "self_attn.o_proj.weight"] = t(cfg.d_model, qdim)
        w[p + "mlp.gate_proj.weight"] = t(cfg.d_ff, cfg.d_model)
        w[p + "mlp.up_proj.weight"] = t(cfg.d_ff, cfg.d_model)
        w[p + "mlp.down_proj.weight"] = t(cfg.d_model, cfg.d_ff)
    return w


def test_shard_shapes():
    w = synthetic_weights(CFG)
    s = shard_weights(w, CFG, rank=0, world=2)
    assert s["model.layers.0.self_attn.q_proj.weight"].shape[0] == CFG.n_heads * CFG.head_dim // 2
    assert s["model.layers.0.self_attn.o_proj.weight"].shape[1] == CFG.n_heads * CFG.head_dim // 2
    assert s["model.layers.0.mlp.down_proj.weight"].shape[1] == CFG.d_ff // 2
    assert s["model.embed_tokens.weight"].shape == w["model.embed_tokens.weight"].shape


def test_tp2_matches_single_process():
    w = synthetic_weights(CFG)
    ids = torch.randint(0, CFG.vocab_size, (24,), generator=torch.Generator().manual_seed(1))
    ref = tp_forward(ids, w, CFG, world=1)                      # plain forward
    tp_logits, stats = run_tp_forward(2, w, CFG, ids, port=29531)
    torch.testing.assert_close(tp_logits, ref, atol=1e-5, rtol=1e-5)
    # exactly two all-reduces per layer, no more, no less
    assert stats["calls"] == 2 * CFG.n_layers
    assert stats["bytes"] == 2 * CFG.n_layers * 24 * CFG.d_model * 4


def test_tp_requires_divisible_heads():
    w = synthetic_weights(CFG)
    import pytest
    with pytest.raises(AssertionError):
        shard_weights(w, CFG, rank=0, world=3)                  # 4 heads / 3 ranks
