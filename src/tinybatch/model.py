"""Qwen2 forward pass implemented from scratch on a paged KV cache.

Why reimplement the model instead of using ``transformers``? Because HF's
``past_key_values`` requires *contiguous* per-sequence KV tensors — the exact
design PagedAttention exists to replace. Owning the forward pass lets
attention read K/V directly through each sequence's block table.

Architecture (Qwen2/2.5 family): RMSNorm -> GQA attention with RoPE ->
RMSNorm -> SwiGLU MLP, residuals around both; tied embeddings on 0.5B.

Simplifications, stated honestly:
- Attention loops over sequences in Python (gathering each sequence's KV
  blocks) instead of a fused paged-attention kernel. QKV/MLP projections —
  where most FLOPs live — are still batched across sequences, which is what
  makes continuous batching pay off even here.
- Prefill for each sequence is processed as one chunk (no chunked prefill).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .kv_cache import PagedKVCache


@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    head_dim: int
    d_ff: int
    vocab_size: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
    max_position: int

    @staticmethod
    def from_hf(cfg: dict) -> "ModelConfig":
        d_model = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        return ModelConfig(
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg["num_key_value_heads"],
            d_model=d_model,
            head_dim=cfg.get("head_dim", d_model // n_heads),
            d_ff=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            rms_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_embeddings=cfg.get("tie_word_embeddings", False),
            max_position=cfg["max_position_embeddings"],
        )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding. x: [n_tokens, n_heads, head_dim], positions: [n_tokens]."""
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]          # [T, hd/2]
    cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]     # [T, 1, hd/2]
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return out.to(x.dtype)


class Qwen2Paged:
    """Weights + forward pass. Not an nn.Module: inference only, no autograd."""

    def __init__(self, model_id: str, device: torch.device, dtype: torch.dtype):
        self.device, self.dtype = device, dtype
        path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
        with open(f"{path}/config.json") as f:
            self.cfg = ModelConfig.from_hf(json.load(f))

        raw: dict[str, torch.Tensor] = {}
        import glob as _glob
        for shard in sorted(_glob.glob(f"{path}/*.safetensors")):
            raw.update(load_file(shard))
        self.w = {k: v.to(device=device, dtype=dtype) for k, v in raw.items()}
        if self.cfg.tie_embeddings and "lm_head.weight" not in self.w:
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,     # [n_tokens] flat batch (mixed prefill+decode)
        positions: torch.Tensor,     # [n_tokens] position of each token in its sequence
        seq_slices: list[tuple[int, int]],   # per-sequence (start, end) into the flat batch
        block_tables: list[list[int]],       # per-sequence physical block ids
        context_lens: list[int],             # per-sequence total context length (incl. new tokens)
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        """Returns logits for the LAST token of each sequence: [n_seqs, vocab]."""
        cfg, w = self.cfg, self.w
        h = F.embedding(token_ids, w["model.embed_tokens.weight"])   # [T, d]

        for layer in range(cfg.n_layers):
            p = f"model.layers.{layer}."
            x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_eps)

            # batched QKV projection over ALL sequences' tokens at once
            q = F.linear(x, w[p + "self_attn.q_proj.weight"], w.get(p + "self_attn.q_proj.bias"))
            k = F.linear(x, w[p + "self_attn.k_proj.weight"], w.get(p + "self_attn.k_proj.bias"))
            v = F.linear(x, w[p + "self_attn.v_proj.weight"], w.get(p + "self_attn.v_proj.bias"))
            q = q.view(-1, cfg.n_heads, cfg.head_dim)
            k = k.view(-1, cfg.n_kv_heads, cfg.head_dim)
            v = v.view(-1, cfg.n_kv_heads, cfg.head_dim)
            q, k = _rope(q, positions, cfg.rope_theta), _rope(k, positions, cfg.rope_theta)

            # scatter new K/V into the paged pool through each block table
            kv_cache.write(layer, k, v, positions, seq_slices, block_tables)

            # per-sequence attention over gathered paged KV
            attn_out = torch.empty(h.shape[0], cfg.n_heads * cfg.head_dim, device=h.device, dtype=h.dtype)
            for (s, e), table, ctx_len in zip(seq_slices, block_tables, context_lens):
                keys, vals = kv_cache.gather(layer, table, ctx_len)   # [ctx, kvh, hd]
                qi = q[s:e].transpose(0, 1)                            # [h, q_len, hd]
                rep = cfg.n_heads // cfg.n_kv_heads
                ki = keys.repeat_interleave(rep, dim=1).transpose(0, 1)  # [h, ctx, hd]
                vi = vals.repeat_interleave(rep, dim=1).transpose(0, 1)
                q_len = e - s
                if q_len == ctx_len:            # full prefill: causal mask
                    out = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
                else:                            # decode (or cached-prefix prefill tail)
                    mask = torch.zeros(q_len, ctx_len, device=h.device, dtype=torch.bool)
                    first_pos = ctx_len - q_len
                    for i in range(q_len):
                        mask[i, : first_pos + i + 1] = True
                    out = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask)
                attn_out[s:e] = out.transpose(0, 1).reshape(q_len, -1)

            h = h + F.linear(attn_out, w[p + "self_attn.o_proj.weight"])

            x = _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
            gate = F.linear(x, w[p + "mlp.gate_proj.weight"])
            up = F.linear(x, w[p + "mlp.up_proj.weight"])
            h = h + F.linear(F.silu(gate) * up, w[p + "mlp.down_proj.weight"])

        h = _rms_norm(h, w["model.norm.weight"], cfg.rms_eps)
        last_tok = torch.tensor([e - 1 for _, e in seq_slices], device=h.device)
        return F.linear(h[last_tok], w["lm_head.weight"]).float()     # [n_seqs, vocab]
