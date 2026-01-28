# -*- coding: utf-8 -*-
"""
vLLM patch (loop + skip KV write) with:
- Public entrypoints:
    apply_qwen2_patch(worker)
    get_patch_stats(worker)
    reset_patch_stats(worker)

Behavior:
- PATCHED_LAYERS must be provided (comma-separated layer IDs).
- PATCHED_TAU_JSON is OPTIONAL:
    * If non-empty → use provided tau per layer.
    * If empty/missing → ADAPTIVE mode:
        - Maintain per-layer online Q1/Q3 estimates of token L2 norm (||h||₂)
        - tau(layer) = Q3 + IQR_K × (Q3 − Q1), with floor MIN_IQR for stability
        - During warmup steps, only collect stats (no looping)
        - After warmup, start looping using computed tau per layer
- Statistics are tracked independently per layer (q1/q3 + loop counters).

Notes:
- This file does NOT implement tensor-parallel (TP) group reduction (as requested).
"""

import os
import json
import time
import torch
from typing import Optional, Dict, List, Union, Any

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    flash_attn_varlen_func,
    cascade_attention,
)
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen2, qwen3, llama


# ===========================================================
# 0) Environment config helpers
# ===========================================================
def _load_layers_from_env() -> List[int]:
    """Load target layers from PATCHED_LAYERS env var."""
    layers_str = os.environ.get("PATCHED_LAYERS", "").strip()
    if not layers_str:
        raise ValueError("Env var PATCHED_LAYERS is empty. Example: '21,22,23'")
    layers = [int(x) for x in layers_str.split(",") if x.strip()]
    if not layers:
        raise ValueError("Parsed PATCHED_LAYERS is empty.")
    return layers


def _load_tau_from_env_optional() -> Optional[Dict[int, float]]:
    """Load fixed tau thresholds if provided; otherwise enable adaptive mode."""
    if os.environ.get("ADAPTIVE", "True") == "True":
        return None
    tau_json = os.environ.get("PATCHED_TAU_JSON", "").strip()
    if not tau_json:
        return None
    d = json.loads(tau_json)
    return {int(k): float(v) for k, v in d.items()}


# ===========================================================
# 1) P² (P-square) online quantile estimator (constant memory)
# ===========================================================
class P2Quantile:
    """
    P² algorithm for online quantile estimation.
    Uses constant memory; suitable for streaming Q1 (0.25) and Q3 (0.75).
    """

    def __init__(self, q: float):
        assert 0.0 < q < 1.0
        self.q = float(q)
        self._n = 0
        self._init: List[float] = []  # first 5 samples

        self.np = None  # desired marker positions
        self.n = None   # actual marker positions (int)
        self.dn = None  # desired increments
        self.x = None   # marker heights (5 values)

    def add(self, x: float) -> None:
        x = float(x)
        self._n += 1

        if self._n <= 5:
            self._init.append(x)
            if self._n == 5:
                self._init.sort()
                self.x = self._init[:]
                self.n = [1, 2, 3, 4, 5]
                q = self.q
                self.np = [1, 1 + 2 * q, 1 + 4 * q, 3 + 2 * q, 5]
                self.dn = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]
            return

        # Find insertion segment
        if x < self.x[0]:
            self.x[0] = x
            k = 0
        elif x < self.x[1]:
            k = 0
        elif x < self.x[2]:
            k = 1
        elif x < self.x[3]:
            k = 2
        elif x < self.x[4]:
            k = 3
        else:
            self.x[4] = x
            k = 3

        # Increment markers above k
        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        # Adjust interior markers (indices 1, 2, 3)
        for i in (1, 2, 3):
            d = self.np[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or (
                d <= -1 and self.n[i - 1] - self.n[i] < -1
            ):
                dsign = 1 if d > 0 else -1
                x_i = self.x[i]
                n_i = self.n[i]
                n_im1 = self.n[i - 1]
                n_ip1 = self.n[i + 1]
                x_im1 = self.x[i - 1]
                x_ip1 = self.x[i + 1]

                # Parabolic prediction
                num = (
                    dsign * (n_i - n_im1 + dsign) * (x_ip1 - x_i) / (n_ip1 - n_i)
                    + dsign * (n_ip1 - n_i - dsign) * (x_i - x_im1) / (n_i - n_im1)
                )
                x_new = x_i + num / (n_ip1 - n_im1)

                # Fallback to linear if out of bounds
                if x_im1 < x_new < x_ip1:
                    self.x[i] = x_new
                else:
                    self.x[i] = x_i + dsign * (self.x[i + dsign] - x_i) / (
                        self.n[i + dsign] - n_i
                    )

                self.n[i] += dsign

    def value(self) -> float:
        if self._n == 0:
            return float("nan")
        if self._n <= 5:
            a = sorted(self._init)
            idx = int(round((len(a) - 1) * self.q))
            idx = max(0, min(idx, len(a) - 1))
            return float(a[idx])
        return float(self.x[2])  # marker 3 tracks target quantile


# ===========================================================
# 2) FlashAttention forward patch (KV write skip)
# ===========================================================
def _make_flashattn_forward_patched():
    def _flashattn_forward_patched(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_meta FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        ctx = get_forward_context()
        dont_save = getattr(ctx, "dont_save_kv_cache", False)

        if getattr(attn_metadata, "dont_save_kv_cache", False) or dont_save:
            pass  # Skip KV cache update
        else:
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            from vllm import _custom_ops as ops
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape((num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale,
            )
            query = query.reshape((num_tokens, num_heads, head_size))

        use_local_attn = (self.use_irope and attn_metadata.local_attn_metadata is not None)

        if not attn_metadata.use_cascade or use_local_attn:
            if use_local_attn:
                local_metadata = attn_metadata.local_attn_metadata
                cu_seqlens_q = local_metadata.local_query_start_loc
                seqused_k = local_metadata.local_seqused_k
                max_seqlen_q = local_metadata.local_max_query_len
                max_seqlen_k = local_metadata.local_max_seq_len
                block_table = local_metadata.local_block_table
                scheduler_metadata = local_metadata.local_scheduler_metadata
            else:
                cu_seqlens_q = attn_metadata.query_start_loc
                seqused_k = attn_metadata.seq_lens
                max_seqlen_q = attn_metadata.max_query_len
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table
                scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape),
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

    return _flashattn_forward_patched


def _patch_flashattn_once_on_worker(worker):
    """Idempotent patching of FlashAttention forward."""
    if getattr(worker, "_loop_flashattn_patched", False):
        return
    FlashAttentionImpl.forward = _make_flashattn_forward_patched()
    worker._loop_flashattn_patched = True


# ===========================================================
# 3) Stats initialization and helpers
# ===========================================================
def _init_loop_stats_on_worker(worker, layers: List[int], device=None) -> None:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._loop_stats = {
        "prefill": torch.zeros((), dtype=torch.int64, device=device),
        "decode": torch.zeros((), dtype=torch.int64, device=device),
        "bad": torch.zeros((), dtype=torch.int64, device=device),
        "loop_counters": {lid: torch.zeros((), dtype=torch.int64, device=device) for lid in layers},
        "norm_history": {lid: [] for lid in layers},  # optional debug
    }

    worker._adaptive = {
        "enabled": False,
        "warmup_steps": 0,
        "step": 0,
        "min_iqr": 1e-3,
        "iqr_k": 1.5,
        "tau_floor": 0.0,
        "q25": {lid: P2Quantile(0.25) for lid in layers},
        "q75": {lid: P2Quantile(0.75) for lid in layers},
        "count": {lid: 0 for lid in layers},
        "tau_map": {lid: None for lid in layers},
    }


def _get_attn_metadata():
    return get_forward_context().attn_metadata


def _is_decode(attn_metadata) -> bool:
    """Detect decode (autoregressive) mode."""
    if hasattr(attn_metadata, "max_query_len") and attn_metadata.max_query_len is not None:
        return int(attn_metadata.max_query_len) == 1
    if hasattr(attn_metadata, "query_start_loc") and attn_metadata.query_start_loc is not None:
        qsl = attn_metadata.query_start_loc
        if qsl.numel() >= 2:
            max_q = int((qsl[1:] - qsl[:-1]).max().item())
            return max_q == 1
    return False


def _update_adaptive_tau(worker, layer_idx: int, norm_vec: torch.Tensor) -> None:
    """Update per-layer quantile estimators using mean(norm_vec) as sample."""
    ad = worker._adaptive
    if not ad["enabled"]:
        return

    val = float(norm_vec.mean().item())
    ad["q25"][layer_idx].add(val)
    ad["q75"][layer_idx].add(val)
    ad["count"][layer_idx] += 1

    q1 = ad["q25"][layer_idx].value()
    q3 = ad["q75"][layer_idx].value()
    iqr = max(float(ad["min_iqr"]), float(q3 - q1))
    tau = max(float(ad["tau_floor"]), float(q3 + float(ad["iqr_k"]) * iqr))
    ad["tau_map"][layer_idx] = tau


def _get_tau(worker, layer_idx: int) -> Optional[float]:
    """Get tau for layer: prefer fixed, fall back to adaptive."""
    cfg = worker._loop_cfg
    tau_map = cfg.get("tau", {})
    if layer_idx in tau_map and tau_map[layer_idx] is not None:
        return float(tau_map[layer_idx])

    ad = getattr(worker, "_adaptive", None)
    if ad and ad.get("enabled", False):
        return ad["tau_map"].get(layer_idx, None)
    return None


# ===========================================================
# 4) Public: apply patch
# ===========================================================
def apply_qwen2_patch(worker):
    """
    Apply loop + KV-write-skip patch to all workers via collective_rpc.

    Environment variables:
      - PATCHED_LAYERS: required (e.g., "21,22,23")
      - PATCHED_TAU_JSON: optional JSON dict of layer→tau
      - ADAPTIVE_WARMUP_STEPS (default: 500)
      - ADAPTIVE_IQR_K (default: 1.5)
      - ADAPTIVE_MIN_IQR (default: 1e-3)
      - ADAPTIVE_TAU_FLOOR (default: 0.0)
    """
    layers = _load_layers_from_env()
    layer_tau = _load_tau_from_env_optional()
    adaptive_enabled = layer_tau is None

    # Fixed hyperparameters
    ALPHA = 1.0
    GROWTH_EPS = 2e-1
    BLOWUP_RATIO = 1.5
    EPS = 1e-6
    ONLY_DECODE = True
    MAX_EXTRA_ITERS = 1

    # Adaptive parameters
    warmup_steps = int(os.environ.get("ADAPTIVE_WARMUP_STEPS", "500"))
    iqr_k = float(os.environ.get("ADAPTIVE_IQR_K", "1.5"))
    min_iqr = float(os.environ.get("ADAPTIVE_MIN_IQR", "1e-3"))
    tau_floor = float(os.environ.get("ADAPTIVE_TAU_FLOOR", "0.0"))

    _patch_flashattn_once_on_worker(worker)

    worker._loop_cfg = {
        "layers": layers,
        "tau": layer_tau if layer_tau is not None else {},
        "only_decode": ONLY_DECODE,
        "alpha": ALPHA,
        "growth_eps": GROWTH_EPS,
        "blowup_ratio": BLOWUP_RATIO,
        "eps": EPS,
        "max_extra_iters": MAX_EXTRA_ITERS,
    }
    _init_loop_stats_on_worker(worker, layers)

    worker._adaptive.update({
        "enabled": adaptive_enabled,
        "warmup_steps": warmup_steps,
        "min_iqr": min_iqr,
        "iqr_k": iqr_k,
        "tau_floor": tau_floor,
    })

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        cfg = worker._loop_cfg
        stats = worker._loop_stats
        ad = worker._adaptive

        # Pipeline parallelism prelude
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Decode detection
        attn_metadata = None
        decode_mode = True
        if cfg["only_decode"]:
            attn_metadata = _get_attn_metadata()
            decode_mode = _is_decode(attn_metadata)
            if decode_mode:
                stats["decode"].add_(1)
            else:
                stats["prefill"].add_(1)
        else:
            stats["prefill"].add_(1)

        # Global step counter for adaptive warmup
        if ad["enabled"] and decode_mode:
            ad["step"] += 1

        class _KVWriteGuard:
            __slots__ = ("ctx", "old")
            def __init__(self, dont_save: bool):
                self.ctx = get_forward_context()
                self.old = getattr(self.ctx, "dont_save_kv_cache", False)
                self.ctx.dont_save_kv_cache = bool(dont_save)
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb):
                self.ctx.dont_save_kv_cache = self.old
                return False

        layers_list = cfg["layers"]
        tau_blow = float(cfg["blowup_ratio"])
        growth_eps = float(cfg["growth_eps"])
        alpha = float(cfg["alpha"])
        eps = float(cfg["eps"])
        max_extra = int(cfg["max_extra_iters"])

        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer], start=self.start_layer):
            if (not decode_mode) or (layer_idx not in layers_list):
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue

            if attn_metadata is None:
                attn_metadata = _get_attn_metadata()

            try:
                h, r = hidden_states, residual

                # Pass 0: KV write enabled
                with _KVWriteGuard(dont_save=False):
                    h1, r1 = layer(positions, h, r)

                real1 = h1 if r1 is None else (h1 + r1)
                norm1 = real1.norm(dim=-1)

                # Update adaptive stats
                if ad["enabled"]:
                    _update_adaptive_tau(worker, layer_idx, norm1)

                # Skip looping during warmup
                if ad["enabled"] and (ad["step"] < ad["warmup_steps"]):
                    hidden_states, residual = h1, r1
                    continue

                tau = _get_tau(worker, layer_idx)
                if tau is None or tau <= 0.0:
                    hidden_states, residual = h1, r1
                    continue
                tau_f = float(tau)

                active = (norm1 > tau_f) & (norm1 < tau_f * tau_blow)
                prev_norm_vec = norm1.detach()
                h, r = h1, r1

                if not bool(active.any().item()):
                    hidden_states, residual = h, r
                else:
                    extra_it = 0
                    while bool(active.any().item()) and (extra_it < max_extra):
                        with _KVWriteGuard(dont_save=True):
                            h_new, r_new = layer(positions, h, r)

                        real = h_new if r_new is None else (h_new + r_new)
                        norm_vec = real.norm(dim=-1)

                        above_tau = norm_vec > tau_f
                        growth = (norm_vec - prev_norm_vec) / (prev_norm_vec + eps)
                        low_growth = growth < growth_eps
                        above_2tau = norm_vec >= (tau_blow * tau_f)

                        stop_mask = low_growth | above_2tau
                        new_active = above_tau & (~stop_mask)
                        active = active & new_active

                        active_f = active.to(h.dtype).unsqueeze(-1)
                        h = h * (1 - active_f) + ((1.0 - alpha) * h + alpha * h_new) * active_f

                        if r is None:
                            r = r_new
                        else:
                            r = r * (1 - active_f) + ((1.0 - alpha) * r + alpha * r_new) * active_f

                        prev_norm_vec = norm_vec.detach()
                        extra_it += 1

                    hidden_states, residual = h, r
                    stats["loop_counters"][layer_idx].add_(int(extra_it))

            finally:
                ctx = get_forward_context()
                ctx.dont_save_kv_cache = False

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Patch supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward
    llama.LlamaModel.forward = patched_forward

    return {
        "pid": os.getpid(),
        "patched": True,
        "layers": layers,
        "adaptive_enabled": adaptive_enabled,
        "warmup_steps": warmup_steps if adaptive_enabled else None,
        "iqr_k": iqr_k if adaptive_enabled else None,
        "stats_device": str(worker._loop_stats["prefill"].device),
    }


# ===========================================================
# 5) Public: get/reset stats
# ===========================================================
def get_patch_stats(worker):
    s = worker._loop_stats
    cfg = getattr(worker, "_loop_cfg", {})
    ad = getattr(worker, "_adaptive", {})

    tau_out = {}
    for lid in cfg.get("layers", []):
        if lid in cfg.get("tau", {}):
            tau_out[lid] = float(cfg["tau"][lid])
        elif ad.get("enabled", False):
            tau_out[lid] = ad["tau_map"].get(lid, None)

    q_snapshot = {}
    if ad.get("enabled", False):
        for lid in cfg.get("layers", []):
            q_snapshot[lid] = {
                "count": int(ad["count"].get(lid, 0)),
                "q25": float(ad["q25"][lid].value()) if lid in ad["q25"] else None,
                "q75": float(ad["q75"][lid].value()) if lid in ad["q75"] else None,
                "tau": tau_out.get(lid, None),
            }

    return {
        "pid": os.getpid(),
        "cfg": {
            "layers": cfg.get("layers"),
            "only_decode": cfg.get("only_decode"),
            "max_extra_iters": cfg.get("max_extra_iters"),
            "growth_eps": cfg.get("growth_eps"),
            "blowup_ratio": cfg.get("blowup_ratio"),
            "adaptive_enabled": bool(ad.get("enabled", False)),
            "adaptive_step": int(ad.get("step", 0)),
            "adaptive_warmup_steps": int(ad.get("warmup_steps", 0)),
            "adaptive_iqr_k": float(ad.get("iqr_k", 0.0)),
            "adaptive_min_iqr": float(ad.get("min_iqr", 0.0)),
            "adaptive_tau_floor": float(ad.get("tau_floor", 0.0)),
        },
        "prefill": int(s["prefill"].item()),
        "decode": int(s["decode"].item()),
        "bad": int(s["bad"].item()),
        "loop_counters": {int(k): int(v.item()) for k, v in s["loop_counters"].items()},
        "adaptive_quantiles": q_snapshot,
    }


def reset_patch_stats(worker):
    s = worker._loop_stats
    s["prefill"].zero_()
    s["decode"].zero_()
    s["bad"].zero_()
    for v in s["loop_counters"].values():
        v.zero_()
    for k in s["norm_history"]:
        s["norm_history"][k].clear()

    ad = getattr(worker, "_adaptive", None)
    if ad is not None:
        layers = list(ad.get("q25", {}).keys())
        ad["step"] = 0
        ad["q25"] = {lid: P2Quantile(0.25) for lid in layers}
        ad["q75"] = {lid: P2Quantile(0.75) for lid in layers}
        ad["count"] = {lid: 0 for lid in layers}
        ad["tau_map"] = {lid: None for lid in layers}

    return "reset_ok"