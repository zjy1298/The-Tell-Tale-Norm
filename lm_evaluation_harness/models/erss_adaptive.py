# -*- coding: utf-8 -*-
"""
Self-contained vLLM patch (loop/injection + optional KV write skip)
Public entrypoints:
  - apply_qwen2_patch(worker)
  - get_patch_stats(worker)
  - reset_patch_stats(worker)

Configuration:
  - PATCHED_LAYERS="21,22,23,24,25,26"   (REQUIRED)
  - PATCHED_TAU_JSON='{"21":1300.0,"22":...}'  (OPTIONAL)
    * If PATCHED_TAU_JSON is empty/missing => ADAPTIVE mode enabled:
      - Per-layer independent online Q1/Q3 estimates of token L2 norm
      - tau(layer) = Q3 + ADAPTIVE_IQR_K * (Q3 - Q1),
        with IQR >= ADAPTIVE_MIN_IQR, tau >= ADAPTIVE_TAU_FLOOR
      - During first ADAPTIVE_WARMUP_STEPS decode steps: only collect stats (no injection)

Environment variables (adaptive mode):
  - ADAPTIVE_WARMUP_STEPS=200
  - ADAPTIVE_IQR_K=1.5
  - ADAPTIVE_MIN_IQR=1e-3
  - ADAPTIVE_TAU_FLOOR=0.0

No module globals for config; everything stored on worker.
"""

import os
import json
import torch
from typing import Optional, Dict, List, Union

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
# Environment configuration helpers
# ===========================================================
def _load_layers_from_env() -> List[int]:
    """Load target layers from PATCHED_LAYERS environment variable."""
    layers_str = os.environ.get("PATCHED_LAYERS", "").strip()
    if not layers_str:
        raise ValueError("Env var PATCHED_LAYERS is empty. Example: '21,22,23,24,25,26'")
    layers = [int(x) for x in layers_str.split(",") if x.strip()]
    if not layers:
        raise ValueError("Parsed PATCHED_LAYERS is empty.")
    return layers


def _load_tau_from_env_optional() -> Optional[Dict[int, float]]:
    """Load fixed tau thresholds if provided; otherwise enable adaptive mode."""
    tau_json = os.environ.get("PATCHED_TAU_JSON", "").strip()
    if not tau_json:
        return None
    d = json.loads(tau_json)
    return {int(k): float(v) for k, v in d.items()}


def _load_adaptive_cfg_from_env() -> Dict[str, float]:
    """Load adaptive configuration parameters from environment variables."""
    return {
        "warmup_steps": int(os.environ.get("ADAPTIVE_WARMUP_STEPS", "200")),
        "iqr_k": float(os.environ.get("ADAPTIVE_IQR_K", "1.5")),
        "min_iqr": float(os.environ.get("ADAPTIVE_MIN_IQR", "1e-3")),
        "tau_floor": float(os.environ.get("ADAPTIVE_TAU_FLOOR", "0.0")),
    }


# ===========================================================
# Adaptive quantile (P²) for IQR tau
# ===========================================================
class P2Quantile:
    """P² algorithm for online quantile estimation (constant memory)."""

    def __init__(self, q: float):
        assert 0.0 < q < 1.0
        self.q = float(q)
        self._n = 0
        self._init: List[float] = []
        self.np = None
        self.n = None
        self.dn = None
        self.x = None

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

        for i in range(k + 1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

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

                num = (
                    dsign * (n_i - n_im1 + dsign) * (x_ip1 - x_i) / (n_ip1 - n_i)
                    + dsign * (n_ip1 - n_i - dsign) * (x_i - x_im1) / (n_i - n_im1)
                )
                x_new = x_i + num / (n_ip1 - n_im1)

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
        return float(self.x[2])


def _adaptive_tau_from_iqr(q1: float, q3: float, iqr_k: float, min_iqr: float, tau_floor: float) -> float:
    """Compute tau threshold using IQR rule: tau = Q3 + IQR_K * IQR."""
    iqr = max(float(min_iqr), float(q3 - q1))
    return max(float(tau_floor), float(q3 + float(iqr_k) * iqr))


# ===========================================================
# FlashAttention forward patch (KV write skip)
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
                layer._q_scale)
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


def _patch_flashattn_once(worker):
    """Idempotent patching of FlashAttention forward method."""
    if getattr(worker, "_patched_flashattn_loopinj", False):
        return
    FlashAttentionImpl.forward = _make_flashattn_forward_patched()
    worker._patched_flashattn_loopinj = True


# ===========================================================
# Worker state initialization
# ===========================================================
def _init_worker_state(worker, layers: List[int], adaptive_enabled: bool, adaptive_cfg: Dict, device=None) -> None:
    """Initialize worker state for loop/injection patch."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._loop_stats = {
        "prefill": torch.zeros((), dtype=torch.int64, device=device),
        "decode": torch.zeros((), dtype=torch.int64, device=device),
        "bad": torch.zeros((), dtype=torch.int64, device=device),
        "norm_history": {lid: [] for lid in layers},
        "inject_total": torch.zeros((), dtype=torch.int64, device=device),
        "inject_by_layer": {lid: torch.zeros((), dtype=torch.int64, device=device) for lid in layers},
        "low_streak_triggered": torch.zeros((), dtype=torch.int64, device=device),
        "low_streak_max": torch.zeros((), dtype=torch.int64, device=device),
        "active_seqs": torch.zeros((), dtype=torch.int64, device=device),
        "inject_debug": [],
        "inject_length": torch.zeros((), dtype=torch.int64, device=device),
        "adaptive_step": torch.zeros((), dtype=torch.int64, device=device),
    }
    worker._inj_state = {}  # key=(layer_idx, seq_slot) -> dict

    # Adaptive state (Python objects)
    worker._adaptive = {
        "enabled": bool(adaptive_enabled),
        "warmup_steps": int(adaptive_cfg["warmup_steps"]),
        "iqr_k": float(adaptive_cfg["iqr_k"]),
        "min_iqr": float(adaptive_cfg["min_iqr"]),
        "tau_floor": float(adaptive_cfg["tau_floor"]),
        "q25": {lid: P2Quantile(0.25) for lid in layers},
        "q75": {lid: P2Quantile(0.75) for lid in layers},
        "count": {lid: 0 for lid in layers},
        "tau_map": {lid: None for lid in layers},  # lid->float
    }


def _get_attn_metadata():
    """Retrieve attention metadata from forward context."""
    return get_forward_context().attn_metadata


def _is_decode(attn_metadata) -> bool:
    """Detect if current forward pass is in decode (autoregressive) mode."""
    if hasattr(attn_metadata, "max_query_len"):
        return int(attn_metadata.max_query_len) == 1
    if hasattr(attn_metadata, "query_start_loc"):
        qsl = attn_metadata.query_start_loc
        if qsl is not None and qsl.numel() >= 2:
            max_q = int((qsl[1:] - qsl[:-1]).max().item())
            return max_q == 1
    return False


def _adaptive_update_layer_tau(worker, layer_idx: int, norm_vec: torch.Tensor) -> None:
    """Update per-layer quantiles with one scalar sample from this step."""
    ad = worker._adaptive
    if not ad["enabled"]:
        return
    val = float(norm_vec.mean().item())  # one sample per step
    ad["q25"][layer_idx].add(val)
    ad["q75"][layer_idx].add(val)
    ad["count"][layer_idx] += 1
    q1 = ad["q25"][layer_idx].value()
    q3 = ad["q75"][layer_idx].value()
    ad["tau_map"][layer_idx] = _adaptive_tau_from_iqr(
        q1=q1, q3=q3,
        iqr_k=ad["iqr_k"],
        min_iqr=ad["min_iqr"],
        tau_floor=ad["tau_floor"],
    )


# ===========================================================
# Public API 1/3: Apply patch
# ===========================================================
def apply_qwen2_patch(worker):
    """
    Apply loop/injection patch to Qwen2/Qwen3/Llama models on each vLLM worker.
    
    Adaptive mode:
      - Enabled automatically if no fixed tau is provided
      - During first ADAPTIVE_WARMUP_STEPS decode steps: only collect stats
      - Tau per layer: Q3 + ADAPTIVE_IQR_K * IQR, with safeguards
    """
    layers = _load_layers_from_env()
    layer_tau = _load_tau_from_env_optional()
    adaptive_cfg = _load_adaptive_cfg_from_env()
    adaptive_enabled = (layer_tau is None) or (os.environ.get("ADAPTIVE", "True") == "True")

    # Fixed hyperparameters
    ONLY_DECODE = True
    LOW_K = 40
    GAMMA = 1.0
    P_INJECT = 0.2
    BETA = 0.5
    COOLDOWN = 32
    BANK_K = 32
    COS_MIN = 0.5
    DEBUG_KEEP = 64

    _patch_flashattn_once(worker)
    _init_worker_state(worker, layers, adaptive_enabled, adaptive_cfg)

    # Store configuration on worker for stats reporting
    worker._loop_cfg = {
        "PATCHED_LAYERS": layers,
        "PATCHED_TAU_JSON_provided": layer_tau,
        "ONLY_DECODE": ONLY_DECODE,
        "adaptive": {
            "enabled": adaptive_enabled,
            **adaptive_cfg,
        },
        "inj": {
            "LOW_K": LOW_K, "GAMMA": GAMMA, "P_INJECT": P_INJECT, "BETA": BETA,
            "COOLDOWN": COOLDOWN, "BANK_K": BANK_K, "COS_MIN": COS_MIN,
            "DEBUG_KEEP": DEBUG_KEEP,
        }
    }

    from vllm.forward_context import get_forward_context

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

    @torch.no_grad()
    def _get_slot_state(layer_idx: int, seq_slot: int, hidden_dim: int, device, dtype):
        key = (layer_idx, seq_slot)
        st = worker._inj_state.get(key, None)
        if st is None or st.get("hidden_dim", None) != hidden_dim:
            st = {
                "hidden_dim": hidden_dim,
                "bank_vecs": torch.empty((0, hidden_dim), device=device, dtype=dtype),
                "bank_norms": torch.empty((0,), device=device, dtype=torch.float32),
                "low_streak": 0,
                "cooldown": 0,
            }
            worker._inj_state[key] = st
        return st

    @torch.no_grad()
    def _bank_update(st, vec: torch.Tensor, vec_norm: float):
        bank_vecs = st["bank_vecs"]
        bank_norms = st["bank_norms"]
        if bank_vecs.numel() == 0:
            bank_vecs = vec.unsqueeze(0)
            bank_norms = torch.tensor([vec_norm], device=vec.device, dtype=torch.float32)
        else:
            bank_vecs = torch.cat([bank_vecs, vec.unsqueeze(0)], dim=0)
            bank_norms = torch.cat([bank_norms, torch.tensor([vec_norm], device=vec.device, dtype=torch.float32)], dim=0)

        if bank_norms.numel() > BANK_K:
            topk = torch.topk(bank_norms, k=BANK_K, largest=True)
            idx = topk.indices
            bank_vecs = bank_vecs.index_select(0, idx)
            bank_norms = topk.values

        st["bank_vecs"] = bank_vecs
        st["bank_norms"] = bank_norms

    @torch.no_grad()
    def _bank_best_by_cos(st, x: torch.Tensor):
        bank_vecs = st["bank_vecs"]
        if bank_vecs.numel() == 0:
            return None, None
        x_hat = x / (x.norm() + 1e-6)
        v_hat = bank_vecs / (bank_vecs.norm(dim=-1, keepdim=True) + 1e-6)
        cos = (v_hat @ x_hat)  # [m]
        best_cos, best_idx = torch.max(cos, dim=0)
        best_cos_val = float(best_cos.item())
        if best_cos_val < COS_MIN:
            return None, best_cos_val
        return bank_vecs[best_idx], best_cos_val

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

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
        if ONLY_DECODE:
            attn_metadata = _get_attn_metadata()
            decode_mode = _is_decode(attn_metadata)
            if decode_mode:
                stats["decode"].add_(1)
                if ad["enabled"]:
                    stats["adaptive_step"].add_(1)
            else:
                stats["prefill"].add_(1)
        else:
            stats["prefill"].add_(1)

        in_warmup = bool(ad["enabled"] and decode_mode and int(stats["adaptive_step"].item()) < int(ad["warmup_steps"]))

        # Process layers
        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer], start=self.start_layer):
            if (not decode_mode) or (layer_idx not in layers):
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue

            # Determine tau: fixed from env or adaptive
            tau = None
            if layer_tau is not None:
                tau = layer_tau.get(layer_idx, None)
            if tau is None and ad["enabled"]:
                tau = ad["tau_map"].get(layer_idx, None)  # may still be None early

            try:
                # Pass 0: always run once with KV write enabled
                with _KVWriteGuard(dont_save=False):
                    h1, r1 = layer(positions, hidden_states, residual)

                real1 = h1 if r1 is None else (h1 + r1)
                norm1 = real1.norm(dim=-1)

                # Update adaptive tau on every decode step
                if ad["enabled"]:
                    _adaptive_update_layer_tau(worker, layer_idx, norm1)
                    tau = ad["tau_map"].get(layer_idx, None)

                if tau is None:
                    hidden_states, residual = h1, r1
                    continue

                tau_f = float(tau)
                low_th = float(GAMMA) * tau_f

                if decode_mode:
                    stats["active_seqs"].add_(int(norm1.numel()))

                # Warmup: only collect stats; do not inject/modify
                if in_warmup:
                    hidden_states, residual = h1, r1
                    continue

                rand = torch.rand((norm1.numel(),), device=norm1.device)
                h_out, r_out = h1, r1
                touched = False

                for i in range(norm1.numel()):
                    seq_slot = i
                    st = _get_slot_state(layer_idx, seq_slot, real1.shape[-1], real1.device, real1.dtype)

                    if st["cooldown"] > 0:
                        st["cooldown"] -= 1

                    nval = float(norm1[i].item())

                    if st["low_streak"] > int(stats["low_streak_max"].item()):
                        stats["low_streak_max"].fill_(st["low_streak"])

                    # Update bank with high-norm vectors
                    if nval > tau_f:
                        _bank_update(st, real1[i].detach(), nval)

                    # Track low-norm streak
                    if nval < low_th:
                        st["low_streak"] += 1
                    else:
                        st["low_streak"] = 0

                    # Trigger injection on long low-norm streak
                    if st["low_streak"] >= LOW_K:
                        stats["low_streak_triggered"].add_(1)

                        if st["cooldown"] == 0 and float(rand[i].item()) < float(P_INJECT):
                            v_best, best_cos = _bank_best_by_cos(st, real1[i].detach())
                            if v_best is not None:
                                if not touched:
                                    h_out = h_out.clone()
                                    if r_out is not None:
                                        r_out = r_out.clone()
                                    touched = True

                                beta = float(BETA)
                                if r_out is None:
                                    h_out[i] = h_out[i] + beta * v_best
                                else:
                                    target_h = v_best - r_out[i]
                                    h_out[i] = h_out[i] + beta * target_h

                                stats["inject_total"].add_(1)
                                stats["inject_by_layer"][layer_idx].add_(1)

                                seq_len_val = None
                                if attn_metadata is not None and hasattr(attn_metadata, "seq_lens"):
                                    sl = attn_metadata.seq_lens
                                    if sl is not None and i < sl.numel():
                                        seq_len_val = int(sl[i].item())

                                if len(stats["inject_debug"]) < DEBUG_KEEP:
                                    stats["inject_debug"].append({
                                        "layer": int(layer_idx),
                                        "seq_slot": int(seq_slot),
                                        "seq_len": seq_len_val,
                                        "norm": float(nval),
                                        "tau": float(tau_f),
                                        "low_th": float(low_th),
                                        "cos": float(best_cos) if best_cos is not None else None,
                                        "chosen_norm": float(v_best.norm().item()),
                                    })
                                    stats["inject_length"] += 1

                                st["low_streak"] = 0
                                st["cooldown"] = COOLDOWN

                hidden_states, residual = h_out, r_out

            finally:
                ctx = get_forward_context()
                ctx.dont_save_kv_cache = False

        # Pipeline parallelism epilogue
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Apply to supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward
    llama.LlamaModel.forward = patched_forward

    return {
        "pid": os.getpid(),
        "patched": True,
        "patched_model_cls": "qwen2/qwen3",
        "layers": layers,
        "adaptive_enabled": adaptive_enabled,
        "adaptive_cfg": adaptive_cfg if adaptive_enabled else None,
        "stats_device": str(worker._loop_stats["prefill"].device),
    }


# ===========================================================
# Public API 2/3: Get stats
# ===========================================================
def get_patch_stats(worker):
    """Return JSON-serializable statistics from worker."""
    s = worker._loop_stats
    cfg = getattr(worker, "_loop_cfg", {})
    ad = getattr(worker, "_adaptive", None)

    out = {
        "pid": os.getpid(),
        "prefill": int(s["prefill"].item()),
        "decode": int(s["decode"].item()),
        "bad": int(s["bad"].item()),
        "norm_history": s["norm_history"],
        "inject_total": int(s["inject_total"].item()),
        "inject_by_layer": {int(k): int(v.item()) for k, v in s["inject_by_layer"].items()},
        "low_streak_triggered": int(s["low_streak_triggered"].item()),
        "low_streak_max": int(s["low_streak_max"].item()),
        "active_seqs": int(s["active_seqs"].item()),
        "inject_debug": s["inject_debug"],
        "inject_length": int(s["inject_length"].item()),
        "adaptive_step": int(s["adaptive_step"].item()),
        "config": cfg,
    }

    if ad is not None and ad.get("enabled", False):
        q_snapshot = {}
        for lid in cfg.get("PATCHED_LAYERS", []):
            q1 = ad["q25"][lid].value()
            q3 = ad["q75"][lid].value()
            tau = ad["tau_map"].get(lid, None)
            q_snapshot[str(lid)] = {
                "count": int(ad["count"].get(lid, 0)),
                "q25": float(q1) if q1 == q1 else None,
                "q75": float(q3) if q3 == q3 else None,
                "tau": float(tau) if tau is not None else None,
            }
        out["adaptive_quantiles"] = q_snapshot
        out["adaptive_cfg"] = {
            "warmup_steps": int(ad["warmup_steps"]),
            "iqr_k": float(ad["iqr_k"]),
            "min_iqr": float(ad["min_iqr"]),
            "tau_floor": float(ad["tau_floor"]),
        }

    return out


# ===========================================================
# Public API 3/3: Reset stats
# ===========================================================
def reset_patch_stats(worker):
    """Reset all counters and adaptive estimators."""
    s = worker._loop_stats
    s["prefill"].zero_()
    s["decode"].zero_()
    s["bad"].zero_()
    for k in list(s["norm_history"].keys()):
        s["norm_history"][k].clear()

    s["inject_total"].zero_()
    for v in s["inject_by_layer"].values():
        v.zero_()
    s["low_streak_triggered"].zero_()
    s["low_streak_max"].zero_()
    s["active_seqs"].zero_()
    s["inject_debug"].clear()
    s["inject_length"].zero_()
    s["adaptive_step"].zero_()

    worker._inj_state.clear()

    # Reset adaptive estimators
    ad = getattr(worker, "_adaptive", None)
    cfg = getattr(worker, "_loop_cfg", {})
    if ad is not None and ad.get("enabled", False):
        layers = cfg.get("PATCHED_LAYERS", [])
        worker._adaptive = {
            "enabled": True,
            "warmup_steps": int(ad["warmup_steps"]),
            "iqr_k": float(ad["iqr_k"]),
            "min_iqr": float(ad["min_iqr"]),
            "tau_floor": float(ad["tau_floor"]),
            "q25": {lid: P2Quantile(0.25) for lid in layers},
            "q75": {lid: P2Quantile(0.75) for lid in layers},
            "count": {lid: 0 for lid in layers},
            "tau_map": {lid: None for lid in layers},
        }

    return "reset_ok"