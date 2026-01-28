# -*- coding: utf-8 -*-
"""
vLLM patch for online ABCD metrics computation per request and layer.
Public entrypoints:
  - apply_qwen2_patch(worker)
  - get_patch_stats(worker)
  - reset_patch_stats(worker)

Configuration:
  - PATCHED_LAYERS="21,22,23,24,25,26"   (REQUIRED)
  - PATCHED_TAU_JSON='{"21":1300.0,"22":...}'  (REQUIRED)

Metrics per layer l for each request ID:
A_sum_norm   = sum over decode steps of mean token norm (step, l)
B_event_area = sum over steps max(0, norm - tau_l)
C_event_count= sum over steps 1(norm > tau_l)
D_late_area  = sum over last LATE_FRAC steps of step event area

Note: Only records during DECODE steps (max_query_len == 1). Prefill is ignored.
"""

import os
import json
import torch
from typing import Optional, Dict, List, Union

import numpy as np
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen2, qwen3, llama
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore


# ===========================================================
# Environment configuration helpers
# ===========================================================
def _load_layers_and_tau_from_env() -> tuple[list[int], dict[int, float]]:
    """Load target layers and tau thresholds from environment variables."""
    layers_str = os.environ.get("PATCHED_LAYERS", "").strip()
    tau_json = os.environ.get("PATCHED_TAU_JSON", "").strip()
    if not layers_str:
        raise ValueError("Env var PATCHED_LAYERS is empty. Example: '21,22,23,24,25,26'")
    if not tau_json:
        raise ValueError("Env var PATCHED_TAU_JSON is empty. Example: '{\"21\":1300,...}'")

    layers = [int(x) for x in layers_str.split(",") if x.strip()]
    d = json.loads(tau_json)
    tau = {int(k): float(v) for k, v in d.items()}
    return layers, tau


# ===========================================================
# Worker statistics initialization
# ===========================================================
def _init_stats(worker, layers: List[int]):
    """Initialize statistics storage on worker."""
    worker._norm_stats = {
        "prefill": 0,
        "decode": 0,
        "per_req": {},  # req_id(str) -> dict
        "meta": {"seen_req_ids": 0, "seen_req_indices": 0, "mapping_errors": 0},
        # Keep config snapshot on worker for get_patch_stats
        "cfg": {"layers": layers},
    }


# ===========================================================
# Runner patch: stash request mapping into attention metadata
# ===========================================================
def _apply_runner_patch(worker):
    """Patch GPUModelRunner to expose request ID mapping."""
    # Idempotent: only patch once per worker
    if getattr(worker, "_runner_patched_normstat", False):
        return {"pid": os.getpid(), "patched_runner": True, "already": True}

    orig = GPUModelRunner._prepare_inputs

    def _prepare_inputs_patched(self, scheduler_output):
        attn_metadata, logits_indices, spec_decode_metadata = orig(self, scheduler_output)

        req_ids = list(self.input_batch.req_ids)
        num_reqs = int(self.input_batch.num_reqs)
        total_tokens = int(getattr(attn_metadata, "num_actual_tokens", 0))

        tokens = [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        req_indices = np.repeat(np.arange(num_reqs, dtype=np.int32), num_scheduled_tokens)

        sm = getattr(attn_metadata, "scheduler_metadata", None)
        if sm is not None:
            sm._normstat_req_ids = req_ids
            sm._normstat_req_indices = req_indices
            sm._normstat_num_reqs = num_reqs
            sm._normstat_total_tokens = total_tokens
        else:
            attn_metadata._normstat_req_ids = req_ids
            attn_metadata._normstat_req_indices = req_indices
            attn_metadata._normstat_num_reqs = num_reqs
            attn_metadata._normstat_total_tokens = total_tokens

        return attn_metadata, logits_indices, spec_decode_metadata

    GPUModelRunner._prepare_inputs = _prepare_inputs_patched
    worker._runner_patched_normstat = True
    return {"pid": os.getpid(), "patched_runner": True, "already": False}


# ===========================================================
# Model forward patch: compute ABCD metrics online
# ===========================================================
def _apply_forward_patch(worker, layers: List[int], tau: Dict[int, float]):
    """Patch model forward to compute ABCD metrics per request and layer."""
    _init_stats(worker, layers)

    ONLY_DECODE = True
    LATE_FRAC = 0.30

    def _is_decode(attn_metadata) -> bool:
        """Detect if current forward pass is in decode (autoregressive) mode."""
        if attn_metadata is None:
            return False
        if hasattr(attn_metadata, "max_query_len"):
            return int(attn_metadata.max_query_len) == 1
        if hasattr(attn_metadata, "query_start_loc"):
            qsl = attn_metadata.query_start_loc
            if qsl is not None and qsl.numel() >= 2:
                max_q = int((qsl[1:] - qsl[:-1]).max().item())
                return max_q == 1
        return False

    def _ensure_req(req_id: str):
        """Ensure per-request storage exists."""
        pr = worker._norm_stats["per_req"]
        if req_id not in pr:
            pr[req_id] = {
                "layers": {
                    lid: {"A_sum": 0.0, "B_area": 0.0, "C_count": 0, "B_steps": []}
                    for lid in layers
                },
                "decode_steps": 0,
            }
        return pr[req_id]

    @torch.no_grad()
    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        stats = worker._norm_stats

        # Pipeline parallelism prelude
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Decode detection and request mapping retrieval
        ctx = get_forward_context()
        attn_metadata = getattr(ctx, "attn_metadata", None)
        decode_mode = _is_decode(attn_metadata)

        if ONLY_DECODE:
            if decode_mode:
                stats["decode"] += 1
            else:
                stats["prefill"] += 1
        else:
            stats["prefill"] += 1

        sm = getattr(attn_metadata, "scheduler_metadata", None) if attn_metadata is not None else None
        if sm is not None:
            req_ids = getattr(sm, "_normstat_req_ids", None)
            req_indices = getattr(sm, "_normstat_req_indices", None)
            if req_indices is not None:
                stats["meta"]["seen_req_indices"] += 1
        else:
            req_ids = getattr(attn_metadata, "_normstat_req_ids", None) if attn_metadata is not None else None
            req_indices = getattr(attn_metadata, "_normstat_req_indices", None) if attn_metadata is not None else None
            if req_indices is not None:
                stats["meta"]["seen_req_indices"] += 1

        if req_ids is not None:
            stats["meta"]["seen_req_ids"] += 1
        else:
            stats["meta"]["mapping_errors"] += 1

        # Process layers
        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer], start=self.start_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Skip if not decode or not a target layer
            if (not decode_mode) or (layer_idx not in layers):
                continue

            tau_l = tau.get(layer_idx, None)
            if tau_l is None:
                continue
            tau_l = float(tau_l)

            real = hidden_states if residual is None else (hidden_states + residual)
            norm_vec = real.norm(dim=-1)  # [num_tokens]
            num_tokens = int(norm_vec.numel())

            # Map tokens to request IDs
            if req_ids is None:
                # Fallback: use slot-based IDs (less reliable)
                for j in range(num_tokens):
                    rid = f"slot_{j}"
                    rec = _ensure_req(rid)
                    n = float(norm_vec[j].item())
                    st = rec["layers"][layer_idx]
                    st["A_sum"] += n
                    b = max(0.0, n - tau_l)
                    st["B_area"] += b
                    st["C_count"] += int(n > tau_l)
                    st["B_steps"].append(b)
                continue

            # Prefer req_indices if shape matches
            if req_indices is not None and len(req_indices) == num_tokens:
                for j in range(num_tokens):
                    req_idx = int(req_indices[j])
                    if req_idx < 0 or req_idx >= len(req_ids):
                        continue
                    rid = str(req_ids[req_idx])
                    rec = _ensure_req(rid)
                    n = float(norm_vec[j].item())
                    st = rec["layers"][layer_idx]
                    st["A_sum"] += n
                    b = max(0.0, n - tau_l)
                    st["B_area"] += b
                    st["C_count"] += int(n > tau_l)
                    st["B_steps"].append(b)
            else:
                # Common decode case: 1 token per request
                if num_tokens == len(req_ids):
                    for j in range(num_tokens):
                        rid = str(req_ids[j])
                        rec = _ensure_req(rid)
                        n = float(norm_vec[j].item())
                        st = rec["layers"][layer_idx]
                        st["A_sum"] += n
                        b = max(0.0, n - tau_l)
                        st["B_area"] += b
                        st["C_count"] += int(n > tau_l)
                        st["B_steps"].append(b)
                else:
                    stats["meta"]["mapping_errors"] += 1

        # Pipeline parallelism epilogue
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Apply to supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward
    llama.LlamaModel.forward = patched_forward

    return {"pid": os.getpid(), "patched_forward": True, "late_frac": LATE_FRAC, "only_decode": ONLY_DECODE}


# ===========================================================
# Public API: apply/get/reset
# ===========================================================
def apply_qwen2_patch(worker):
    """
    Main entrypoint used by lm-eval wrapper:
      - Patches runner and model forward
      - Reads PATCHED_LAYERS/PATCHED_TAU_JSON from environment
    """
    layers, tau = _load_layers_and_tau_from_env()
    print("layers:", layers)
    print("tau:", tau)

    r = _apply_runner_patch(worker)
    f = _apply_forward_patch(worker, layers, tau)
    # Store config snapshot for get_patch_stats
    worker._norm_stats["cfg"].update({
        "PATCHED_LAYERS": layers,
        "PATCHED_TAU": {str(k): float(v) for k, v in tau.items()},
        "ONLY_DECODE": True,
        "LATE_FRAC": 0.30,
    })
    return {"pid": os.getpid(), "patched": True, "runner": r, "forward": f}


def get_patch_stats(worker):
    """
    Returns JSON-serializable statistics and finalizes D_late metric.
    """
    LATE_FRAC = float(worker._norm_stats.get("cfg", {}).get("LATE_FRAC", 0.30))

    out = worker._norm_stats
    per_req = out["per_req"]

    # Finalize D_late per layer per request
    for rid, rec in per_req.items():
        for lid, st in rec["layers"].items():
            b_steps = st.get("B_steps", [])
            if not b_steps:
                st["D_late"] = 0.0
            else:
                start = max(0, int((1.0 - LATE_FRAC) * len(b_steps)))
                st["D_late"] = float(sum(b_steps[start:]))

            # Clear B_steps to save memory
            st["B_steps"] = []

    return out


def reset_patch_stats(worker):
    """Reset all statistics counters."""
    layers = worker._norm_stats.get("cfg", {}).get("PATCHED_LAYERS", None)
    if layers is None:
        # Best-effort reset
        worker._norm_stats = {}
        return "reset_ok"
    _init_stats(worker, layers) 
    return "reset_ok"