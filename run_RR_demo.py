# -*- coding: utf-8 -*-
"""
Self-contained vLLM patch:
1) Patch FlashAttentionImpl.forward to optionally skip KV-cache write
2) Patch the real model class' forward on each worker
3) Keep loop counters on worker instance (NO module globals)
4) All counters are torch tensors; RPC returns Python ints + norm history lists
"""

import os
import torch
import json
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
import torch
from transformers import AutoTokenizer
from vllm.v1.worker.gpu_model_runner import *
from vllm.v1.attention.backends.flash_attn import *
from vllm.entrypoints.openai.api_server import *

# =========================
# Predefined thresholds
# =========================
LOOP_LAYER_IDS: List[int] = [25]

LAYER_TAU: Dict[int, float] = {
    25: 300.0
}

ALPHA = 1.0  # Damping factor: 1.0 means no damping.
GROWTH_EPS = 1e-2          # Relative growth threshold for stopping refinement
PATIENCE_GROWTH = 1        # Not used in current logic (kept for compatibility)
BLOWUP_RATIO = 7           # Stop if norm exceeds BLOWUP_RATIO * tau
EPS = 1e-6                 # Small constant for numerical stability
ONLY_DECODE = True         # Only apply looping during decode phase

# ===========================================================
# 1) Patch flash-attn forward: add dont_save_kv_cache flag
# ===========================================================
def _flashattn_forward_patched(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with optional KV cache write skip."""
    assert output is not None, "Output tensor must be provided."

    if attn_metadata is None:
        return output  # Profiling run

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

    assert not use_local_attn, "Cascade attention does not support local attention."
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


# Apply patch at import time
FlashAttentionImpl.forward = _flashattn_forward_patched


# ===========================================================
# 2) Worker-side stats (all counters are tensors; history is list)
# ===========================================================
def _init_loop_stats_on_worker(worker, device=None) -> None:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._loop_stats = {
        "prefill": torch.zeros((), dtype=torch.int64, device=device),
        "decode":  torch.zeros((), dtype=torch.int64, device=device),
        "bad":     torch.zeros((), dtype=torch.int64, device=device),
        "loop_counters": {lid: torch.zeros((), dtype=torch.int64, device=device)
                          for lid in LOOP_LAYER_IDS},
        "norm_history": {lid: [] for lid in LOOP_LAYER_IDS},  # list of [step, mean_norm]
    }


def _get_attn_metadata():
    return get_forward_context().attn_metadata


def _is_decode(attn_metadata) -> bool:
    """Heuristic to detect decode phase (single-token generation)."""
    if hasattr(attn_metadata, "max_query_len"):
        return int(attn_metadata.max_query_len) == 1
    if hasattr(attn_metadata, "query_start_loc"):
        qsl = attn_metadata.query_start_loc
        if qsl is not None and qsl.numel() >= 2:
            max_q = int((qsl[1:] - qsl[:-1]).max().item())
            return max_q == 1
    return False


# ===========================================================
# 3) Patch model forward on each worker (no globals)
# ===========================================================
def apply_forward_patch(worker):
    """
    Patch model forward on each vLLM worker.
    Adds adaptive layer refinement during decode for specified layers.
    """
    _init_loop_stats_on_worker(worker)

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        global ALPHA, GROWTH_EPS, PATIENCE_GROWTH, BLOWUP_RATIO, EPS, ONLY_DECODE, LOOP_LAYER_IDS, LAYER_TAU
        stats = worker._loop_stats

        # Original pre-processing
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        attn_metadata = None
        decode_mode = True

        if ONLY_DECODE:
            attn_metadata = _get_attn_metadata()
            decode_mode = _is_decode(attn_metadata)
            if decode_mode:
                stats["decode"].add_(1)
            else:
                stats["prefill"].add_(1)
        else:
            stats["prefill"].add_(1)

        class _KVWriteGuard:
            """Context manager to temporarily disable KV cache writes."""
            __slots__ = ("ctx", "old")

            def __init__(self, dont_save: bool):
                self.ctx = get_forward_context()
                self.old = getattr(self.ctx, "dont_save_kv_cache", False)
                self.ctx.dont_save_kv_cache = bool(dont_save)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.ctx.dont_save_kv_cache = self.old
                return False

        # Layer-by-layer processing
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer:self.end_layer],
            start=self.start_layer
        ):
            if (not decode_mode) or (layer_idx not in LOOP_LAYER_IDS):
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue

            tau = LAYER_TAU.get(layer_idx, None)
            if tau is None:
                hidden_states, residual = layer(positions, hidden_states, residual)
                continue

            if attn_metadata is None:
                attn_metadata = _get_attn_metadata()

            try:
                MAX_EXTRA_ITERS = 10
                h = hidden_states
                r = residual

                # Pass 0: always run once with KV write enabled
                with _KVWriteGuard(dont_save=False):
                    h1, r1 = layer(positions, h, r)

                real1 = h1 if r1 is None else (h1 + r1)
                norm1 = real1.norm(dim=-1)
                stats["norm_history"][layer_idx].append([0, float(norm1.mean().item())])

                tau_f = float(tau)
                active = (norm1 > tau_f) & (norm1 < tau_f * BLOWUP_RATIO)
                prev_norm_vec = norm1.detach()
                h, r = h1, r1

                if not bool(active.any().item()):
                    hidden_states, residual = h, r
                else:
                    extra_it = 0
                    while bool(active.any().item()) and (extra_it < MAX_EXTRA_ITERS):
                        with _KVWriteGuard(dont_save=True):
                            h_new, r_new = layer(positions, h, r)

                        real = h_new if r_new is None else (h_new + r_new)
                        norm_vec = real.norm(dim=-1)
                        stats["norm_history"][layer_idx].append([extra_it + 1, float(norm_vec.mean().item())])

                        above_tau = norm_vec > tau_f
                        growth = (norm_vec - prev_norm_vec) / (prev_norm_vec + EPS)
                        low_growth = growth < float(GROWTH_EPS)
                        above_2tau = norm_vec >= (BLOWUP_RATIO * tau_f)

                        stop_mask = low_growth | above_2tau
                        new_active = above_tau & (~stop_mask)
                        active = active & new_active

                        active_f = active.to(h.dtype).unsqueeze(-1)
                        h = h * (1 - active_f) + ((1.0 - ALPHA) * h + ALPHA * h_new) * active_f

                        if r is None:
                            r = r_new
                        else:
                            r = r * (1 - active_f) + ((1.0 - ALPHA) * r + ALPHA * r_new) * active_f

                        prev_norm_vec = norm_vec.detach()
                        extra_it += 1

                    hidden_states, residual = h, r
                    stats["loop_counters"][layer_idx].add_(int(extra_it))

            finally:
                ctx = get_forward_context()
                ctx.dont_save_kv_cache = False

        # Pipeline parallelism handling
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Apply patch to supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward
    llama.LlamaModel.forward = patched_forward

    return {
        "pid": os.getpid(),
        "patched_model_cls": "qwen3",
        "stats_device": str(worker._loop_stats["prefill"].device),
    }


# ===========================================================
# 4) RPC helpers: get/reset stats (return python-serializable)
# ===========================================================
def get_counters(worker):
    s = worker._loop_stats
    return {
        "prefill": int(s["prefill"].item()),
        "decode": int(s["decode"].item()),
        "bad": int(s["bad"].item()),
        "loop_counters": {k: int(v.item()) for k, v in s["loop_counters"].items()},
        "norm_history": s["norm_history"],
    }


def reset_counters(worker):
    s = worker._loop_stats
    s["prefill"].zero_()
    s["decode"].zero_()
    s["bad"].zero_()
    for v in s["loop_counters"].values():
        v.zero_()
    for k in s["norm_history"].keys():
        s["norm_history"][k].clear()
    return "reset_ok"


# ===========================================================
# 5) Example test (driver-side)
# ===========================================================
def test_loop():
    from vllm import LLM, SamplingParams

    MODEL_PATH = "your/model/path"  # Anonymized model path

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,
        compilation_config=0,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )

    patch_results = llm.llm_engine.collective_rpc(apply_forward_patch)
    print("patch_results:", patch_results)

    llm.llm_engine.collective_rpc(reset_counters)

    questions = [
        "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.",
        "Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^{2}-xy-6y^{2}=0$.",
        "Four unit squares form a $2 times 2$ grid. Each of the 12 unit line segments forming the sides of the squares is colored either red or blue in such a way that each unit square has 2 red sides and 2 blue sides. Find the number of such colorings."
    ]

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.05, max_tokens=4096
    )

    jsonl_file = "your/output/file.jsonl"  # Anonymized output path

    def save_and_clear_norm_history(worker_idx=None):
        stats = llm.llm_engine.collective_rpc(get_counters)
        worker_stats = stats[worker_idx] if isinstance(stats, list) and worker_idx is not None else (stats[0] if isinstance(stats, list) else stats)
        norm_hist = worker_stats["norm_history"]

        entry = {
            "question": questions[0],
            "norm_history": {
                layer_id: [(step, float(norm)) for step, norm in hist]
                for layer_id, hist in norm_hist.items()
            }
        }

        with open(jsonl_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

        llm.llm_engine.collective_rpc(reset_counters)

    for i, question in enumerate(questions):
        messages = [
            {"role": "system", "content": "You are a helpful math tutor who explains step by step."},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

        outputs = llm.generate([prompt], sampling_params)
        print(f"Question {i}: {question}")
        print(f"Answer: {outputs[0].outputs[0].text}")

        save_and_clear_norm_history()

    print(f"Norm histories saved to {jsonl_file}")


if __name__ == "__main__":
    test_loop()