# -*- coding: utf-8 -*-
"""
Self-contained vLLM patch (Injection on low-norm streak):
1) Patch FlashAttentionImpl.forward to optionally skip KV-cache write (reads forward_context flag)
2) Patch the real model class' forward on each worker
3) Keep counters on worker instance (NO module globals)
4) All counters are torch tensors; RPC returns Python ints + histories
"""

import os
import json
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
from vllm.model_executor.models import qwen2, qwen3

# =========================
# Predefined thresholds
# =========================
LOOP_LAYER_IDS: List[int] = [21, 22, 23, 24, 25, 26]

LAYER_TAU: Dict[int, float] = {
    21: 1300.0, 22: 1550.0, 23: 1850.0,
    24: 2300.0, 25: 2800.0, 26: 3600.0
}

ALPHA = 1.0
GROWTH_EPS = 2e-1
PATIENCE_GROWTH = 1
BLOWUP_RATIO = 1.5
EPS = 1e-6
ONLY_DECODE = True

# =========================
# Injection hyperparameters (Scheme 1)
# =========================
LOW_K = 40              # Number of consecutive low-norm tokens to trigger injection
GAMMA = 1.0             # Low-norm threshold: norm < GAMMA * tau(layer)
P_INJECT = 0.2          # Probability of injection after trigger
BETA = 0.5              # Mixing strength for injected vector
COOLDOWN = 16           # Steps to wait before next injection for same sequence slot
BANK_K = 500            # Max number of high-norm vectors to store per sequence slot
COS_MIN = 0.5           # Minimum cosine similarity to allow injection
DEBUG_KEEP = 500        # Max number of debug records to keep
SAVE_BANK_VECTORS = True  # Whether to save bank vectors in snapshots
MAX_BANK_SAVES = 500      # Max number of bank snapshots to store

# ===========================================================
# 1) Patch flash-attn forward: read ctx.dont_save_kv_cache
# ===========================================================
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
    """Forward pass with optional KV cache write skip."""
    assert output is not None, "Output tensor must be provided."

    if attn_metadata is None:
        return output  # Profiling run

    num_actual_tokens = attn_metadata.num_actual_tokens
    key_cache, value_cache = kv_cache.unbind(0)

    ctx = get_forward_context()
    dont_save = getattr(ctx, "dont_save_kv_cache", False)

    if getattr(attn_metadata, "dont_save_kv_cache", False) or dont_save:
        pass
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
# 2) Worker-side stats and state
# ===========================================================
def _init_worker_state(worker, device=None) -> None:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._loop_stats = {
        "prefill": torch.zeros((), dtype=torch.int64, device=device),
        "decode": torch.zeros((), dtype=torch.int64, device=device),
        "bad": torch.zeros((), dtype=torch.int64, device=device),
        "norm_history": {lid: [] for lid in LOOP_LAYER_IDS},

        "inject_total": torch.zeros((), dtype=torch.int64, device=device),
        "inject_by_layer": {lid: torch.zeros((), dtype=torch.int64, device=device) for lid in LOOP_LAYER_IDS},
        "low_streak_triggered": torch.zeros((), dtype=torch.int64, device=device),
        "low_streak_max": torch.zeros((), dtype=torch.int64, device=device),
        "active_seqs": torch.zeros((), dtype=torch.int64, device=device),
        "inject_debug": [],
        "bank_snapshots": [],
        "token_counter": torch.zeros((), dtype=torch.int64, device=device),
    }

    worker._inj_state = {}


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


# ===========================================================
# 3) Patch model forward on each worker
# ===========================================================
def apply_forward_patch(worker):
    _init_worker_state(worker)

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
            return None, None, None
        x_hat = x / (x.norm() + 1e-6)
        v_hat = bank_vecs / (bank_vecs.norm(dim=-1, keepdim=True) + 1e-6)
        cos = (v_hat @ x_hat)
        best_cos, best_idx = torch.max(cos, dim=0)
        best_cos_val = float(best_cos.item())
        if best_cos_val < COS_MIN:
            return None, best_cos_val, None
        return bank_vecs[best_idx], best_cos_val, int(best_idx.item())

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        stats = worker._loop_stats

        # Pipeline parallelism prelude
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

        # Layer loop
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

            try:
                tau_f = float(tau)
                low_th = float(GAMMA) * tau_f

                # Pass 0: always run once with KV write enabled
                with _KVWriteGuard(dont_save=False):
                    h1, r1 = layer(positions, hidden_states, residual)

                real1 = h1 if r1 is None else (h1 + r1)
                norm1 = real1.norm(dim=-1)  # [num_tokens]
                stats["norm_history"][layer_idx].append(float(norm1.mean().item()))

                if decode_mode:
                    stats["active_seqs"].add_(int(norm1.numel()))

                # Injection logic (Scheme 1)
                if decode_mode:
                    rand = torch.rand((norm1.numel(),), device=norm1.device)
                    h_out = h1
                    r_out = r1
                    touched = False

                    for i in range(norm1.numel()):
                        token_id = int(stats["token_counter"].item())
                        stats["token_counter"].add_(1)

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

                        if nval < low_th:
                            st["low_streak"] += 1
                        else:
                            st["low_streak"] = 0

                        if st["low_streak"] >= LOW_K:
                            stats["low_streak_triggered"].add_(1)

                            if st["cooldown"] == 0 and float(rand[i].item()) < float(P_INJECT):
                                v_best, best_cos, best_id = _bank_best_by_cos(st, real1[i].detach())
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

                                    seq_lens = None
                                    if attn_metadata is not None and hasattr(attn_metadata, "seq_lens"):
                                        sl = attn_metadata.seq_lens
                                        if sl is not None:
                                            seq_lens = sl.detach().to("cpu").view(-1)
                                    seq_len_val = int(seq_lens[i].item()) if seq_lens is not None and i < seq_lens.numel() else None

                                    if len(stats["inject_debug"]) < DEBUG_KEEP:
                                        chosen_norm = float(v_best.norm().item())
                                        stats["inject_debug"].append({
                                            "layer": int(layer_idx),
                                            "seq_slot": int(seq_slot),
                                            "seq_len": seq_len_val,
                                            "token_id": token_id,
                                            "norm": float(nval),
                                            "tau": float(tau_f),
                                            "low_th": float(low_th),
                                            "cos": float(best_cos) if best_cos is not None else None,
                                            "chosen_norm": float(chosen_norm),
                                        })

                                    # Save bank snapshot
                                    if SAVE_BANK_VECTORS and len(stats["bank_snapshots"]) < MAX_BANK_SAVES:
                                        bank_vecs = st["bank_vecs"]
                                        bank_norms = st["bank_norms"]

                                        snapshot = {
                                            "layer": int(layer_idx),
                                            "seq_slot": int(seq_slot),
                                            "seq_len": seq_len_val,
                                            "token_id": token_id,
                                            "trigger_norm": float(nval),
                                            "bank_size": int(bank_vecs.shape[0]),
                                            "hidden_dim": int(bank_vecs.shape[1]) if bank_vecs.numel() > 0 else 0,
                                            "bank_vectors": bank_vecs.cpu().half().numpy().tolist() if bank_vecs.numel() > 0 else [],
                                            "bank_norms": bank_norms.cpu().numpy().tolist() if bank_norms.numel() > 0 else [],
                                            "chosen_idx": None,
                                            "chosen_norm": float(v_best.norm().item()),
                                            "cos_sim": float(best_cos),
                                        }

                                        # Find index of chosen vector
                                        if bank_vecs.numel() > 0:
                                            v_best_cpu = v_best.cpu().half()
                                            for idx in range(bank_vecs.shape[0]):
                                                if torch.allclose(bank_vecs[idx].cpu().half(), v_best_cpu, atol=1e-3):
                                                    snapshot["chosen_idx"] = int(idx)
                                                    break

                                        stats["bank_snapshots"].append(snapshot)

                                    st["low_streak"] = 0
                                    st["cooldown"] = COOLDOWN

                    hidden_states, residual = h_out, r_out
                else:
                    hidden_states, residual = h1, r1

            finally:
                ctx = get_forward_context()
                ctx.dont_save_kv_cache = False

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward

    return {
        "pid": os.getpid(),
        "patched_model_cls": "qwen2/qwen3",
        "stats_device": str(worker._loop_stats["prefill"].device),
    }


# ===========================================================
# 4) RPC helpers
# ===========================================================
def get_counters(worker):
    s = worker._loop_stats
    return {
        "prefill": int(s["prefill"].item()),
        "decode": int(s["decode"].item()),
        "bad": int(s["bad"].item()),
        "norm_history": s["norm_history"],
        "inject_total": int(s["inject_total"].item()),
        "inject_by_layer": {k: int(v.item()) for k, v in s["inject_by_layer"].items()},
        "low_streak_triggered": int(s["low_streak_triggered"].item()),
        "low_streak_max": int(s["low_streak_max"].item()),
        "active_seqs": int(s["active_seqs"].item()),
        "inject_debug": s["inject_debug"],
        "bank_snapshots": s["bank_snapshots"],
        "token_counter": int(s["token_counter"].item()),
        "config": {
            "LOW_K": LOW_K, "GAMMA": GAMMA, "P_INJECT": P_INJECT, "BETA": BETA,
            "COOLDOWN": COOLDOWN, "BANK_K": BANK_K, "COS_MIN": COS_MIN,
            "LOOP_LAYER_IDS": LOOP_LAYER_IDS, "LAYER_TAU": LAYER_TAU,
            "ONLY_DECODE": ONLY_DECODE,
            "SAVE_BANK_VECTORS": SAVE_BANK_VECTORS,
            "MAX_BANK_SAVES": MAX_BANK_SAVES,
        }
    }


def reset_counters(worker):
    s = worker._loop_stats
    s["prefill"].zero_()
    s["decode"].zero_()
    s["bad"].zero_()
    for k in s["norm_history"].keys():
        s["norm_history"][k].clear()

    s["inject_total"].zero_()
    for v in s["inject_by_layer"].values():
        v.zero_()
    s["low_streak_triggered"].zero_()
    s["low_streak_max"].zero_()
    s["active_seqs"].zero_()
    s["inject_debug"].clear()
    s["bank_snapshots"].clear()
    s["token_counter"].zero_()

    worker._inj_state.clear()
    return "reset_ok"


# ===========================================================
# 5) Example test (driver-side)
# ===========================================================
def test_inject():
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

    question = """Solve this problem step by step:

Three friends Alice, Bob, and Carol share some money. Alice gets 1/3 of the total,
Bob gets 1/4 of the remaining, and Carol gets the rest which is $60.
How much money did they have in total?
Please conclude your answer with "The answer is"
"""

    tokenizer = llm.get_tokenizer()
    messages = [
        {"role": "system", "content": "You are a helpful math tutor who explains step by step."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=4096,
    )

    num_samples = 4
    outputs = llm.generate([prompt] * num_samples, sampling_params)

    all_generated_tokens = []
    prompt_tokens = tokenizer.encode(prompt)
    prompt_length = len(prompt_tokens)

    print("\n" + "="*70)
    print("📝 Generation Results & Token Collection")
    print("="*70)

    for i in range(num_samples):
        output_text = outputs[i].outputs[0].text
        print(f"\n--- Output {i} ---")
        print(output_text)

        full_text = prompt + output_text
        full_tokens = tokenizer.encode(full_text)
        generated_tokens = full_tokens[prompt_length:]
        all_generated_tokens.extend(generated_tokens)

    # Build token_id -> token string mapping
    token_map = {}
    cumulative_token_id = 0

    for sample_idx in range(num_samples):
        output_text = outputs[sample_idx].outputs[0].text
        full_text = prompt + output_text
        full_tokens = tokenizer.encode(full_text)

        for pos, token_id in enumerate(full_tokens):
            token_str = tokenizer.decode([token_id])
            token_map[cumulative_token_id] = {
                'token_id_in_vocab': int(token_id),
                'token_str': token_str,
                'sample_idx': sample_idx,
                'position_in_seq': pos,
                'is_prompt': (pos < prompt_length),
            }
            cumulative_token_id += 1

    stats_all = llm.llm_engine.collective_rpc(get_counters)

    # Inject token info into stats
    for worker_stats in stats_all:
        if 'bank_snapshots' in worker_stats:
            for snap in worker_stats['bank_snapshots']:
                tid = snap.get('token_id', -1)
                if tid in token_map:
                    snap['token_info'] = token_map[tid]

        if 'inject_debug' in worker_stats:
            for record in worker_stats['inject_debug']:
                tid = record.get('token_id', -1)
                if tid in token_map:
                    record['token_info'] = token_map[tid]

    save_path = "your/output/file.json"  # Anonymized output path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(stats_all, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved stats with token info to: {save_path}")

    total_snapshots = sum(len(w.get("bank_snapshots", [])) for w in stats_all)
    print(f"\n📊 Captured {total_snapshots} bank snapshots")
    print(f"📊 Total tokens processed: {len(token_map)}")


if __name__ == "__main__":
    test_inject()