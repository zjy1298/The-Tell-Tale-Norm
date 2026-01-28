# -*- coding: utf-8 -*-
"""
Self-contained vLLM patch (ONLINE stats with stable req_id mapping):
- Patch GPUModelRunner._prepare_inputs to stash (req_ids, req_indices, num_reqs, total_tokens)
  into forward_context (NO print needed).
- Patch Qwen2/Qwen3 model forward to compute per-layer ABCD metrics online.
- Metrics are accumulated per request id (req_id) and per layer independently.
- After generation, driver pulls metrics via collective_rpc and aligns with outputs.

ABCD per layer l for each req_id:
A_sum_norm   = sum over decode steps of mean_token_norm(step,l)   (decode: 1 token/req)
B_event_area = sum over steps max(0, norm - tau_l)
C_event_count= sum over steps 1(norm > tau_l)
D_late_area  = sum over last LATE_FRAC steps of step_event_area   (computed at fetch time)

Important:
- We only record during DECODE steps (max_query_len == 1). Prefill is ignored by default.
- This patch assumes vLLM v1 and that your codebase matches the snippets shown.
"""

import os
import json
import torch
from typing import Optional, Dict, List, Union, Any

from vllm import LLM, SamplingParams
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen2, qwen3

# vLLM internal runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # type: ignore
import numpy as np


# =========================
# Configuration
# =========================
LOOP_LAYER_IDS: List[int] = [21, 22, 23, 24, 25, 26]
LAYER_TAU: Dict[int, float] = {
    21: 1300.0, 22: 1550.0, 23: 1850.0,
    24: 2300.0, 25: 2800.0, 26: 3600.0
}
ONLY_DECODE = True
LATE_FRAC = 0.30  # D: last 30% of decode steps


# ===========================================================
# 1) Patch GPUModelRunner._prepare_inputs: expose req_id mapping
# ===========================================================
def apply_runner_patch(worker):
    import numpy as np

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
    return {"pid": os.getpid(), "patched_runner": True}


# ===========================================================
# 2) Worker-side store + model forward patch
# ===========================================================
def _init_stats(worker):
    worker._norm_stats = {
        "prefill": 0,
        "decode": 0,
        # per req_id -> per layer -> ABC + step list for D
        "per_req": {},  # req_id(str) -> dict
        # for debug: record what mapping keys were present
        "meta": {"seen_req_ids": 0, "seen_req_indices": 0, "mapping_errors": 0},
    }


def apply_forward_patch(worker):
    _init_stats(worker)

    def _is_decode(attn_metadata) -> bool:
        if hasattr(attn_metadata, "max_query_len"):
            return int(attn_metadata.max_query_len) == 1
        if hasattr(attn_metadata, "query_start_loc"):
            qsl = attn_metadata.query_start_loc
            if qsl is not None and qsl.numel() >= 2:
                max_q = int((qsl[1:] - qsl[:-1]).max().item())
                return max_q == 1
        return False

    def _ensure_req(req_id: str):
        pr = worker._norm_stats["per_req"]
        if req_id not in pr:
            pr[req_id] = {
                "layers": {
                    lid: {"A_sum": 0.0, "B_area": 0.0, "C_count": 0, "B_steps": []}
                    for lid in LOOP_LAYER_IDS
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

        # Detect decode mode
        ctx = get_forward_context()
        attn_metadata = getattr(ctx, "attn_metadata", None)

        def _is_decode(attn_metadata) -> bool:
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

        decode_mode = _is_decode(attn_metadata)
        if ONLY_DECODE:
            if decode_mode:
                stats["decode"] += 1
            else:
                stats["prefill"] += 1
        else:
            stats["prefill"] += 1

        # Retrieve request mapping from metadata
        sm = getattr(attn_metadata, "scheduler_metadata", None) if attn_metadata is not None else None
        if sm is not None:
            req_ids = getattr(sm, "_normstat_req_ids", None)
            req_indices = getattr(sm, "_normstat_req_indices", None)
        else:
            req_ids = getattr(attn_metadata, "_normstat_req_ids", None) if attn_metadata is not None else None
            req_indices = getattr(attn_metadata, "_normstat_req_indices", None) if attn_metadata is not None else None

        # Update meta counters
        if req_ids is not None:
            stats["meta"]["seen_req_ids"] += 1
        else:
            stats["meta"]["mapping_errors"] += 1

        # Helper to initialize per-request storage
        def _ensure_req(req_id: str):
            pr = stats["per_req"]
            if req_id not in pr:
                pr[req_id] = {
                    "layers": {
                        lid: {"A_sum": 0.0, "B_area": 0.0, "C_count": 0, "B_steps": []}
                        for lid in LOOP_LAYER_IDS
                    },
                    "decode_steps": 0,
                }
            return pr[req_id]

        # Process layers
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer:self.end_layer],
            start=self.start_layer
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Skip if not decode or not a target layer
            if (not decode_mode) or (layer_idx not in LOOP_LAYER_IDS):
                continue

            tau = float(LAYER_TAU[layer_idx])
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
                    b = max(0.0, n - tau)
                    st["B_area"] += b
                    st["C_count"] += int(n > tau)
                    st["B_steps"].append(b)
                continue

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
                    b = max(0.0, n - tau)
                    st["B_area"] += b
                    st["C_count"] += int(n > tau)
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
                        b = max(0.0, n - tau)
                        st["B_area"] += b
                        st["C_count"] += int(n > tau)
                        st["B_steps"].append(b)
                else:
                    stats["meta"]["mapping_errors"] += 1

        # Pipeline parallelism epilogue
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward

    return {"pid": os.getpid(), "patched": True}


# ===========================================================
# 3) RPC helpers
# ===========================================================
def reset_stats(worker):
    _init_stats(worker)
    return "reset_ok"


def get_stats(worker):
    """Finalize D_late metric on worker side."""
    out = worker._norm_stats
    per_req = out["per_req"]
    for rid, rec in per_req.items():
        for lid, st in rec["layers"].items():
            b_steps = st["B_steps"]
            if not b_steps:
                st["D_late"] = 0.0
            else:
                start = max(0, int((1.0 - LATE_FRAC) * len(b_steps)))
                st["D_late"] = float(sum(b_steps[start:]))
    return out


def align(chosen_stats, outs):
    """Align metrics with generated outputs using request_id."""
    per_req = chosen_stats["per_req"]
    aligned_results = []
    for i, o in enumerate(outs):
        rid = str(o.request_id)
        text = o.outputs[0].text

        metrics = per_req.get(rid, None)
        aligned_results.append({
            "idx": i,
            "request_id": int(o.request_id),
            "text": text,
            "metrics_per_layer": metrics["layers"] if metrics is not None else None,
        })

    # Print summary
    for r in aligned_results:
        print("=" * 80)
        print("idx:", r["idx"], "request_id:", r["request_id"])
        print("text head:", r["text"][:200].replace("\n", "\\n"))
        if r["metrics_per_layer"] is None:
            print("NO METRICS")
            continue
        for lid, st in r["metrics_per_layer"].items():
            print(" layer", lid,
                  "A_sum", st.get("A_sum"),
                  "B_area", st.get("B_area"),
                  "C_count", st.get("C_count"),
                  "D_late", st.get("D_late"))


# ===========================================================
# 4) Test script: generate 4 answers and dump per-answer metrics
# ===========================================================
def test_online_stats():
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

    llm.llm_engine.collective_rpc(apply_runner_patch)
    llm.llm_engine.collective_rpc(apply_forward_patch)
    llm.llm_engine.collective_rpc(reset_stats)

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

    sp = SamplingParams(
        temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.05, max_tokens=1024
    )

    outs = llm.generate([prompt] * 4, sp)

    # Inspect request IDs
    for i, o in enumerate(outs):
        print(i, "request_id:", getattr(o, "request_id", None))

    texts = [o.outputs[0].text for o in outs]
    stats_workers = llm.llm_engine.collective_rpc(get_stats)

    # Choose worker with most recorded requests
    chosen = stats_workers[0]
    for w in stats_workers:
        if len(w.get("per_req", {})) > len(chosen.get("per_req", {})):
            chosen = w

    align(chosen, outs)
    per_req = chosen["per_req"]

    result = {
        "outputs": texts,
        "per_req_metrics": per_req,
        "meta": chosen.get("meta", {}),
        "config": {
            "layers": LOOP_LAYER_IDS,
            "tau": LAYER_TAU,
            "late_frac": LATE_FRAC,
            "only_decode": ONLY_DECODE,
        }
    }

    save_path = "your/output/file.json"  # Anonymized output path
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved to:", save_path)
    print("Meta:", result["meta"])
    print("Num req_ids in metrics:", len(per_req))
    print("Example req_id keys:", list(per_req.keys())[:8])
    print("Output0 head:", texts[0][:200].replace("\n", "\\n"))


if __name__ == "__main__":
    test_online_stats()