# -*- coding: utf-8 -*-
"""
Self-contained vLLM patch (SUPPRESS ONLY, GLOBAL CONFIG ONLY)

Two independent suppression modes (choose one via SUPPRESS_MODE):
  - Mode1: "high_norm"
      Trigger: per-token L2 norm of real output > tau(layer)
      Action: either clamp to tau/4 or scale by 0.1 (direction preserved)
      No token lists required.

  - Mode2: "random"
      Trigger: random mask (via probability or approx K tokens per response)
      Action: same as above
      Controlled by RANDOM_SUPPRESS_P or RANDOM_SUPPRESS_K_PER_RESPONSE.

No RPC config passing. Only patch application and stats collection via collective_rpc.
Test code included.
"""

import os
import torch
from typing import Optional, Dict, List, Union

from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen2, qwen3


# ==========================================================
# =============== GLOBAL CONFIG (edit here) =================
# ==========================================================

# Choose mode: "high_norm" or "random"
SUPPRESS_MODE: str = "high_norm"

# Layers to apply suppression on
SUPPRESS_LAYER_IDS: List[int] = [21, 22, 23, 24, 25, 26]

LAYER_TAU: Dict[int, float] = {
    21: 1240.0, 22: 1541.0, 23: 1857.0,
    24: 2238.0, 25: 2778.0, 26: 3430.0
}

# Apply suppression only during decode phase?
SUPPRESS_ONLY_DECODE: bool = True

# Suppression action type
SUPPRESS_ACTION: str = "mul"  # or "clamp"
SUPPRESS_MUL_SCALE: float = 0.1          # used if action == "mul"
SUPPRESS_CLAMP_DIV: float = 4.0          # clamp to tau / SUPPRESS_CLAMP_DIV if "clamp"

# ---------- Random mode controls ----------
RANDOM_SUPPRESS_P: float = 0.05  # per-token suppression probability

# Alternative: suppress ~K tokens per full response
RANDOM_SUPPRESS_K_PER_RESPONSE: int = 0  # set >0 to override RANDOM_SUPPRESS_P
EST_AVG_DECODE_LEN: int = 512            # estimated average generation length

# RNG seed (worker-local, augmented by PID)
RANDOM_SEED: int = 1234


# ==========================================================
# ===================== Helper functions ===================
# ==========================================================

def _get_attn_metadata():
    return get_forward_context().attn_metadata


def _is_decode(attn_metadata) -> bool:
    """Detect if current forward pass is in decode (autoregressive) mode."""
    if attn_metadata is None:
        return False
    if hasattr(attn_metadata, "max_query_len") and attn_metadata.max_query_len is not None:
        return int(attn_metadata.max_query_len) == 1
    if hasattr(attn_metadata, "query_start_loc") and attn_metadata.query_start_loc is not None:
        qsl = attn_metadata.query_start_loc
        if qsl.numel() >= 2:
            max_q = int((qsl[1:] - qsl[:-1]).max().item())
            return max_q == 1
    return False


def _apply_mul(x: torch.Tensor, mul: float) -> torch.Tensor:
    return x * float(mul)


def _apply_clamp(x: torch.Tensor, tau: float, div: float) -> torch.Tensor:
    """Clamp vector norms to at most tau/div, preserving direction."""
    target = float(tau) / float(div)
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = (target / n).clamp_max(1.0)  # only shrink, never expand
    return x * scale


def _write_back_real_output(hidden_states: torch.Tensor,
                            residual: Optional[torch.Tensor],
                            real_output_new: torch.Tensor):
    """Reconstruct hidden_states so that hidden + residual = real_output_new."""
    if residual is None:
        return real_output_new, None
    return real_output_new - residual, residual


def _suppress_masked(hidden_states: torch.Tensor,
                     residual: Optional[torch.Tensor],
                     tau: float,
                     mask: torch.Tensor):
    """Apply suppression only to tokens where mask=True."""
    real_output = hidden_states if residual is None else (hidden_states + residual)

    if SUPPRESS_ACTION == "mul":
        real2 = real_output.clone()
        real2[mask] = _apply_mul(real_output[mask], SUPPRESS_MUL_SCALE)
    elif SUPPRESS_ACTION == "clamp":
        real2 = real_output.clone()
        real2[mask] = _apply_clamp(real_output[mask], tau, SUPPRESS_CLAMP_DIV)
    else:
        raise ValueError(f"Unknown SUPPRESS_ACTION={SUPPRESS_ACTION}")

    return _write_back_real_output(hidden_states, residual, real2)


def _random_p_runtime() -> float:
    """Compute effective suppression probability at runtime."""
    if RANDOM_SUPPRESS_K_PER_RESPONSE > 0:
        return float(RANDOM_SUPPRESS_K_PER_RESPONSE) / float(max(1, EST_AVG_DECODE_LEN))
    return float(RANDOM_SUPPRESS_P)


# ==========================================================
# ============ Worker stats tracking + RPC helpers =========
# ==========================================================

def _init_stats(worker, device=None):
    """Initialize suppression statistics on worker."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._suppress_stats = {
        "prefill_calls": torch.zeros((), dtype=torch.int64, device=device),
        "decode_calls": torch.zeros((), dtype=torch.int64, device=device),
        "prefill_seqs": torch.zeros((), dtype=torch.int64, device=device),  # proxy for #responses
        "decode_seq_steps": torch.zeros((), dtype=torch.int64, device=device),  # total decode tokens processed
        "suppressed_high": torch.zeros((), dtype=torch.int64, device=device),
        "suppressed_random": torch.zeros((), dtype=torch.int64, device=device),
        "layer_suppressed": {lid: torch.zeros((), dtype=torch.int64, device=device)
                             for lid in SUPPRESS_LAYER_IDS},
        "rng": None,
    }

    # Initialize CPU-based RNG per worker (seed + PID for uniqueness)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(RANDOM_SEED) + os.getpid())
    worker._suppress_stats["rng"] = g


def get_suppress_stats(worker):
    """Return JSON-serializable suppression statistics."""
    s = worker._suppress_stats
    prefill_seqs = int(s["prefill_seqs"].item())

    out = {
        "pid": os.getpid(),
        "mode": SUPPRESS_MODE,
        "action": SUPPRESS_ACTION,
        "layers": SUPPRESS_LAYER_IDS,
        "layer_tau": {k: LAYER_TAU.get(k, None) for k in SUPPRESS_LAYER_IDS},
        "prefill_calls": int(s["prefill_calls"].item()),
        "decode_calls": int(s["decode_calls"].item()),
        "prefill_seqs": prefill_seqs,
        "decode_seq_steps": int(s["decode_seq_steps"].item()),
        "layer_suppressed": {k: int(v.item()) for k, v in s["layer_suppressed"].items()},
    }

    if SUPPRESS_MODE == "high_norm":
        suppressed = int(s["suppressed_high"].item())
        avg_per_resp = (suppressed / prefill_seqs) if prefill_seqs > 0 else None
        out.update({
            "suppressed_high": suppressed,
            "avg_suppressed_high_per_response": avg_per_resp,
        })
    elif SUPPRESS_MODE == "random":
        suppressed = int(s["suppressed_random"].item())
        avg_per_resp = (suppressed / prefill_seqs) if prefill_seqs > 0 else None
        out.update({
            "suppressed_random": suppressed,
            "avg_suppressed_random_per_response": avg_per_resp,
            "random_p_runtime": _random_p_runtime(),
            "random_p_config": RANDOM_SUPPRESS_P,
            "random_k_per_response": RANDOM_SUPPRESS_K_PER_RESPONSE,
            "est_avg_decode_len": EST_AVG_DECODE_LEN,
        })
    else:
        out["warning"] = f"Unknown SUPPRESS_MODE={SUPPRESS_MODE}"

    return out


def reset_suppress_stats(worker):
    """Reset all counters."""
    s = worker._suppress_stats
    s["prefill_calls"].zero_()
    s["decode_calls"].zero_()
    s["prefill_seqs"].zero_()
    s["decode_seq_steps"].zero_()
    s["suppressed_high"].zero_()
    s["suppressed_random"].zero_()
    for v in s["layer_suppressed"].values():
        v.zero_()
    return "reset_ok"


# ==========================================================
# =============== Patch model forward on worker =============
# ==========================================================

def apply_suppress_patch(worker):
    """
    Patch Qwen2/Qwen3 model forward pass on each vLLM worker.
    Applies suppression after each selected layer during decode (or always).
    """
    _init_stats(worker)

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        s = worker._suppress_stats

        # Pipeline parallelism: handle input embeddings
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Detect decode vs prefill
        attn_metadata = _get_attn_metadata()
        decode_mode = _is_decode(attn_metadata)

        if decode_mode:
            s["decode_calls"].add_(1)
        else:
            s["prefill_calls"].add_(1)
            if hasattr(attn_metadata, "query_start_loc") and attn_metadata.query_start_loc is not None:
                num_seqs = int(attn_metadata.query_start_loc.shape[0] - 1)
                if num_seqs > 0:
                    s["prefill_seqs"].add_(num_seqs)

        # Process layers
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer:self.end_layer],
            start=self.start_layer
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Skip if not a suppression layer or not in decode (if restricted)
            if layer_idx not in SUPPRESS_LAYER_IDS:
                continue
            if SUPPRESS_ONLY_DECODE and (not decode_mode):
                continue

            tau = LAYER_TAU.get(layer_idx, None)
            if tau is None:
                continue

            real_output = hidden_states if residual is None else (hidden_states + residual)
            n_tokens = int(real_output.shape[0])
            if decode_mode:
                s["decode_seq_steps"].add_(n_tokens)

            if SUPPRESS_MODE == "high_norm":
                token_norm = real_output.norm(dim=-1)  # [n_tokens]
                mask = token_norm > float(tau)
                if mask.any():
                    hidden_states, residual = _suppress_masked(hidden_states, residual, float(tau), mask)
                    cnt = int(mask.sum().item())
                    s["suppressed_high"].add_(cnt)
                    s["layer_suppressed"][layer_idx].add_(cnt)

            elif SUPPRESS_MODE == "random":
                p = _random_p_runtime()
                if p <= 0:
                    continue
                m_cpu = torch.rand((n_tokens,), generator=s["rng"], device="cpu") < float(p)
                mask = m_cpu.to(real_output.device)
                if mask.any():
                    hidden_states, residual = _suppress_masked(hidden_states, residual, float(tau), mask)
                    cnt = int(mask.sum().item())
                    s["suppressed_random"].add_(cnt)
                    s["layer_suppressed"][layer_idx].add_(cnt)

            else:
                raise ValueError(f"Unknown SUPPRESS_MODE={SUPPRESS_MODE}")

        # Final pipeline handling
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Apply to supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward

    return {
        "pid": os.getpid(),
        "patched": True,
        "mode": SUPPRESS_MODE,
        "layers": SUPPRESS_LAYER_IDS,
        "taus": {k: LAYER_TAU.get(k, None) for k in SUPPRESS_LAYER_IDS},
    }


# ==========================================================
# ========================= TEST ============================
# ==========================================================

def test_suppress_only():
    """Example usage: patch workers, generate, and collect stats."""
    from vllm import LLM, SamplingParams

    MODEL_PATH = "your/model/path"  # Anonymized model path

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=True,        # disable CUDA graphs for patch compatibility
        compilation_config=0,      # disable torch.compile
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    print("patch_results:", llm.llm_engine.collective_rpc(apply_suppress_patch))
    llm.llm_engine.collective_rpc(reset_suppress_stats)

    tokenizer = llm.get_tokenizer()
    question = """Solve this problem step by step:

Three friends Alice, Bob, and Carol share some money. Alice gets 1/3 of the total,
Bob gets 1/4 of the remaining, and Carol gets the rest which is $60.
How much money did they have in total?

Please show your detailed reasoning.
"""
    messages = [
        {"role": "system", "content": "You are a helpful math tutor who explains step by step."},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=2048
    )

    outputs = llm.generate([prompt] * 4, params)  # batch for stable statistics
    for i in range(min(4, len(outputs))):
        print(f"Response {i}:\n{outputs[i].outputs[0].text}\n")

    stats_all = llm.llm_engine.collective_rpc(get_suppress_stats)
    print("Suppression stats:", stats_all)


if __name__ == "__main__":
    test_suppress_only()