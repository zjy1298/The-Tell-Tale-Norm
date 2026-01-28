# -*- coding: utf-8 -*-
"""
vLLM suppression patch with adaptive tau estimation.
Supports two modes: "high_norm" (threshold-based) and "random" (probability-based).
Adaptive mode automatically enables if no fixed tau is provided via environment variables.
"""

import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union

from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import qwen2, qwen3, llama


# ==========================================================
# ===================== Helper functions ====================
# ==========================================================

def _get_attn_metadata():
    """Retrieve attention metadata from forward context."""
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
    """Scale tensor by multiplicative factor."""
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
    """Reconstruct hidden states so that hidden + residual = real_output_new."""
    if residual is None:
        return real_output_new, None
    return real_output_new - residual, residual


def _suppress_masked(hidden_states: torch.Tensor,
                     residual: Optional[torch.Tensor],
                     tau: float,
                     mask: torch.Tensor,
                     action: str,
                     mul_scale: float,
                     clamp_div: float):
    """Apply suppression only to tokens where mask=True."""
    real_output = hidden_states if residual is None else (hidden_states + residual)

    if action == "mul":
        real2 = real_output.clone()
        real2[mask] = _apply_mul(real_output[mask], mul_scale)
    elif action == "clamp":
        real2 = real_output.clone()
        real2[mask] = _apply_clamp(real_output[mask], tau, clamp_div)
    else:
        raise ValueError(f"Unknown action={action}")

    return _write_back_real_output(hidden_states, residual, real2)


def _parse_bool_env(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    v = os.environ.get(name, None)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _parse_layers_env(s: str) -> List[int]:
    """Parse comma-separated layer IDs from string."""
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


# ==========================================================
# ========= Adaptive quantile (P²) for IQR tau ==============
# ==========================================================

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
                self.np = [1, 1 + 2*q, 1 + 4*q, 3 + 2*q, 5]
                self.dn = [0.0, q/2.0, q, (1.0+q)/2.0, 1.0]
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

        for i in range(k+1, 5):
            self.n[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in (1, 2, 3):
            d = self.np[i] - self.n[i]
            if (d >= 1 and self.n[i+1] - self.n[i] > 1) or (d <= -1 and self.n[i-1] - self.n[i] < -1):
                dsign = 1 if d > 0 else -1

                x_i = self.x[i]
                n_i = self.n[i]
                n_im1 = self.n[i-1]
                n_ip1 = self.n[i+1]
                x_im1 = self.x[i-1]
                x_ip1 = self.x[i+1]

                num = (
                    dsign * (n_i - n_im1 + dsign) * (x_ip1 - x_i) / (n_ip1 - n_i) +
                    dsign * (n_ip1 - n_i - dsign) * (x_i - x_im1) / (n_i - n_im1)
                )
                x_new = x_i + num / (n_ip1 - n_im1)

                if x_im1 < x_new < x_ip1:
                    self.x[i] = x_new
                else:
                    self.x[i] = x_i + dsign * (self.x[i + dsign] - x_i) / (self.n[i + dsign] - n_i)

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


# ==========================================================
# =================== Configuration parsing =================
# ==========================================================

def _parse_tau_env(layers: List[int]) -> Dict[int, float]:
    """
    Parse tau thresholds from environment variables.
    Priority:
      1) PATCHED_TAU_JSON='{"21":1300,...}'
      2) VLLM_SUPPRESS_TAU='21:1300,22:1550,...'
    """
    tau_json = os.environ.get("PATCHED_TAU_JSON", "").strip()
    if tau_json:
        d = json.loads(tau_json)
        return {int(k): float(v) for k, v in d.items()}

    tau_str = os.environ.get("VLLM_SUPPRESS_TAU", "").strip()
    if tau_str:
        d: Dict[int, float] = {}
        for pair in tau_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split(":")
            d[int(k.strip())] = float(v.strip())
        return d

    return {}  # empty dict triggers adaptive mode


def _random_p_runtime(cfg: Dict) -> float:
    """Compute effective suppression probability at runtime."""
    k = int(cfg["random_k_per_response"])
    if k > 0:
        return float(k) / float(max(1, int(cfg["est_avg_decode_len"])))
    return float(cfg["random_p"])


def _load_cfg_from_env() -> Dict:
    """Load all configuration from environment variables."""
    mode = os.environ.get("VLLM_SUPPRESS_MODE", "high_norm").strip().lower()
    layers = _parse_layers_env(os.environ.get("PATCHED_LAYERS", "").strip())
    layer_tau = _parse_tau_env(layers)

    # Adaptive parameters
    adaptive_warmup_steps = int(os.environ.get("ADAPTIVE_WARMUP_STEPS", "500"))
    adaptive_iqr_k = float(os.environ.get("ADAPTIVE_IQR_K", "1.5"))
    adaptive_min_iqr = float(os.environ.get("ADAPTIVE_MIN_IQR", "1e-3"))
    adaptive_tau_floor = float(os.environ.get("ADAPTIVE_TAU_FLOOR", "0.0"))

    # Enable adaptive mode if no fixed tau provided
    adaptive_enabled = (len(layer_tau) == 0) or os.environ.get("ADAPTIVE", "True") == "True"

    cfg = {
        "mode": mode,  # "high_norm" | "random"
        "layers": layers,
        "layer_tau": layer_tau,

        "only_decode": _parse_bool_env("VLLM_SUPPRESS_ONLY_DECODE", True),

        "action": os.environ.get("VLLM_SUPPRESS_ACTION", "mul").strip().lower(),
        "mul_scale": float(os.environ.get("VLLM_SUPPRESS_MUL_SCALE", "0.1")),
        "clamp_div": float(os.environ.get("VLLM_SUPPRESS_CLAMP_DIV", "4.0")),

        "random_p": float(os.environ.get("VLLM_RANDOM_SUPPRESS_P", "0.05")),
        "random_k_per_response": int(os.environ.get("VLLM_RANDOM_SUPPRESS_K_PER_RESPONSE", "0")),
        "est_avg_decode_len": int(os.environ.get("VLLM_EST_AVG_DECODE_LEN", "512")),
        "random_seed": int(os.environ.get("VLLM_RANDOM_SEED", "1234")),

        # Adaptive mode
        "adaptive_enabled": adaptive_enabled,
        "adaptive_warmup_steps": adaptive_warmup_steps,
        "adaptive_iqr_k": adaptive_iqr_k,
        "adaptive_min_iqr": adaptive_min_iqr,
        "adaptive_tau_floor": adaptive_tau_floor,
    }

    if cfg["mode"] not in ("high_norm", "random"):
        raise ValueError(f"VLLM_SUPPRESS_MODE must be high_norm/random, got {cfg['mode']}")
    if cfg["action"] not in ("mul", "clamp"):
        raise ValueError(f"VLLM_SUPPRESS_ACTION must be mul/clamp, got {cfg['action']}")
    if not cfg["layers"]:
        raise ValueError("PATCHED_LAYERS is empty; must provide at least one layer ID.")

    return cfg


# ==========================================================
# ================= Worker stats tracking ===================
# ==========================================================

def _init_stats(worker, cfg: Dict, device=None):
    """Initialize suppression statistics on worker."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    worker._suppress_cfg = cfg

    worker._suppress_stats = {
        "prefill_calls": torch.zeros((), dtype=torch.int64, device=device),
        "decode_calls": torch.zeros((), dtype=torch.int64, device=device),
        "prefill_seqs": torch.zeros((), dtype=torch.int64, device=device),
        "decode_seq_steps": torch.zeros((), dtype=torch.int64, device=device),
        "suppressed_high": torch.zeros((), dtype=torch.int64, device=device),
        "suppressed_random": torch.zeros((), dtype=torch.int64, device=device),
        "layer_suppressed": {lid: torch.zeros((), dtype=torch.int64, device=device) for lid in cfg["layers"]},
        "rng": None,
        "adaptive_step": torch.zeros((), dtype=torch.int64, device=device),
    }

    # Initialize CPU-based RNG for random mode
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg["random_seed"]) + os.getpid())
    worker._suppress_stats["rng"] = g

    # Initialize adaptive quantile estimators if needed
    worker._adaptive = None
    if cfg["adaptive_enabled"]:
        worker._adaptive = {
            "q25": {lid: P2Quantile(0.25) for lid in cfg["layers"]},
            "q75": {lid: P2Quantile(0.75) for lid in cfg["layers"]},
            "count": {lid: 0 for lid in cfg["layers"]},
            "tau_map": {lid: None for lid in cfg["layers"]},
        }


def get_patch_stats(worker):
    """Return JSON-serializable suppression statistics."""
    s = worker._suppress_stats
    cfg = worker._suppress_cfg
    prefill_seqs = int(s["prefill_seqs"].item())

    out = {
        "pid": os.getpid(),
        "mode": cfg["mode"],
        "action": cfg["action"],
        "only_decode": cfg["only_decode"],
        "layers": cfg["layers"],
        "layer_tau": {str(k): float(v) for k, v in cfg["layer_tau"].items()},
        "prefill_calls": int(s["prefill_calls"].item()),
        "decode_calls": int(s["decode_calls"].item()),
        "prefill_seqs": prefill_seqs,
        "decode_seq_steps": int(s["decode_seq_steps"].item()),
        "layer_suppressed": {str(k): int(v.item()) for k, v in s["layer_suppressed"].items()},
        "adaptive_enabled": bool(cfg.get("adaptive_enabled", False)),
        "adaptive_warmup_steps": int(cfg.get("adaptive_warmup_steps", 0)),
        "adaptive_step": int(s["adaptive_step"].item()),
        "adaptive_iqr_k": float(cfg.get("adaptive_iqr_k", 0.0)),
        "adaptive_min_iqr": float(cfg.get("adaptive_min_iqr", 0.0)),
        "adaptive_tau_floor": float(cfg.get("adaptive_tau_floor", 0.0)),
    }

    # Include adaptive quantiles if enabled
    if cfg.get("adaptive_enabled", False) and worker._adaptive is not None:
        ad = worker._adaptive
        q_snapshot = {}
        for lid in cfg["layers"]:
            q1 = ad["q25"][lid].value()
            q3 = ad["q75"][lid].value()
            tau = ad["tau_map"].get(lid, None)
            q_snapshot[str(lid)] = {
                "count": int(ad["count"].get(lid, 0)),
                "q25": float(q1) if q1 == q1 else None,  # NaN-safe
                "q75": float(q3) if q3 == q3 else None,
                "tau": float(tau) if tau is not None else None,
            }
        out["adaptive_quantiles"] = q_snapshot

    # Mode-specific counters
    if cfg["mode"] == "high_norm":
        suppressed = int(s["suppressed_high"].item())
        out.update({
            "suppressed_high": suppressed,
            "avg_suppressed_high_per_response": (suppressed / prefill_seqs) if prefill_seqs > 0 else None,
        })
    else:
        suppressed = int(s["suppressed_random"].item())
        out.update({
            "suppressed_random": suppressed,
            "avg_suppressed_random_per_response": (suppressed / prefill_seqs) if prefill_seqs > 0 else None,
            "random_p_runtime": _random_p_runtime(cfg),
            "random_p_config": float(cfg["random_p"]),
            "random_k_per_response": int(cfg["random_k_per_response"]),
            "est_avg_decode_len": int(cfg["est_avg_decode_len"]),
        })

    return out


def reset_patch_stats(worker):
    """Reset all counters and adaptive estimators."""
    s = worker._suppress_stats
    s["prefill_calls"].zero_()
    s["decode_calls"].zero_()
    s["prefill_seqs"].zero_()
    s["decode_seq_steps"].zero_()
    s["suppressed_high"].zero_()
    s["suppressed_random"].zero_()
    s["adaptive_step"].zero_()
    for v in s["layer_suppressed"].values():
        v.zero_()

    # Reset adaptive estimators
    cfg = worker._suppress_cfg
    if cfg.get("adaptive_enabled", False) and worker._adaptive is not None:
        worker._adaptive = {
            "q25": {lid: P2Quantile(0.25) for lid in cfg["layers"]},
            "q75": {lid: P2Quantile(0.75) for lid in cfg["layers"]},
            "count": {lid: 0 for lid in cfg["layers"]},
            "tau_map": {lid: None for lid in cfg["layers"]},
        }
    return "reset_ok"


# ==========================================================
# =============== Patch model forward on worker =============
# ==========================================================

def apply_qwen2_patch(worker):
    """
    Apply suppression patch to Qwen2/Qwen3/Llama models on each vLLM worker.
    
    Adaptive mode:
      - Enabled automatically if no fixed tau is provided
      - During first ADAPTIVE_WARMUP_STEPS decode steps: only collect stats
      - Tau per layer: Q3 + ADAPTIVE_IQR_K * IQR, with safeguards
    """
    cfg = _load_cfg_from_env()
    _init_stats(worker, cfg)

    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        s = worker._suppress_stats
        cfg = worker._suppress_cfg
        ad = worker._adaptive

        # Pipeline parallelism prelude
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
            if cfg.get("adaptive_enabled", False):
                s["adaptive_step"].add_(1)
        else:
            s["prefill_calls"].add_(1)
            if hasattr(attn_metadata, "query_start_loc") and attn_metadata.query_start_loc is not None:
                num_seqs = int(attn_metadata.query_start_loc.shape[0] - 1)
                if num_seqs > 0:
                    s["prefill_seqs"].add_(num_seqs)

        adaptive_warmup = int(cfg.get("adaptive_warmup_steps", 0))
        in_adaptive_warmup = bool(cfg.get("adaptive_enabled", False) and decode_mode and int(s["adaptive_step"].item()) < adaptive_warmup)

        # Process layers
        for layer_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer], start=self.start_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # Skip if not a suppression layer or not in decode (if restricted)
            if layer_idx not in cfg["layers"]:
                continue
            if cfg["only_decode"] and (not decode_mode):
                continue

            # Determine tau: fixed from env or adaptive
            tau = cfg["layer_tau"].get(layer_idx, None)
            if tau is None and cfg.get("adaptive_enabled", False) and ad is not None:
                if decode_mode:
                    real_output = hidden_states if residual is None else (hidden_states + residual)
                    token_norm = real_output.norm(dim=-1)
                    val = float(token_norm.mean().item())

                    ad["q25"][layer_idx].add(val)
                    ad["q75"][layer_idx].add(val)
                    ad["count"][layer_idx] += 1

                    q1 = ad["q25"][layer_idx].value()
                    q3 = ad["q75"][layer_idx].value()
                    tau_new = _adaptive_tau_from_iqr(
                        q1=q1, q3=q3,
                        iqr_k=cfg["adaptive_iqr_k"],
                        min_iqr=cfg["adaptive_min_iqr"],
                        tau_floor=cfg["adaptive_tau_floor"],
                    )
                    ad["tau_map"][layer_idx] = tau_new

                tau = ad["tau_map"].get(layer_idx, None)

            if tau is None:
                continue

            real_output = hidden_states if residual is None else (hidden_states + residual)
            n_tokens = int(real_output.shape[0])
            if decode_mode:
                s["decode_seq_steps"].add_(n_tokens)

            # Skip suppression during adaptive warmup
            if in_adaptive_warmup:
                continue

            if cfg["mode"] == "high_norm":
                token_norm = real_output.norm(dim=-1)
                mask = token_norm > float(tau)
                if mask.any():
                    hidden_states, residual = _suppress_masked(
                        hidden_states, residual, float(tau), mask,
                        action=cfg["action"],
                        mul_scale=float(cfg["mul_scale"]),
                        clamp_div=float(cfg["clamp_div"]),
                    )
                    cnt = int(mask.sum().item())
                    s["suppressed_high"].add_(cnt)
                    s["layer_suppressed"][layer_idx].add_(cnt)

            else:  # random mode
                p = _random_p_runtime(cfg)
                if p <= 0:
                    continue
                m_cpu = torch.rand((n_tokens,), generator=s["rng"], device="cpu") < float(p)
                mask = m_cpu.to(real_output.device)
                if mask.any():
                    hidden_states, residual = _suppress_masked(
                        hidden_states, residual, float(tau), mask,
                        action=cfg["action"],
                        mul_scale=float(cfg["mul_scale"]),
                        clamp_div=float(cfg["clamp_div"]),
                    )
                    cnt = int(mask.sum().item())
                    s["suppressed_random"].add_(cnt)
                    s["layer_suppressed"][layer_idx].add_(cnt)

        # Pipeline parallelism epilogue
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # Apply to supported models
    qwen2.Qwen2Model.forward = patched_forward
    qwen3.Qwen3Model.forward = patched_forward
    llama.LlamaModel.forward = patched_forward

    # Report final tau values
    taus_out = {str(k): float(v) for k, v in cfg["layer_tau"].items()}
    if cfg.get("adaptive_enabled", False) and worker._adaptive is not None:
        for lid, v in worker._adaptive["tau_map"].items():
            if v is not None:
                taus_out[str(lid)] = float(v)

    return {
        "pid": os.getpid(),
        "patched": True,
        "mode": cfg["mode"],
        "layers": cfg["layers"],
        "taus": taus_out,
        "action": cfg["action"],
        "only_decode": cfg["only_decode"],
        "adaptive_enabled": bool(cfg.get("adaptive_enabled", False)),
        "adaptive_warmup_steps": int(cfg.get("adaptive_warmup_steps", 0)),
    }