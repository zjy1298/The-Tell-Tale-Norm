# -*- coding: utf-8 -*-
import copy
import atexit
import time
import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Union

from importlib.metadata import version
from importlib.util import find_spec
from typing import TYPE_CHECKING

from more_itertools import distribute
from packaging.version import parse as parse_version
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    handle_stop_sequences,
    undistribute,
)
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)

try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass


def _quantile_from_layer_stats(layer_stats: Dict, q: float) -> float:
    """
    Extract a quantile value from layer statistics dict.
    Args:
        layer_stats: dict containing keys like "p50", "p90", etc.
        q: target quantile in (0, 1), e.g., 0.95
    Returns:
        Interpolated or exact quantile value.
    """
    assert 0.0 < q < 1.0
    pts = []
    for k, v in layer_stats.items():
        if isinstance(k, str) and k.startswith("p"):
            try:
                p = int(k[1:]) / 100.0
                pts.append((p, float(v)))
            except Exception:
                pass
    if not pts:
        raise ValueError("No quantile keys like p90/p95 found in layer_stats")

    pts.sort(key=lambda x: x[0])

    # Exact match
    for p, val in pts:
        if abs(p - q) < 1e-12:
            return float(val)

    # Clamp to boundaries
    if q <= pts[0][0]:
        return float(pts[0][1])
    if q >= pts[-1][0]:
        return float(pts[-1][1])

    # Linear interpolation
    for (p0, v0), (p1, v1) in zip(pts[:-1], pts[1:]):
        if p0 <= q <= p1:
            t = (q - p0) / (p1 - p0)
            return float(v0 + t * (v1 - v0))

    return float(pts[-1][1])


def _compute_and_set_patched_tau_env():
    """
    Read VLLM_PATCH_STATE_PATH and VLLM_PATCH_Q from environment,
    compute tau thresholds for top layers, and set PATCHED_LAYERS / PATCHED_TAU_JSON.
    """
    stats_path = os.environ.get("VLLM_PATCH_STATE_PATH", "").strip()
    q_str = os.environ.get("VLLM_PATCH_Q", "").strip()
    if not stats_path or not q_str:
        return

    q = float(q_str)
    if not (0.0 < q < 1.0):
        raise ValueError(f"VLLM_PATCH_Q must be in (0,1), got {q}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    per_layer = stats["per_layer"]
    layer_ids = sorted(int(k) for k in per_layer.keys())
    num_layers = max(layer_ids) + 1

    quarter = max(1, num_layers // 4)
    start = num_layers - quarter
    end = num_layers - 2
    patched_layers: List[int] = list(range(start, end + 1))

    tau_dict: Dict[int, float] = {}
    for lid in patched_layers:
        layer_stats = per_layer.get(str(lid))
        if layer_stats is not None:
            tau_dict[lid] = _quantile_from_layer_stats(layer_stats, q)

    os.environ["PATCHED_LAYERS"] = ",".join(map(str, patched_layers))
    os.environ["PATCHED_TAU_JSON"] = json.dumps(
        {str(k): float(v) for k, v in tau_dict.items()}, ensure_ascii=False
    )
    os.environ["PATCHED_Q_USED"] = str(q)


def _load_patch_fns():
    """Dynamically load patch functions based on VLLM_PATCH env var."""
    patch_mode = os.environ.get("VLLM_PATCH", "off").strip().lower()
    if patch_mode in ("", "off", "none", "0"):
        return None, None, None, patch_mode

    # ⚠️ ANONYMIZED: Replace with your actual patch module paths if needed.
    mod_name_map = {
        "suppress": "your.patch.module.suppress",
        "rr":     "your.patch.module.rr",
        "erss":     "your.patch.module.erss",
        "rerank":   "your.patch.module.rerank",
    }

    mod_name = mod_name_map.get(patch_mode)
    if mod_name is None:
        raise ValueError(f"Unknown VLLM_PATCH={patch_mode}")

    import importlib
    mod = importlib.import_module(mod_name)
    return (
        getattr(mod, "apply_qwen2_patch", None),
        getattr(mod, "get_patch_stats", None),
        getattr(mod, "reset_patch_stats", None),
        patch_mode
    )


@register_model("vllm")
class VLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        dtype: str = "auto",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        tokenizer_revision: Optional[str] = None,
        add_bos_token: bool = False,
        prefix_token_id: Optional[int] = None,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        max_gen_toks: int = 256,
        swap_space: int = 4,
        batch_size: Union[str, int] = 1,
        max_length: Optional[int] = None,
        max_model_len: Optional[int] = None,
        seed: int = 1234,
        gpu_memory_utilization: float = 0.9,
        device: str = "cuda",
        data_parallel_size: int = 1,
        lora_local_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if not find_spec("vllm"):
            raise ModuleNotFoundError(
                "Package `vllm` is not installed. Please install via `pip install vllm`."
            )

        assert "cuda" in device or device is None, "vLLM only supports CUDA"

        self._max_length = max_model_len if max_model_len is not None else max_length
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.data_parallel_size = int(data_parallel_size)

        self.model_args = {
            "model": pretrained,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "revision": revision,
            "dtype": dtype,
            "tokenizer": tokenizer,
            "tokenizer_mode": tokenizer_mode,
            "tokenizer_revision": tokenizer_revision,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": int(swap_space),
            "quantization": quantization,
            "seed": int(seed),
        }
        self.model_args.update(kwargs)

        self.batch_size = "auto" if isinstance(batch_size, str) and "auto" in batch_size else batch_size

        if self.data_parallel_size <= 1:
            self.model = LLM(**self.model_args)
            self._patch_name = os.environ.get("VLLM_PATCH", "off").strip().lower()
            self._patch_stats_path = os.environ.get("VLLM_PATCH_STATS_PATH", "").strip()
            self._patch_dump_every = int(os.environ.get("VLLM_PATCH_DUMP_EVERY", "0"))

            self._apply_qwen2_patch_fn, self._get_patch_stats_fn, self._reset_patch_stats_fn, self._patch_name = _load_patch_fns()

            if self._apply_qwen2_patch_fn is not None:
                self.model.llm_engine.collective_rpc(self._apply_qwen2_patch_fn)
                eval_logger.info(f"[patch] Applied VLLM_PATCH={self._patch_name} via collective_rpc")

                if self._patch_stats_path:
                    atexit.register(self._dump_patch_stats_once)
            elif self._apply_qwen2_patch_fn is not None and self.data_parallel_size > 1:
                eval_logger.warning("[patch] Patching not supported with data_parallel_size > 1.")
        else:
            eval_logger.warning("Data parallelism may cause issues during weight download. Use data_parallel_size=1 initially.")
            self.model_args["worker_use_ray"] = True
            self.batch_size = "auto"

            from transformers import AutoConfig
            self._config = AutoConfig.from_pretrained(
                pretrained, trust_remote_code=trust_remote_code, revision=revision
            )

        self.tokenizer = get_tokenizer(
            tokenizer if tokenizer else pretrained,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            revision=tokenizer_revision,
        )
        self.tokenizer = configure_pad_token(self.tokenizer)
        self.add_bos_token = add_bos_token
        if "gemma" in pretrained.lower():
            self.add_bos_token = True
            eval_logger.info("Gemma model detected: BOS token enabled.")

        self.custom_prefix_token_id = prefix_token_id
        self._max_gen_toks = max_gen_toks

        if lora_local_path is not None:
            assert parse_version(version("vllm")) > parse_version("0.3.0"), "LoRA requires vLLM > 0.3.0"
            self.lora_request = LoRARequest("finetuned", 1, lora_local_path)
        else:
            self.lora_request = None

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        if self.data_parallel_size <= 1:
            return self.model.llm_engine.model_config.max_model_len
        else:
            seqlen_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_attrs:
                if hasattr(self._config, attr):
                    return getattr(self._config, attr)
            if hasattr(self.tokenizer, "model_max_length"):
                if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                    return self._DEFAULT_MAX_LENGTH
                return self.tokenizer.model_max_length
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: Optional[int] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = self.add_bos_token
        encoding = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        if left_truncate_len:
            if isinstance(string, list):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]
        return encoding

    def _dump_patch_stats_once(self):
        if not self._patch_stats_path or self._get_patch_stats_fn is None:
            return
        try:
            stats_workers = self.model.llm_engine.collective_rpc(self._get_patch_stats_fn)
            payload = {
                "ts": time.time(),
                "patch": self._patch_name,
                "tokenizer": self.tokenizer_name,
                "model": self.model_args.get("model"),
                "stats_workers": stats_workers,
            }
            path = self._patch_stats_path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            if path.endswith(".jsonl"):
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

            eval_logger.info(f"[patch] dumped stats to {path}")
        except Exception as e:
            eval_logger.warning(f"[patch] failed to dump stats: {e}")

        if self._reset_patch_stats_fn is not None and self._patch_stats_path.endswith(".jsonl"):
            self.model.llm_engine.collective_rpc(self._reset_patch_stats_fn)

    def _model_generate(
        self,
        requests: List[List[int]],
        generate: bool = False,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )

        if self.data_parallel_size > 1:
            raise NotImplementedError("Data parallelism with tensor parallelism is unstable (see vLLM issue #973).")

        outputs = self.model.generate(
            prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=(self.batch_size == "auto"),
            lora_request=self.lora_request,
        )

        if getattr(self, "_patch_dump_every", 0) > 0 and self._patch_stats_path.endswith(".jsonl"):
            if not hasattr(self, "_patch_call_count"):
                self._patch_call_count = 0
            self._patch_call_count += 1
            if self._patch_call_count % self._patch_dump_every == 0:
                self._dump_patch_stats_once()

        return outputs

    # --- Remaining methods (loglikelihood_rolling, generate_until, etc.) unchanged ---
    # They are standard lm-eval methods and contain no sensitive info.
    # For brevity, they are omitted here but should be kept in full implementation.

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        # ... (standard implementation)
        pass

    def generate_until(self, requests, disable_tqdm=False):
        # ... (standard implementation)
        pass

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # ... (standard implementation)
        pass

    @staticmethod
    def _parse_logprobs(tokens, outputs, ctxlen):
        # ... (standard implementation)
        pass

    @staticmethod
    def modify_gen_kwargs(kwargs):
        # ... (standard implementation)
        pass