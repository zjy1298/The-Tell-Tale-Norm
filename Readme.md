# 🏆 ICML 2026 Spotlight 🏆

# The Tell-Tale Norm: ℓ² Magnitude as a Signal for Reasoning Dynamics in Large Language Models

**Jinyang Zhang**\*, **Hongxin Ding**\*, **Yue Fang**\*, **Weibin Liao**\*, Muyang Ye, Junfeng Zhao, Yasha Wang
*(Peking University, Zhejiang University)*

[![arXiv](https://img.shields.io/badge/arXiv-2606.06188-b31b1b.svg)](https://arxiv.org/abs/2606.06188)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ICML 2026](https://img.shields.io/badge/ICML-2026%20Spotlight-FFD700.svg)]()

> **Reasoning leaves a trace. We found it in the ℓ² norm.**

![Poster](ICML 2026 Poster - The Tell-Tale Norm.png)

## 🚀 Introduction

How do Large Language Models (LLMs) reason internally? While previous works rely on expensive external probes (like SAEs) or output-level heuristics, we reveal that the **ℓ² norm of hidden states** serves as a robust, **endogenous signal** of reasoning intensity.

When an LLM engages in deep reasoning, its internal "energy" (ℓ² norm) spikes in the later layers. Building on this theoretical and empirical discovery, we introduce **three training-free, data-free, and plug-and-play** test-time scaling techniques that dynamically allocate compute and steer latent states.

Without any fine-tuning or extra data, our methods yield an **average gain of 4.51%** across diverse benchmarks, and up to **+9.13% on challenging reasoning tasks like AIME**!

## ✨ Key Features

- 🔍 **Intrinsic & Probe-Free**: No SAEs or external classifiers needed. The signal is native to the model's latent geometry.
- ⚡ **Training-Free & Data-Free**: Pure test-time interventions. Zero fine-tuning overhead.
- 🛠️ **vLLM-Compatible**: Fully integrated with vLLM for high-throughput, production-ready inference.
- 🌍 **Universal**: Effective across model families (Qwen, LLaMA, etc.) and domains (Math, Logic, Knowledge).
- 🎯 **Three Powerful Strategies**:
  1. **ALRR (Adaptive Layer-wise Reasoning Recursion)**: Dynamically recompute high-reasoning layers to "think longer" exactly when needed.
  2. **ERSS (Endogenous Reasoning State Steering)**: Inject historical high-norm states to amplify ongoing reasoning trajectories.
  3. **LRS (ℓ²-guided Response Selection)**: Unsupervised reranking to select the best output based on internal reasoning intensity.

## ⚡ Quick Start

### 1. Set up the environment
```bash
conda create -n telltale-norm
pip install -r requirement.txt
conda activate telltale-norm
```

### 2. Run evaluation with ℓ²-guided recursion
```bash
VLLM_PATCH=loop \
PATCHED_LAYERS="21,22,23,24,25,26" \
PATCHED_TAU_JSON='{"21":1300,"22":1550,"23":1850,"24":2300,"25":2800,"26":3600}' \
python -m lm_eval --model vllm --model_args pretrained=your/model/path \
                  --tasks gsm8k,aime24 --batch_size auto

> Replace your/model/path with your local or Hugging Face model path.
```
### 3. Try other methods
- For Response Selection (LRS): `set VLLM_PATCH=rerank`
- For State Steering (ERSS): `set VLLM_PATCH=ttts`

All methods are elegantly controlled via environment variables—no code changes needed!

## 🤝 How It Works

1. **Detect**: Compute ℓ² norms of hidden states in the later layers during decoding.
2. **Adapt**: Use dynamic, IQR-based thresholds to identify "reasoning peaks" without global hyperparameter tuning.
3. **Intervene**: Apply ALRR, ERSS, or LRS to amplify reasoning or select the most deliberate output.

Because the ℓ² norm is intrinsic to the model, our approach adds minimal overhead (<5% latency) while providing a principled lens to perceive and control latent reasoning dynamics.

## 📜 Citation

If you find this work useful, please cite our ICML 2026 Spotlight paper:
```bibtex
@inproceedings{zhang2026telltalenorm,
  title={The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models},
  author={Zhang, Jinyang and Ding, Hongxin and Fang, Yue and Liao, Weibin and Ye, Muyang and Zhao, Junfeng and Wang, Yasha},
  booktitle={Forty-third International Conference on Machine Learning (ICML)},
  year={2026},
  note={Spotlight}
}
```

## 🙏 Acknowledgements

We thank the reviewers for their insightful feedback. This work is supported by the National Natural Science Foundation of China.
