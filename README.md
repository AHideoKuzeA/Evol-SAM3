<div align="center">

<h1>🧬 Evol-SAM3</h1>

<h3>Evolving, Not Training: Zero-Shot Reasoning Segmentation via Evolutionary Prompting</h3>

[Kai Ye](https://github.com/yourusername)<sup>1</sup>, [Xiaotong You](https://github.com/yourusername)<sup>1</sup>, [Jianghang Lin](https://github.com/yourusername)<sup>1</sup>, [Jiayi Ji](https://github.com/yourusername)<sup>1,2</sup>, [Pingyang Dai](https://github.com/yourusername)<sup>1</sup>, [Liujuan Cao](https://github.com/yourusername)<sup>1</sup>

<sup>1</sup>Xiamen University, <sup>2</sup>National University of Singapore

<a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg"></a>
<a href="https://github.com/yourusername/Evol-SAM3/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E=1.12-ee4c2c.svg"></a>

<br>

<img src="pic/first2-1.jpg" width="40%">

<p align="center">
  <strong>Evol-SAM3 reformulates reasoning segmentation as an inference-time evolutionary search.</strong><br>
  It achieves state-of-the-art zero-shot performance without updating any parameters.
</p>

</div>

---

## 🔥 News
* **[2025-12-31]** 🚀 Code and paper are released!
* **[2025-12-XX]** 🚧 We are preparing the demo on HuggingFace.

---

## 💡 Abstract

Reasoning Segmentation requires models to interpret complex linguistic queries for pixel-level localization. While current SFT and RL methods suffer from catastrophic forgetting and training instability, we propose **Evol-SAM3**, a novel **zero-shot framework**.

Instead of a static "generate-then-segment" paradigm, we model the task as an **Evolutionary Search** process:
1.  **Dynamic Evolution**: We maintain a population of prompts and refine them via a "Generate-Evaluate-Evolve" loop.
2.  **Visual Arena**: A tournament-based selection mechanism using MLLMs to assess mask quality without ground truth.
3.  **Semantic Mutation**: Injecting diversity and correcting hallucinations during inference.
4.  **Heterogeneous Arbitration**: A final safeguard combining text-based reasoning with geometric intuition.

Evol-SAM3 significantly outperforms static baselines (e.g., SAM3 Agent) and even fully supervised SOTA methods (e.g., LISA-13B) on **ReasonSeg** and **RefCOCO** benchmarks.

---

## 🛠️ Methodology

<div align="center">
  <img src="pic/pipline.jpg" width="95%">
</div>

Our framework consists of three phases:
* **Phase 1: Initialization.** A meta-generator expands the query into diverse hypotheses.
* **Phase 2: Evolutionary Loop.** Prompts compete in a **Visual Arena**, and winners undergo **Semantic Mutation** to breed better generations.
* **Phase 3: Final Arbitration.** A double-blind swap mechanism selects the best mask between evolutionary results and geometric priors.

---

## 📊 Performance

We conduct extensive experiments on **ReasonSeg** and **RefCOCO** series benchmarks. **Evol-SAM3** achieves superior performance without any parameter updates.

### 🏆 Comparison on ReasonSeg (Zero-Shot vs. SFT)

Evol-SAM3 (7B) outperforms not only other training-free agents but also fully supervised SOTA methods (e.g., LISA-13B).

| Method | Type | Backbone | Val gIoU | Test gIoU |
| :--- | :---: | :---: | :---: | :---: |
| **LISA** [CVPR'24] | SFT | LLaVA-1.5 13B | 65.0 | 61.3 |
| **GLaMM** [CVPR'24] | SFT | Vicuna 7B | 47.4 | -- |
| **SAM 3 Agent** [arXiv'25] | Training-free | Qwen2.5-VL 72B | **74.6** | 70.8 |
| **RSVP** [arXiv'25] | Training-free | GPT-4o | 64.7 | 60.3 |
| **Evol-SAM3 (Ours)** | **Training-free** | **Qwen2.5-VL 7B** | 70.7 | **72.5** |

> **Note:** Our 7B model surpasses the 72B baseline on the challenging Test set, proving the efficiency of evolutionary search.

<br>

### 📈 Comparison on Referring Expression Segmentation (Zero-Shot)

Comparison with state-of-the-art zero-shot methods on RefCOCO/+/g.

| Method | Backbone | RefCOCO (val) | RefCOCO+ (val) | RefCOCOg (val-U) |
| :--- | :---: | :---: | :---: | :---: |
| **SAM 3 Agent** | Qwen2.5-VL 7B | 59.4 | 51.4 | 57.2 |
| **Evol-SAM3 (Ours)** | **Qwen2.5-VL 7B** | **68.7** | **64.4** | **64.7** |
| *Improvement* | -- | <span style="color:green">**+9.3**</span> | <span style="color:green">**+13.0**</span> | <span style="color:green">**+7.5**</span> |

> All results are reported in **cIoU**. Evol-SAM3 significantly narrows the gap with supervised methods.

---

## 🖼️ Qualitative Results

<div align="center">
  <img src="pic/visual-1.jpg" width="95%">
</div>

Comparison between **SAM3 Agent** (Baseline) and **Evol-SAM3** (Ours). Our method successfully handles functional descriptions and corrects visual biases.

---

## 🚀 Quick Start

### 1. Environment Setup

Create and activate the conda environment:

```bash
conda env create -f Evol-SAM3.yml
conda activate Evol-SAM3
```

### 2. Data Preparation

Organize your datasets in the `DATASET` directory as follows:

```
DATASET/
├── reason_seg/
│   └── ReasonSeg/
│       ├── train/
│       ├── val/
│       └── test/
└── refer_seg/
    ├── images/
    │   └── train2014/
    ├── refcoco/
    ├── refcoco+/
    └── refcocog/
```

### 3. Model Weights

Download the required model weights:

- **SAM3 Checkpoint**: Download `sam3.pt` from [Hugging Face](https://huggingface.co/facebook/sam3) (requires access request).
- **MLLM Checkpoint**: 
The `MLLM` directory structure should look like this:
```
MLLM/
└── Qwen2.5_VL_7B/
    └── Qwen/
        └── Qwen2___5-VL-7B-Instruct/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── ...
```
Download [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) .

### 4. Configuration

Update the configuration file `configs/ReasonSeg_7B.yaml` to point to the correct paths:

```yaml
paths:
  qwen_model_path: "MLLM/Qwen2.5_VL_7B/Qwen/Qwen2___5-VL-7B-Instruct"
  sam3_ckpt_path: "sam3/sam3.pt"
  dataset_root: "DATASET/reason_seg/ReasonSeg"
  log_dir: "logs/ReasonSeg_7B"
```

### 5. Inference

Run the inference script:

```bash
bash ReasonSeg_7B.sh
```

This script will execute:
```bash
python main.py --config configs/ReasonSeg_7B.yaml
```

Resume Inference:
If the inference is interrupted, you can resume it by specifying the log directory with the `--resume` argument:

```bash
python main.py --config configs/ReasonSeg_7B.yaml --resume logs/ReasonSeg_7B/ReasonSeg_7B_xxxxx
```
*(Replace `ReasonSeg_7B_xxxxx` with your actual log directory name)*

## 💻 Usage

(Coming Soon)

---

## 📝 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{ye2025evolsam3,
  title={Evolving, Not Training: Zero-Shot Reasoning Segmentation via Evolutionary Prompting},
  author={Ye, Kai and You, Xiaotong and Lin, Jianghang and Ji, Jiayi and Dai, Pingyang and Cao, Liujuan},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
