<div align="center">

<h1>🧬 Evol-SAM3</h1>

<h3>Evolving, Not Training: Zero-Shot Reasoning Segmentation via Evolutionary Prompting</h3>

[Kai Ye](https://github.com/yourusername)<sup>1</sup>, [Xiaotong You](https://github.com/yourusername)<sup>1</sup>, [Jianghang Lin](https://github.com/yourusername)<sup>1</sup>, [Jiayi Ji](https://github.com/yourusername)<sup>1,2</sup>, [Pingyang Dai](https://github.com/yourusername)<sup>1</sup>, [Liujuan Cao](https://github.com/yourusername)<sup>1</sup>

<sup>1</sup>Xiamen University, <sup>2</sup>National University of Singapore

<a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg"></a>
<a href="https://github.com/yourusername/Evol-SAM3/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E=1.12-ee4c2c.svg"></a>

<br>

<img src="assets/teaser.png" width="80%">

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
  <img src="assets/pipeline.png" width="95%">
</div>

Our framework consists of three phases:
* **Phase 1: Initialization.** A meta-generator expands the query into diverse hypotheses.
* **Phase 2: Evolutionary Loop.** Prompts compete in a **Visual Arena**, and winners undergo **Semantic Mutation** to breed better generations.
* **Phase 3: Final Arbitration.** A double-blind swap mechanism selects the best mask between evolutionary results and geometric priors.

---

## 📊 Performance

<div align="center">
  <img src="assets/performance.png" width="85%">
</div>

**Key Results:**
* **ReasonSeg:** Achieved **72.5 gIoU** (Zero-shot), surpassing the supervised LISA-13B (65.0 gIoU).
* **RefCOCO:** Outperformed SAM 3 Agent baseline by **+9.3 cIoU**.
* **Efficiency:** Optimal performance reached at just **Gen=2**.

---

## 🖼️ Qualitative Results

<div align="center">
  <img src="assets/qualitative.png" width="95%">
</div>

Comparison between **SAM3 Agent** (Baseline) and **Evol-SAM3** (Ours). Our method successfully handles functional descriptions and corrects visual biases.

---

## 🚀 Quick Start

(Coming Soon)

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
