
<p align="center">
  <img src="/pic/pipline.jpg" alt="Fig. 3: SAARN overall framework" width="100%"><br>
</p>


<h1 align="center">Evolving, Not Training: Zero-Shot Reasoning Segmentation via Evolutionary Prompting</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2507.20920">
    <img src="https://img.shields.io/badge/arXiv-2507.20920-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://drive.google.com/file/d/1PmtaQH_F0AUoGWgpmDSpPu27E2XSdGd4/view?usp=sharing">
    <img src="https://img.shields.io/badge/Dataset-Google%20Drive-ffb300?logo=googledrive&logoColor=white" alt="coming soon">
  </a>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-4C9A2A" alt="Code coming soon">
</p>

<p align="center">
  <b>RIS-LAD</b> is the first fine-grained Referring Image Segmentation benchmark for <b>low-altitude drone (LAD)</b> scenes, featuring <b>13,871</b> image–text–mask triplets.<br>
  We introduce <b>SAARN</b> (Semantic-Aware Adaptive Reasoning Network) with <b>CDLE</b> and <b>ARFM</b> to tackle LAD-specific <i>category drift</i> and <i>object drift</i>.
</p>

---

## 🔗 Quick Links

- Paper (arXiv): https://arxiv.org/abs/2507.20920  
- Dataset (Google Drive): https://drive.google.com/file/d/1PmtaQH_F0AUoGWgpmDSpPu27E2XSdGd4/view?usp=sharing
- Model & Code: _coming soon_

---

## 🧾 Introduction

Low-altitude drones (typically operating below ~200 m) are increasingly deployed in real-world perception systems thanks to their flexibility and cost-effectiveness. However, most existing referring image segmentation (RIS) research focuses on conventional ground-view scenes or high-altitude remote sensing imagery. These settings differ substantially from low-altitude drone (LAD) views, where perspectives are oblique, objects are tiny and densely packed, and illumination varies widely (including night scenes).

To bridge this gap, we present <b>RIS-LAD</b>, a fine-grained benchmark specifically designed for <b>Referring Low-Altitude Drone Image Segmentation (RLADIS)</b>. RIS-LAD offers <b>13,871</b> carefully verified image–text–mask triplets collected from real LAD footage. Beyond providing data, RIS-LAD formalizes two key failure modes frequently observed under LAD settings:
- <b>Category drift</b>: when tiny targets cause models to latch onto larger, semantically similar objects.
- <b>Object drift</b>: when crowds of same-class instances lead to confusion about which instance the expression refers to.

We further propose <b>SAARN</b> (Semantic-Aware Adaptive Reasoning Network) to tackle these challenges. SAARN introduces:
- <b>CDLE</b> (Category-Dominated Linguistic Enhancement): injects <i>class-level</i> linguistic cues early in the encoder to anchor visual features to the correct category and suppress category drift.
- <b>ARFM</b> (Adaptive Reasoning Fusion Module): performs <i>scale-aware</i> fusion of <i>global (l)</i>, <i>class (c)</i>, and <i>descriptive (d)</i> text cues, enabling coarse→fine reasoning and disambiguation among dense same-class instances.

RIS-LAD is built with a semi-automatic pipeline that combines high-quality instance masks (prompted SAM-2) and multimodal LLM–generated initial expressions (given cropped instances and location cues), followed by human refinement. Experiments on RIS-LAD show that <b>SAARN</b> achieves state-of-the-art results on core segmentation metrics and yields pronounced gains at stricter localization thresholds (e.g., P@0.9), demonstrating stronger instance-level discrimination under LAD conditions.

---

## ✨ Highlights

<p align="center">
  <img src="first-1.jpg" alt="Fig. 1: RLADIS challenges teaser" width="45%"><br>
  <sub><b>Figure 1.</b> RLADIS challenges vs. RRSIS (category/object drift, tiny & dense objects, illumination).</sub>
</p>

- <b>Low-altitude & oblique views</b> (≈30–100 m, 30°–60°) → strong perspective change & foreshortening.

- <b>Tiny & dense targets</b> → easy confusion among many same-class instances.

- <b>Variable illumination</b> (incl. night) → distribution shift from standard RS/RIS datasets.

- <b>Category drift</b>: tiny targets bias the model to large, semantically similar objects.

- <b>Object drift</b>: crowded same-class instances hinder precise instance selection.

---

## 📊 Dataset Characteristics

| Dataset            | Image Source              | Shooting Angle | Nighttime Scene |
| ------------------ | ------------------------- | -------------- | --------------- |
| RefDIOR            | Google Earth              | fixed          | ✗               |
| NWPU-Refer         | Google Earth              | fixed          | ✗               |
| RISBench           | Google Earth, GF-2, JL-1  | fixed          | ✗               |
| RefSegRS           | Helicopter (above 1000 m) | fixed          | ✗               |
| RRSIS-D            | Google Earth              | fixed          | ✗               |
| **RIS-LAD (ours)** | **Drone (30–100 m)**      | **30°–60°**    | **✓**           |

---
## 📈 Benchmark Results

| Method           | Publication    | oIoU (Val) | oIoU (Test) | mIoU (Val) | mIoU (Test) |
| ---------------- | -------------- | ---------: | ----------: | ---------: | ----------: |
| LAVT             | CVPR 2022      |      44.03 |       41.97 |      32.25 |       30.14 |
| ASDA             | MM 2024        |      38.70 |       37.53 |      35.46 |       33.33 |
| VATEX            | WACV 2025      |      24.83 |       24.27 |      20.32 |       18.53 |
| LGCE             | IEEE TGRS 2024 |      41.72 |       40.75 |      27.68 |       26.17 |
| FIANet           | IEEE TGRS 2024 |      45.24 |       43.39 |      39.61 |       37.44 |
| RMSIN            | CVPR 2024      |      50.17 |       48.82 |      42.08 |       39.60 |
| RSRefSeg         | IGARSS 2025    |      50.04 |       47.71 |      43.42 |       41.16 |
| CADFormer        | JSTARS 2025    |      47.37 |       46.47 |      41.36 |       39.32 |
| **SAARN (ours)** | —              |  **51.54** |   **49.60** |  **44.30** |   **41.67** |

---

## 🖼️ Qualitative Comparisons

<p align="center">
  <img src="visual-1.jpg" alt="Qualitative comparisons: tiny objects & dense scenes" width="80%"><br>
  <sub><b>Qualitative Results.</b> SAARN vs. prior SOTA on tiny-object and dense same-class cases.</sub>
</p>


---

## 🧭 Roadmap

- [ ] Release training & evaluation code for SAARN  
- [ ] Add dataset loaders for popular toolkits  
- [ ] Provide pretrained weights & full configs  
- [ ] Add extended qualitative gallery and failure cases

---

## 📝 Citation

```bibtex
@misc{ye2025risladbenchmarkmodelreferring, 
  title        = {RIS-LAD: A Benchmark and Model for Referring Low-Altitude Drone Image Segmentation}, 
  author       = {Kai Ye and YingShi Luan and Zhudi Chen and Guangyue Meng and Pingyang Dai and Liujuan Cao},
  year         = {2025},
  eprint       = {2507.20920},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2507.20920}
}
```

---

## 📫 Contact & License

- Issues or questions: **yekai@stu.xmu.edu.cn**  
- Dataset: research use only (see dataset README)  
- Code: license will be provided upon release
