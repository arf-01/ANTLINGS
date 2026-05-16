<div align="center">
  <h1>🦅 ANTLINGS</h1>
  <p><b>Advanced Aerial Human & Car Detection with YOLOv11</b></p>
  <p><i>Real-time object detection and instance counting from drone imagery using the VisDrone 2019 dataset.</i></p>
</div>

---

## 📑 Table of Contents
1. [Dataset Understanding & Preprocessing](#1-dataset-understanding--preprocessing)
2. [Model Training](#2-model-training)
3. [Human & Car Detection with Human Counting](#3-human--car-detection-with-human-counting)
4. [Evaluation & Visualization](#4-evaluation--visualization)
5. [Key Performance Metrics](#5-key-performance-metrics)
6. [Project Analysis (Strengths & Limitations)](#6-project-analysis)
7. [Challenges Faced](#7-challenges-faced)

---

## 1. 📊 Dataset Understanding & Preprocessing

The foundation of this project is the **VisDrone 2019 DET** dataset, which provides high-resolution aerial imagery captured from drones.

**Dataset Composition:**
- **Train:** 6,471 images
- **Validation:** 548 images
- **Test:** 1,610 images
- **Original Resolution:** Variable sizes (e.g., up to 2000×1500), downscaled to 640×640 during training due to resource constraints.

### Preprocessing Highlights:
- **Class Filtering:** The original dataset contained 10 classes. We merged "pedestrian" and "people" into a single **"human"** class, resulting in 9 classes total. We then performed targeted label remapping to focus training on **Class 0 (Human)** and **Class 2 (Car)**.
- **Addressing Class Imbalance:** Data analysis revealed a strong imbalance, with Humans and Cars comprising ~73% of all training instances. We evaluated three strategies:
  1. Training only on Human/Car labels (Risk: background confusion).
  2. Full 9-class training followed by fine-tuning (Risk: catastrophic forgetting).
  3. **Full 9-class training (Chosen Strategy):** Selected to maintain optimal class boundaries and context awareness.
- **Data Localization:** To overcome Google Drive I/O bottlenecks (which slowed training to ~10 it/s), we implemented a script to dynamically localize the 8,600+ image dataset directly to the runtime NVMe disk, achieving read speeds of ~1997 MB/s.

---

## 2. 🧠 Model Training

We selected **YOLO11m (Medium)** due to its excellent balance of parameter size (20M) and computational efficiency (68.2 GFLOPs), making it ideal for detecting tiny objects typical in drone footage.

**Memory-Safe Optimized Training Configuration**
- **Epochs:** 30
- **Batch Size:** 4
- **Optimizer:** Auto-selected by Ultralytics as optimal default (AdamW, Learning Rate: 0.000769, Momentum: 0.9)
- **Precision:** Automatic Mixed Precision (AMP) enabled for VRAM efficiency.

### Aggressive Augmentation Pipeline
To combat variable drone altitudes, camera angles, and occlusions, we utilized a heavy augmentation strategy:
- **Mosaic (1.0) & Mixup (0.15):** Enhances context variation.
- **Copy-Paste (0.1):** Increases instance density.
- **Spatial Transformations:** Flip Left/Right (0.5), Flip Up/Down (0.5), Degrees (±10°).

---

## 3. 🧍🚗 Human & Car Detection with Human Counting

The core objective of the pipeline is not just bounding-box localization, but actionable intelligence.

- **Targeted Inference:** The model is optimized to confidently isolate humans and cars amidst complex urban backdrops (e.g., intersections, plazas, traffic jams).
- **Human Counting:** By aggregating detected instances per frame, the pipeline acts as an automated census tool. This is highly valuable for crowd estimation, search-and-rescue, and event monitoring from UAV feeds.

---

## 4. 📈 Evaluation & Visualization

Following the 30-epoch training cycle, the best weights (`best.pt`) were rigorously evaluated against the unseen **VisDrone test-dev** split.

- **Inference Sandbox:** `Antlings3.ipynb` serves as the evaluation suite, calculating latency, processing the test dataset, and generating annotated frames.
- **Visual Output:** A sample of 10 high-confidence detection frames has been exported to the `VisDrone_Detections/` directory, visually demonstrating the model's capability to draw tight bounding boxes around dense crowds and vehicle clusters.

---

## 5. ⚡ Key Performance Metrics

The model achieves **Real-Time Performance** on a Tesla T4 GPU, making it viable for live drone video streams.

**System Latency & Speed:**
- **Latency:** 22.29 ms per image (0.7ms preprocess + 18.2ms inference + 3.3ms postprocess)
- **Speed:** **44.86 FPS** 🚀

### Test-Set Detection Metrics

| Class | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **🚗 Car** | **0.7039** | **0.7486** | **0.3691** | **0.4608** |
| **🧍 Human** | 0.5105 | 0.3036 | 0.3691 | 0.1104 |

---

## 6. ⚖️ Project Analysis

### 💪 Strengths
- **Live-Stream Ready:** Processing speeds of ~45 FPS easily handle standard 30 FPS drone video feeds without frame dropping.
- **Robust Car Detection:** With ~70% precision and ~75% recall, the model reliably tracks vehicles regardless of orientation.
- **High Data Throughput:** Migrating data to local storage eliminated I/O bottlenecks and drastically reduced training time.
- **VRAM Efficiency:** Using AMP and optimized batch sizes allowed training a medium-tier model on standard cloud hardware without Out-Of-Memory (OOM) crashes.

### ⚠️ Limitations
- **Human Detection Recall (0.30):** The model struggles to detect all humans, missing roughly 70% of targets. Humans often occupy as few as 5-15 pixels in high-altitude shots, blending into the background.
- **Imprecise Human Bounding Boxes:** The low mAP50-95 score for humans (0.11) indicates that while the model finds a human, the bounding box edges are often not perfectly aligned.
- **Input Downscaling:** Shrinking large variable-sized images down to 640px due to GPU memory constraints destroys crucial micro-details needed for tiny object detection.

---

## 7. 🧗 Challenges Faced

| Challenge | Impact | Resolution Implemented |
| :--- | :--- | :--- |
| **Microscopic Object Size** | Humans disappear into pixel noise from high drone altitudes. | Scaled up from YOLO Nano to YOLO Medium (`11m`); applied Mosaic augmentation to force the model to look at varied object scales. |
| **Severe Class Imbalance** | The dataset is heavily skewed towards cars and pedestrians, neglecting other vehicle types. | Chose to train on all 9 classes to maintain strict classification boundaries, rather than forcefully dropping classes and confusing the model. |
| **Cloud Storage Bottlenecks** | Training directly off a mounted Google Drive yielded a sluggish 10 iterations/sec. | Wrote an automated Python localization script to copy the 5GB dataset directly to the Colab instance's high-speed NVMe before training. |

