# RADR-Det

Official code repository for the manuscript:

**RADR-Det: Lightweight Adaptive Re-Fusion Framework for Dense Oriented Object Detection in Remote Sensing Images

The framework is built around three main components:

- **C3k2_RFAConv** for backbone feature enhancement
- **DBRF-PF** for dual-stage bidirectional re-fusion in the neck
- **P2-Detail Branch** for high-resolution detail injection

The detector uses a four-scale oriented bounding box head on **P2–P5** feature levels.

## Method Overview

RADR-Det is designed for dense oriented object detection in remote sensing imagery. The overall network consists of a backbone, a dual-stage re-fusion neck, and an oriented detection head.

The backbone introduces **C3k2_RFAConv** to strengthen local structural representation and enhance edge and contour responses. The neck adopts **DBRF-PF**, which organizes feature interaction through a top-down semantic propagation stage and a bottom-up detail re-fusion stage. In addition, a **P2-Detail Branch** is used to inject high-resolution features into the fusion pipeline for better representation of dense small objects. The final head performs oriented bounding box prediction on four scales: **P2, P3, P4, and P5**.

## Repository Structure

```text
RADR-Det/
│  LICENSE.txt
│  README.txt
│  requirements.txt
│  train.py
│  val.py
│
├─Datasets
│      DIOR-R Filtered Subset.py
│      README.txt
│
└─ultralytics
    │  CONTRIBUTING.md
    │
    ├─cfg
    │  │  README.md
    │  │
    │  └─models
    │          DADR-Det.yaml
    │          yolo11-obb.yaml
    │          yolo11.yaml
    │
    └─nn
        │  autobackend.py
        │  tasks.py
        │  text_model.py
        │  __init__.py
        │
        ├─conv
        │      conv.py
        │      RFAConv.py
        │
        ├─modules
        │      activation.py
        │      block.py
        │      conv.py
        │      head.py
        │      transformer.py
        │      utils.py
        │      __init__.py
        │
        └─neck
                DBRF_PF.py
