# MVSCD
Multi-View Self-Consistency Decoding for Ancient Stele Inscriptions OCR
# Environment
PyTorch == 1.7
transformers == 4.57.6
# Data Preparation
Before OCR evaluation, all images are preprocessed through a two-stage segmentation pipeline based on the **CRAFT** text detector.

### Pipeline Overview

```text
Raw Rubbing Image
        │
        ▼
CRAFT Detection
        │
        ▼
Column Segmentation
        │
        ▼
Column Images
        │
        ▼
Character Segmentation
        │
        ▼
Character Patches
```

### Step 1: Column Segmentation

We first perform column-level segmentation using CRAFT. Character bounding boxes detected by CRAFT are grouped into text columns according to the horizontal coordinates of their centers.

To preserve complete textual information:

- The top boundary of each column is extended to the top of the image.
- The bottom boundary is extended to the bottom of the image.
- The left and right boundaries are adjusted according to the midpoints between adjacent columns.

This strategy prevents character truncation and minimizes interference between neighboring columns.

### Step 2: Character Segmentation

For each extracted column image, character-level segmentation is performed using CRAFT.

The detected quadrilateral coordinates are converted into axis-aligned bounding boxes. To preserve incomplete strokes and edge information, each bounding box is expanded by 5 pixels in all directions before boundary clipping.

### CRAFT Model

Download the pretrained CRAFT model and place it in the following directory:

```text
pretrained/
└── craft_mlt_25k.pth
```

The official CRAFT implementation can be found at:

https://github.com/clovaai/CRAFT-pytorch

# How To Test
