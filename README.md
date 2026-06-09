# MVSCD
Multi-View Self-Consistency Decoding for Ancient Stele Inscriptions OCR
# Environment
PyTorch == 1.7 \
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

# Model Download

MVSCD relies on three pretrained models:

1. **Qwen3-VL-8B-Instruct** for OCR candidate generation.
2. **Xunzi-MLLM** for ancient Chinese text understanding and linguistic prior scoring.
3. **Chinese-CLIP** for cross-modal visual-semantic matching.

---

### Qwen3-VL-8B-Instruct

The Qwen3-VL-8B-Instruct model can be downloaded from ModelScope:

- ModelScope: https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct

Download example:

```bash
modelscope download \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir pretrained_models/Qwen3-VL-8B-Instruct
```

---

### Xunzi-Qwen3-8B

We use Xunzi-Qwen3-8B as the ancient Chinese language model for semantic verification and linguistic prior scoring.

- HuggingFace:
  https://huggingface.co/Xunzi-MLLM/Xunzi-Qwen3-8B

Download:

```bash
git lfs install

git clone \
https://huggingface.co/Xunzi-MLLM/Xunzi-Qwen3-8B \
pretrained_models/Xunzi-Qwen3-8B
```

---

### Chinese-CLIP (ViT-Huge/14)

We use the largest Chinese-CLIP model, **chinese-clip-vit-huge-patch14**, for visual-semantic similarity computation between candidate characters and OCR predictions.

- HuggingFace:
  https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14

- Official Repository:
  https://github.com/OFA-Sys/Chinese-CLIP

Download:

```bash
git lfs install

git clone \
https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14 \
pretrained_models/chinese-clip-vit-huge-patch14
```

# How To Test

After completing data preprocessing and downloading all pretrained models, modify the corresponding model and dataset paths in `LLM_agent.py`, then run:

```bash
python LLM_agent.py
```

MVSCD will automatically perform:

- OCR candidate generation
- Multi-view consistency decoding
- Visual-semantic matching
- Linguistic verification
- Final transcription generation

The recognition results will be saved to the output directory specified in `LLM_agent.py`.
