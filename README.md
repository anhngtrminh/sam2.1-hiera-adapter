# CDW-Seg

Semantic segmentation of **Construction and Demolition Waste (CDW)** built on top of [SAM 2.1](https://github.com/facebookresearch/segment-anything-2) (Segment Anything Model 2.1). The image encoder backbone (Hiera) is frozen and extended with lightweight FFT-based prompt adapters, so only a small fraction of parameters are trained (~5M of 225M).

---

## Table of Contents

- [Overview](#overview)
- [Supported Classes](#supported-classes)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Configuration Reference](#configuration-reference)
- [Checkpoints](#checkpoints)
- [Evaluation Metrics](#evaluation-metrics)
- [Multi-GPU Training](#multi-gpu-training)

---

## Overview

CDW-Seg fine-tunes SAM 2.1's Hiera backbone for multi-class semantic segmentation using a parameter-efficient adapter strategy:

- **Frozen backbone** — SAM 2.1 Hiera image encoder weights are kept frozen
- **FFT prompt adapters** — handcrafted frequency-domain prompts and lightweight embedding adapters are inserted at each stage of the backbone
- **FPN neck** — a Feature Pyramid Network projects multi-scale backbone features to a uniform 256-channel representation
- **SAM mask decoder** — the original SAM 2 two-way transformer decoder predicts per-class segmentation masks

Training uses standard COCO-format instance annotations which are automatically converted to semantic masks.

---

## Supported Classes

| Index | Class |
|-------|-------|
| 0 | background |
| 1 | bin |
| 2 | concrete\_bricks\_tiles |
| 3 | soils |
| 4 | green\_waste\_timber |
| 5 | plastic |
| 6 | metals\_e\_waste |
| 7 | non\_recyclable\_waste |
| 8 | cardboard |
| 9 | plaster\_board |

Additional classes (`paint_can`, `needles_syringes`, `dead_animal`) can be enabled in the config once labelled data is available.

---

## Project Structure

```
.
├── finetune.py                          # Main training script
├── configs/
│   └── train.yaml                       # Training configuration
├── models/
│   ├── sam.py                           # SAM model wrapper (SAM class)
│   ├── sam2/
│   │   └── modeling/
│   │       └── backbones/
│   │           ├── hieradet.py          # Hiera backbone + PromptGenerator adapters
│   │           └── image_encoder.py     # ImageEncoder (trunk + FPN neck)
│   └── mmseg/
│       └── models/
│           └── sam/
│               └── mask_decoder.py      # Two-way transformer mask decoder
├── pretrained/
│   └── sam2.1_hiera_large.pt            # SAM 2.1 pre-trained weights (download separately)
├── cdw_coco/                            # Dataset root (produced by prepare step)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
├── save/                                # Training outputs (checkpoints, logs, TensorBoard)
├── eval_iou.py                          # Segmentation metrics
└── utils.py                             # Logging, optimiser factory, timer
```

---

## Requirements

```
torch >= 2.0
torchvision
pycocotools
pyyaml
numpy
Pillow
tqdm
prettytable
tensorboard
mmcv
```

Install dependencies:

```bash
pip install torch torchvision pycocotools pyyaml numpy pillow tqdm prettytable tensorboard mmcv
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-org/cdw-seg.git
cd cdw-seg
```

**2. Download the SAM 2.1 checkpoint**

```bash
mkdir -p pretrained
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Or for the smaller variant:

```bash
wget -P pretrained https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

---

## Dataset Preparation

Your source dataset must be a flat folder with images and a single COCO-format annotation file:

```
cdw_coco/
├── images/          # image files (.jpg / .png / ...)
└── annotations.json # single COCO JSON covering all images
```

Run the preparation step once to split and restructure into train/val/test:

```bash
python finetune.py --config configs/train.yaml --prepare-only
```

This produces the standard COCO layout under `coco_root` (configured in `train.yaml`). To force a re-split if source data changes:

```bash
python finetune.py --config configs/train.yaml --prepare-only --force-prepare
```

The default split ratios are 80% train / 10% val / 10% test (configurable via `split_preparation` in the config).

---

## Training

**Single GPU:**

```bash
python finetune.py --config configs/train.yaml --name my_run
```

**Outputs** are written to `save/my_run/`:
- `model_epoch_best.pth` — checkpoint with best validation mIoU
- `model_epoch_last.pth` — most recent checkpoint
- `config.yaml` — copy of the config used
- `train.log` — full training log
- TensorBoard events (view with `tensorboard --logdir save/my_run`)

**Resume from checkpoint:**

```yaml
# in train.yaml
resume:      save/my_run/model_epoch_last.pth
start_epoch: 11
```

---


### RAM reduction tips

If you run out of memory, apply these in order:

| Change | Effect |
|--------|--------|
| `batch_size: 1` + increase `accum_steps` | Largest single reduction |
| `inp_size: 512` (from 1024) | ~4× fewer activations |
| `use_amp: true` | ~2× memory via float16/bfloat16 |
| `grad_checkpoint: true` | Trades compute for activation memory |
| `freq_nums: 0.10` (from 0.25) | Smaller FFT prompt tensors |

---

## Checkpoints

| Model | Params | Config |
|-------|--------|--------|
| SAM 2.1 Hiera-Large | 225M total / ~5M trainable | `embed_dim: 144`, `stages: [2,6,36,4]` |
| SAM 2.1 Hiera-Small | ~45M total / ~3M trainable | `embed_dim: 96`, `stages: [1,2,11,2]` |

> **Note:** the checkpoint variant must match the model architecture in `encoder_mode`. Loading a Large checkpoint into a Small model (or vice versa) will raise a `RuntimeError` on `load_state_dict`.

---

## Evaluation Metrics

Validation runs automatically every `epoch_val` epochs and reports:

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union across all classes |
| **IoU** | Per-class IoU |
| **Precision / Recall / F1** | Per-class and mean |
| **OA** | Overall pixel accuracy |
| **FwIoU** | Frequency-weighted IoU |
| **Confusion matrix** | Normalised per-column |

Results are printed as a table to the log and written to TensorBoard under `val/mIoU`.

---

## Multi-GPU Training

Distributed Data Parallel training is supported via `torchrun`:

```bash
torchrun --nproc_per_node=2 finetune.py --config configs/train.yaml --name my_run
```

On CUDA the `nccl` backend is used automatically; on CPU it falls back to `gloo`.