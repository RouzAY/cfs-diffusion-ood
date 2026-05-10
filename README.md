# CFS: Backbone-Equated Diffusion OOD via Sparse Internal Snapshots

Official implementation of:

**Canonical Feature Snapshots (CFS)** and  
**Mutualized Backbone Evaluation (MBE)** for diffusion-based OOD detection

---

## Overview

This repository provides a **reproducible and fair evaluation framework** for out-of-distribution (OOD) detection using diffusion models.

We introduce:

* **CFS (Canonical Feature Snapshots)** — a lightweight, training-free OOD detector (1 forward pass)
* **MBE (Mutualized Backbone Evaluation)** — a protocol ensuring *strict fairness across methods*

We benchmark against:

* MSMA
* DiffPath
* DDPM-OOD
* GEPC

---

## Key Contributions

* **Training-free OOD detection** via canonical feature extraction
* **Minimal compute (1F)** vs heavy diffusion baselines
* **Strict protocol alignment (MBE)**:

  * same backbone
  * same corruption levels
  * same compute budget
* **Clean separation** between:

  * *main benchmark (MBE)*
  * *external experiments (ablations, variants)*

---

## Installation

```bash
conda create -n cfs-ood python=3.10
conda activate cfs-ood

pip install -r requirements.txt
```

---

## Checkpoints (Reproducibility)

We **do not redistribute checkpoints**.

To ensure reproducibility, traceability, and licensing compliance, please download all pretrained models from their original sources and place them under:

```text
checkpoints/
```

### Improved Diffusion (OpenAI)

Repository:

```text
https://github.com/openai/improved-diffusion
```

ImageNet-64 checkpoint used by the large-scale configs:

```bash
mkdir -p checkpoints
wget -O checkpoints/imagenet64_uncond_100M_1500K.pt \
  https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt
```

Optional CIFAR-10 checkpoint:

```bash
wget -O checkpoints/cifar10_uncond_50M_500K.pt \
  https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt
```

---

### EDM (NVIDIA)

Repository:

```text
https://github.com/NVlabs/edm
```

CIFAR-10 unconditional VP checkpoint:

```bash
wget -O checkpoints/edm-cifar10-32x32-uncond-vp.pkl \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

If this exact file is unavailable in your EDM mirror, use the baseline VP checkpoint and update the YAML path accordingly:

```bash
wget -O checkpoints/baseline-cifar10-32x32-uncond-vp.pkl \
  https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/baseline/baseline-cifar10-32x32-uncond-vp.pkl
```

---

### U-ViT

Repository:

```text
https://github.com/baofff/U-ViT
```

Clone the repository and follow its checkpoint instructions:

```bash
mkdir -p repos
git clone https://github.com/baofff/U-ViT.git repos/U-ViT-main
```

Then place the corresponding U-ViT checkpoint under:

```text
checkpoints/
```

---

### Additional Checkpoints (CelebA)

Some source-family robustness experiments in the paper use CelebA pretrained diffusion models that are **not part of the original OpenAI improved-diffusion or NVIDIA EDM checkpoint releases**.

These models are sourced from prior work and should be retrieved from the corresponding repositories or model hubs.

#### DiffPath CelebA 32×32

Repository:

```text
https://github.com/clear-nus/diffpath
```

Checkpoint:

```bash
wget -O checkpoints/celeba_ema_0.9999_499999.pt \
  https://huggingface.co/ajrheng/diffpath/resolve/main/celeba_ema_0.9999_499999.pt
```

#### EigenScore models

Repository:

```text
https://github.com/wustl-cig/EigenScore
```

EigenScore provides model download links in its repository. Download the required pretrained model checkpoint (CelebA (edm)) from the source indicated there, then place it under:

```text
checkpoints/
```

Notes:

* CelebA checkpoints are included for reproducing the corresponding secondary/source-family experiments.
* The main MBE protocol should use the official backbones/checkpoints specified in `configs/mbe/`.
* We intentionally avoid redistributing third-party checkpoints in this repository.

---

## Quick Start

### 1. Main benchmark (MBE)

```bash
python main.py --config configs/mbe/cifar10.yaml --adapter improved --method cfs
```
or

```bash
python scripts/benchmark_images.py --config configs/mbe/cifar10.yaml --adapter improved --method cfs
```

Switch backbone:

```bash
python main.py --config configs/mbe/cifar10.yaml --adapter edm --method cfs
```

---

### 2. External experiments (CFS variants)

```bash
python main.py --config configs/external/cfs_cifar10_diag.yaml --method cfs
```

Variants available:

* `diag` → default (paper results)
* `knn` → alternative head (external analysis)
* `uvit` → transformer backbone

---


## Repository Structure

```
configs/
├── mbe/           # Main NeurIPS table (strict fairness)
├── external/      # CFS variants (not part of MBE)
├── debug/         # Fast experiments

checkpoints/

cfs/
├── methods/
├── models/
├── utils/
```

---

## Method: CFS

CFS extracts **low-noise canonical representations** from diffusion models.

Key ideas:

* evaluate at fixed logSNR levels (λ)
* sparse feature hooks (encoder / decoder)
* simple density modeling (diag / knn / gmm)

Variants:

* **CFS-1x2**: encoder + decoder
* **CFS-dec (1x1)**: decoder only (most efficient)


### CFS Variants (practical settings)

The main CFS variants can be controlled via:

- `region_mode`:
  - `both` → **CFS-1x2** (encoder + decoder)
  - `dec_only` → **CFS-dec (1x1)**

- `head`:
  - `diag` → default (used in main results)
  - `knn` → alternative density estimator (external experiments)

Example:

```yaml
cfs:
  explicit_lambdas: [5.0]
  region_mode: dec_only   # CFS-dec
  head: diag
```

---

## MBE Protocol (Core of the Paper)

MBE ensures that all methods are compared under identical conditions:

* same pretrained backbone
* same corruption schedule (λ_min → λ_max)
* same number of forward passes
* no retraining

This eliminates common biases in diffusion OOD evaluation.

---

## Key Config Parameters

### CFS

```yaml
cfs:
  explicit_lambdas: [5.0]
  hook_policy: sparse_ed_id
  region_mode: both
  head: diag  # diag | knn | gmm
```

### MSMA

```yaml
msma:
  head: gmm
  gmm_components: [2,4,6,8]
```

### DiffPath

```yaml
diffpath:
  variant: "6d"
```

---

## Datasets

* CIFAR10 / CIFAR100
* SVHN
* CelebA
* DTD
* ImageNet-1k / ImageNet-200
* OpenOOD (NINCO, Texture, etc.)

### ImageNet-scale

For ImageNet-scale and OpenOOD experiments, datasets are not downloaded automatically. Set `data_root` in the corresponding YAML files to your local OpenOOD data directory containing `benchmark_imglist/`, `images_largescale/`, and/or `images_classic/`.

### CelebA

CelebA is handled through `torchvision.datasets.CelebA` by default.  
Depending on the environment, automatic download may fail because TorchVision requires `gdown` for Google Drive downloads.

`gdown` is included in `requirements.txt`, but if needed you can install it manually:

```bash
pip install gdown
```

Alternatively, download CelebA manually and organize it under your data_root following the structure expected by TorchVision:

data_root/
└── celeba/
    ├── img_align_celeba/
    ├── list_attr_celeba.txt
    ├── identity_CelebA.txt
    ├── list_bbox_celeba.txt
    ├── list_landmarks_align_celeba.txt
    └── list_eval_partition.txt

---

## Metrics

* AUROC
* AUPR
* FPR95

---

## Compute Cost

We distinguish:

* **Logical cost (#F)** = number of forward passes
* **Runtime** = full execution time

| Method   | Cost  |
| -------- | ----- |
| CFS      | 1F    |
| GEPC     | 8F    |
| DiffPath | 10F   |
| DDPM-OOD | 364F |

---

## Reproducibility Checklist

* Official checkpoints only
* Fixed seeds
* No training required
* Deterministic loaders (optional)
* Full configs provided

---

## Notes

* `configs/mbe/` = **main paper results**
* `configs/external/` = **analysis / ablations**
* U-ViT is evaluated outside MBE (different architecture but same canonical levels)

---

## Citation

```bibtex
@article{cfs2026,
  title={Backbone-Equated Diffusion OOD via Sparse Internal Snapshots},
  author={Rouzoumka, Yadang Alexis and Pinsolle, Jean and Terreaux, Eug{\'e}nie and Morisseau, Christ{\`e}le and Ovarlez, Jean-Philippe and Ren, Chengfang},
  year={2026}
}
```

---

## Acknowledgements

* OpenAI (improved diffusion)
* NVIDIA (EDM)
* U-ViT authors
* OpenOOD benchmark
