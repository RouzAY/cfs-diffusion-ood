# External CFS configurations

This folder contains configurations used for **external positioning experiments** and **non-MBE settings**.

<!-- These experiments are not part of the main MBE (Mutualized Backbone-Equated) protocol.
The main MBE comparisons are defined in `configs/mbe/`.
 -->
---

## Purpose

This folder serves three purposes:

1. **CFS variants comparison**

   * Different heads (diag, knn)
   * Different region modes (dec-only vs both)
   * Different architectural setups

2. **Backbone sanity checks**

   * Example: U-ViT vs diffusion U-Net

3. **ImageNet-scale evaluation**

   * Typically using a single backbone (e.g., improved-diffusion)
   * Not strictly MBE-compliant

---

## Available configurations

### CIFAR-10 (reference setup)

* `cfs_cifar10_diag.yaml`
  → CFS with diagonal Gaussian head (default paper variant)

* `cfs_cifar10_knn.yaml`
  → CFS with k-NN head

* `cfs_cifar10_uvit_diag.yaml`
  → CFS on a U-ViT backbone (transformer sanity check)

---

### ImageNet-scale

* `cfs_imagenet1k_dec_diag.yaml`
  → Decoder-only CFS (recommended for stability at scale)

* `cfs_imagenet1k_both_knn.yaml`
  → Encoder+decoder with k-NN head (higher capacity, more expensive)

These configs typically rely on:

* improved-diffusion backbone
* 64×64 ImageNet checkpoints

---

## Using other ID datasets (SVHN / CelebA)

The CIFAR-10 configs can be reused directly by modifying only the `eval` section.

### SVHN as ID

```yaml
eval:
  id_train: { name: svhn, split: train, limit: null, download: true }
  id_test:  { name: svhn, split: test,  limit: null, download: true }
  ood:
    - { name: cifar10,  split: test, limit: null, download: true }
    - { name: cifar100, split: test, limit: null, download: true }
    - { name: celeba,   split: test, limit: null, download: true }
    - { name: dtd,      split: test, limit: null, download: true }
```

### CelebA as ID

```yaml
eval:
  id_train: { name: celeba, split: train, limit: null, download: true }
  id_test:  { name: celeba, split: test,  limit: null, download: true }
  ood:
    - { name: cifar10,  split: test, limit: null, download: true }
    - { name: svhn,     split: test, limit: null, download: true }
    - { name: cifar100, split: test, limit: null, download: true }
    - { name: dtd,      split: test, limit: null, download: true }
```

---

## Typical commands

```bash
# Diagonal head
python scripts/benchmark_images.py \
  --config configs/external/cfs_cifar10_diag.yaml \
  --method cfs

# k-NN head
python scripts/benchmark_images.py \
  --config configs/external/cfs_cifar10_knn.yaml \
  --method cfs

# U-ViT backbone
python scripts/benchmark_images.py \
  --config configs/external/cfs_cifar10_uvit_diag.yaml \
  --method cfs
```

---

## Notes

* The backbone can still be overridden with `--adapter` if multiple backbones are defined.
* These configs prioritize **clarity and reproducibility of variants**, not strict fairness.
* For paper tables, always refer to `configs/mbe/` for the canonical comparison protocol.
