# dtd/datasets/images.py
from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from PIL import Image


# ============================================================
#  Small helpers
# ============================================================

def _interp(mode: str):
    from torchvision.transforms import InterpolationMode as I
    mode = str(mode).lower()
    return {
        "bilinear": I.BILINEAR,
        "nearest": I.NEAREST,
        "nearest_exact": I.NEAREST_EXACT,
        "bicubic": I.BICUBIC,
        "box": I.BOX,
        "hamming": I.HAMMING,
        "lanczos": I.LANCZOS,
    }[mode]


def _resize_pipeline(img_size, model_image_size=None, interpolation="bilinear"):
    """
    Resize -> ToTensor -> Normalize([-1,1]).
    Output shape: (C, model_image_size, model_image_size).
    """
    img_size = int(img_size)
    if model_image_size is None:
        model_image_size = img_size
    model_image_size = int(model_image_size)

    ops = []
    ops.append(transforms.Resize((img_size, img_size), interpolation=_interp(interpolation)))
    if model_image_size != img_size:
        ops.append(
            transforms.Resize((model_image_size, model_image_size), interpolation=_interp(interpolation))
        )

    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def _celeba_pipeline(img_size, model_image_size=None, interpolation="bilinear", resized32=False):
    img_size = int(img_size)
    if model_image_size is None:
        model_image_size = img_size
    model_image_size = int(model_image_size)

    ops = [transforms.CenterCrop(140)]
    if resized32:
        ops.append(transforms.Resize((32, 32), interpolation=_interp(interpolation)))

    ops.append(transforms.Resize((img_size, img_size), interpolation=_interp(interpolation)))
    if model_image_size != img_size:
        ops.append(
            transforms.Resize((model_image_size, model_image_size), interpolation=_interp(interpolation))
        )

    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def _apply_limit(ds, limit):
    if limit is None:
        return ds
    limit = int(min(int(limit), len(ds)))
    return Subset(ds, range(limit))


def _openood_warn(msg: str):
    print(f"[OpenOOD][warn] {msg}")


def _canonical_openood_name(name: str) -> str:
    """
    Canonicalise les noms côté benchmark_imglist.

    Important pour ton OpenOOD local :
      - benchmark_imglist/imagenet/
      - benchmark_imglist/imagenet200/
      - fichiers test_textures.txt (et non test_texture.txt)
    """
    name = str(name).lower()
    alias = {
        "imagenet1k": "imagenet",
        "imagenet-1k": "imagenet",
        "imagenet_1k": "imagenet",
        "imagenetv2": "imagenet_v2",
        "texture": "textures",
    }
    return alias.get(name, name)


def _extract_targets(ds) -> Optional[List[int]]:
    """
    Essaie d'extraire les labels pour split stratifié.
    """
    if isinstance(ds, OpenOODImgListDataset):
        return [int(y) for _, y in ds.items]

    if isinstance(ds, ImageFolder):
        return [int(y) for _, y in ds.samples]

    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            try:
                return [int(x) for x in arr]
            except Exception:
                pass

    return None


def _random_split_subset(ds, which="train", ratio=0.5, seed=0):
    n = len(ds)
    if n == 0:
        raise RuntimeError("Cannot split an empty dataset.")
    if not (0.0 < float(ratio) < 1.0):
        raise ValueError(f"ratio must be in (0,1), got {ratio}")

    idx = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(idx)

    cut = int(round(float(ratio) * n))
    cut = max(1, min(cut, n - 1))

    if which == "train":
        sel = idx[:cut]
    else:
        sel = idx[cut:]

    sel = sorted(sel.tolist())
    return Subset(ds, sel)


def _stratified_val_split(ds, which="train", ratio=0.5, seed=0):
    """
    Split déterministe sans chevauchement.
    - which='train' : première partie (bank / fit)
    - which='test'  : deuxième partie (ID eval)
    - stratifié par classe quand possible
    """
    which = str(which).lower()
    if which not in ("train", "test"):
        raise ValueError(f"which must be 'train' or 'test', got {which}")
    if not (0.0 < float(ratio) < 1.0):
        raise ValueError(f"ratio must be in (0,1), got {ratio}")

    labels = _extract_targets(ds)
    if labels is None or len(labels) != len(ds):
        return _random_split_subset(ds, which=which, ratio=ratio, seed=seed)

    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    rng = np.random.RandomState(int(seed))
    selected: List[int] = []

    for y in sorted(by_class.keys()):
        inds = np.asarray(by_class[y], dtype=np.int64)
        rng.shuffle(inds)

        if len(inds) == 1:
            if which == "train":
                selected.extend(inds.tolist())
            continue

        cut = int(round(float(ratio) * len(inds)))
        cut = max(1, min(cut, len(inds) - 1))

        if which == "train":
            selected.extend(inds[:cut].tolist())
        else:
            selected.extend(inds[cut:].tolist())

    if len(selected) == 0:
        return _random_split_subset(ds, which=which, ratio=ratio, seed=seed)

    selected = sorted(selected)
    return Subset(ds, selected)


# ============================================================
#  Toy dataset
# ============================================================

class GaussianToyDataset(Dataset):
    """
    Toy dataset : images ~ N(mu, sigma^2 I_d) in 3xHxW, already in [-1,1].
    """
    def __init__(self, mean: float, std: float, image_size: int, split: str = "train", length: int = 10000):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)
        self.image_size = int(image_size)
        self.length = int(length)

        base_seed = 12345 + int(1000 * self.mean)
        if split.startswith("train"):
            seed = base_seed
        elif split.startswith("val"):
            seed = base_seed + 1
        else:
            seed = base_seed + 2

        rng = np.random.RandomState(seed)
        C, H, W = 3, self.image_size, self.image_size
        data = rng.normal(loc=self.mean, scale=self.std, size=(self.length, C, H, W)).astype(np.float32)
        data = np.clip(data, -1.0, 1.0)

        self.data = torch.from_numpy(data)
        self.targets = torch.zeros(self.length, dtype=torch.long)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================================================
#  OpenOOD imglist dataset (robust path resolution)
# ============================================================

class OpenOODImgListDataset(Dataset):
    """
    Dataset reading OpenOOD benchmark_imglist/*.txt files.

    Each line typically:
        <relative_path> <label>
    Some lists may omit label -> label=0.

    We resolve paths by trying multiple roots (images_largescale, images_classic, etc.).
    A small cache is used to avoid repeated filesystem checks.
    """
    def __init__(self, imglist_path: str, img_roots: List[str], transform=None):
        super().__init__()
        self.imglist_path = os.path.abspath(imglist_path)
        self.img_roots = [os.path.abspath(r) for r in img_roots if r]
        self.transform = transform
        self.items: List[Tuple[str, int]] = []
        self._resolved_cache: Dict[str, str] = {}

        if not os.path.isfile(self.imglist_path):
            raise FileNotFoundError(f"[OpenOODImgListDataset] imglist not found: {self.imglist_path}")

        with open(self.imglist_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel = parts[0]
                y = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
                self.items.append((rel, y))

        if len(self.items) == 0:
            raise RuntimeError(f"[OpenOODImgListDataset] Empty imglist: {self.imglist_path}")

    def __len__(self):
        return len(self.items)

    def _try_candidates(self, rel: str) -> Tuple[Optional[str], List[str]]:
        rel_norm = rel.replace("\\", "/").lstrip("/")
        first = rel_norm.split("/", 1)[0] if "/" in rel_norm else rel_norm

        tried: List[str] = []

        for root in self.img_roots:
            c1 = os.path.join(root, rel_norm)
            tried.append(c1)
            if os.path.isfile(c1):
                return c1, tried

            if os.path.basename(root) == first and rel_norm.startswith(first + "/"):
                c2 = os.path.join(root, rel_norm[len(first) + 1 :])
                tried.append(c2)
                if os.path.isfile(c2):
                    return c2, tried

            for mid in ("images", "imgs", "data"):
                c3 = os.path.join(root, mid, rel_norm)
                tried.append(c3)
                if os.path.isfile(c3):
                    return c3, tried

                if os.path.basename(root) == first and rel_norm.startswith(first + "/"):
                    c4 = os.path.join(root, mid, rel_norm[len(first) + 1 :])
                    tried.append(c4)
                    if os.path.isfile(c4):
                        return c4, tried

        return None, tried

    def _resolve(self, rel: str) -> str:
        if rel in self._resolved_cache:
            return self._resolved_cache[rel]

        p, tried = self._try_candidates(rel)
        if p is None:
            show = tried[:10]
            msg = (
                f"Image not found for imglist item: {rel}\n"
                f"Tried (first {len(show)} candidates):\n  - " + "\n  - ".join(show)
            )
            raise FileNotFoundError(msg)

        self._resolved_cache[rel] = p
        return p

    def __getitem__(self, idx):
        rel, y = self.items[idx]
        path = self._resolve(rel)

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _openood_find_imglist(
    benchmark_root: str,
    split: str,
    name: str,
    benchmark_name: Optional[str] = None,
) -> Optional[str]:
    """
    Find a benchmark_imglist file.

    Priority:
      1) benchmark_imglist/<benchmark_name>/{split}_{name}.txt  (if benchmark_name provided)
      2) benchmark_imglist/<name>/{split}_{name}.txt
      3) recursive search under benchmark_imglist/<benchmark_name>/...
      4) recursive search under all benchmark_imglist/**
    """
    split = str(split).lower()
    name = _canonical_openood_name(name)
    benchmark_name = _canonical_openood_name(benchmark_name) if benchmark_name is not None else None

    tried = []

    if benchmark_name is not None:
        direct = os.path.join(benchmark_root, benchmark_name, f"{split}_{name}.txt")
        tried.append(direct)
        if os.path.isfile(direct):
            return direct

        patt = os.path.join(benchmark_root, benchmark_name, "**", f"{split}_{name}.txt")
        cands = sorted(glob.glob(patt, recursive=True))
        if len(cands) > 0:
            return cands[0]

    direct = os.path.join(benchmark_root, name, f"{split}_{name}.txt")
    tried.append(direct)
    if os.path.isfile(direct):
        return direct

    patt = os.path.join(benchmark_root, "**", f"{split}_{name}.txt")
    cands = sorted(glob.glob(patt, recursive=True))
    if len(cands) > 0:
        if benchmark_name is None and len(cands) > 1:
            _openood_warn(
                f"Multiple imglists found for split={split}, name={name}. "
                f"Using first match: {cands[0]}"
            )
        return cands[0]

    return None


# ============================================================
#  Main loader
# ============================================================

def load_data(
    name,
    data_dir,
    batch_size,
    image_size,
    train=False,
    split=None,
    limit=None,
    interpolation="bilinear",
    shuffle=True,
    num_workers=2,
    download=True,
    model_image_size=None,
    val_split_ratio=0.5,
    val_split_seed=0,
    benchmark_name=None,
):
    """
    Supports:
      - torchvision: cifar10/cifar100/svhn/dtd/celeba/places365
      - OpenOOD imglist: imagenet/imagenet200/imagenet_v2/imagenet_c/... etc (if benchmark_imglist exists)
      - manual ImageNet via ImageFolder (fallback)
      - gaussian toy: gauss_mu<value>

    Extra options:
      - split='val_train' / 'val_test':
          uses the val imglist/dataset and creates a deterministic disjoint split.
      - val_split_ratio:
          proportion allocated to the "train/bank" part.
      - val_split_seed:
          seed for the deterministic split.
      - benchmark_name:
          helpful for OOD txts that exist under both benchmark_imglist/imagenet/
          and benchmark_imglist/imagenet200/.
    """
    if model_image_size is None:
        model_image_size = image_size

    name = str(name).lower()
    if split is None:
        split = "train" if train else "test"
    split = str(split).lower()

    requested_split = split
    val_only_mode = None
    if split in ("val_train", "val_id_train", "val_bank"):
        split = "val"
        val_only_mode = "train"
    elif split in ("val_test", "val_id_test"):
        split = "val"
        val_only_mode = "test"

    data_dir = os.path.abspath(data_dir)

    # ------------------ CIFARs ------------------
    if name in ("cifar10", "cifar10_resized"):
        ds = datasets.CIFAR10(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            train=(split == "train"),
        )

    elif name in ("cifar100", "cifar100_resized"):
        ds = datasets.CIFAR100(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            train=(split == "train"),
        )

    # ------------------ SVHN ------------------
    elif name in ("svhn", "svhn_resized"):
        svhn_split = split if split in ("train", "test", "extra") else ("train" if train else "test")
        ds = datasets.SVHN(
            data_dir,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
            split=svhn_split,
        )

    # ------------------ Textures (DTD) ------------------
    elif name in ("dtd", "textures", "dtd_resized", "textures_resized"):
        dtd_split = split if split in ("train", "val", "test") else ("test" if not train else "train")
        ds = datasets.DTD(
            data_dir,
            split=dtd_split,
            partition=1,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
        )

    # ------------------ CelebA ------------------
    elif name in ("celeba", "celeba_resized"):
        resized32 = name.endswith("_resized")
        celeba_split = "train" if split == "train" else "test"
        ds = datasets.CelebA(
            data_dir,
            download=download,
            transform=_celeba_pipeline(image_size, model_image_size, interpolation, resized32=resized32),
            split=celeba_split,
        )

    # ------------------ Places365 ------------------
    elif name in ("places365", "places365_resized", "places"):
        places_split = split.lower()
        if places_split in ("train", "train-standard"):
            places_split = "train-standard"
        else:
            places_split = "val"
        ds = datasets.Places365(
            data_dir,
            split=places_split,
            small=True,
            download=download,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
        )

    # ------------------ SUN397 (manual) ------------------
    elif name in ("sun397", "sun", "sun397_resized"):
        root = os.path.join(data_dir, "SUN397")
        if not os.path.isdir(root):
            raise RuntimeError(
                f"[sun397] Dossier introuvable: {root}\n"
                "Télécharge SUN397 à la main et extrais dans ce dossier."
            )
        ds = ImageFolder(
            root,
            transform=_resize_pipeline(image_size, model_image_size, interpolation),
        )

    # ------------------ Gaussian toy ------------------
    elif name.startswith("gauss_mu"):
        try:
            mu_str = name.split("gauss_mu", 1)[1]
            mean = float(mu_str)
        except Exception as e:
            raise ValueError(
                f"[gaussian] name='{name}' invalide. Exemple: 'gauss_mu0.0' ou 'gauss_mu1.5'."
            ) from e

        std = 0.25
        length = 20000 if requested_split.startswith("train") else 5000
        H = int(model_image_size)
        ds = GaussianToyDataset(mean=mean, std=std, image_size=H, split=requested_split, length=length)

    else:
        # ============================================================
        # OpenOOD route (benchmark_imglist) if available
        # ============================================================
        bench_root = os.path.join(data_dir, "benchmark_imglist")
        use_openood = os.path.isdir(bench_root)

        if use_openood:
            name2 = _canonical_openood_name(name)

            benchmark_name2 = None
            if benchmark_name is not None:
                benchmark_name2 = _canonical_openood_name(benchmark_name)
            elif name2 in ("imagenet", "imagenet200"):
                benchmark_name2 = name2

            imglist = _openood_find_imglist(
                benchmark_root=bench_root,
                split=split,
                name=name2,
                benchmark_name=benchmark_name2,
            )

            # Fallback train -> val only for ImageNet-like families if no train images exist
            if imglist is not None and split == "train" and name2 in ("imagenet", "imagenet200"):
                img_roots = [
                    os.path.join(data_dir, "images_largescale"),
                    os.path.join(data_dir, "images_classic"),
                    os.path.join(data_dir, "images"),
                    data_dir,
                ]

                has_train = False
                for r in img_roots:
                    if os.path.isdir(os.path.join(r, "imagenet_1k", "train")) or os.path.isdir(
                        os.path.join(r, "imagenet", "train")
                    ):
                        has_train = True
                        break

                if not has_train:
                    alt = _openood_find_imglist(
                        benchmark_root=bench_root,
                        split="val",
                        name=name2,
                        benchmark_name=benchmark_name2,
                    )
                    if alt is not None:
                        _openood_warn(
                            f"Requested split=train for {name2}, but no ImageNet train folder detected under data/. "
                            f"Falling back to split=val using: {alt}"
                        )
                        imglist = alt

            if imglist is not None:
                img_roots = [
                    os.path.join(data_dir, "images_largescale"),
                    os.path.join(data_dir, "images_classic"),
                    os.path.join(data_dir, "images"),
                    data_dir,
                ]
                tfm = _resize_pipeline(image_size, model_image_size, interpolation)
                ds = OpenOODImgListDataset(imglist_path=imglist, img_roots=img_roots, transform=tfm)

            else:
                # ============================================================
                # Fallback: manual ImageNet via ImageFolder (val/val_raw)
                # ============================================================
                if name2 in ("imagenet", "imagenet200"):
                    base = os.path.join(data_dir, "imagenet")
                    if os.path.isdir(os.path.join(base, "val")):
                        root = os.path.join(base, "val")
                    elif os.path.isdir(os.path.join(base, "val_raw")):
                        root = os.path.join(base, "val_raw")
                    else:
                        raise RuntimeError(
                            f"[imagenet] Aucun dossier 'val' ou 'val_raw' trouvé sous {base}.\n"
                            "Vérifie ta structure: data/imagenet/val/<class>/*.JPEG "
                            "ou utilise OpenOOD benchmark_imglist+images_largescale."
                        )
                    ds = ImageFolder(
                        root,
                        transform=_resize_pipeline(image_size, model_image_size, interpolation),
                    )
                else:
                    raise ValueError(
                        f"Unknown dataset {name} (and no OpenOOD imglist found for split={split})."
                    )

        else:
            raise ValueError(f"Unknown dataset {name} (benchmark_imglist not found under {data_dir}).")

    # Apply disjoint val-only split before any limit
    if val_only_mode is not None:
        ds = _stratified_val_split(
            ds,
            which=val_only_mode,
            ratio=float(val_split_ratio),
            seed=int(val_split_seed),
        )

    ds = _apply_limit(ds, limit)

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=False,
    )