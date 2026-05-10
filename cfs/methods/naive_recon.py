# dtd/methods/naive_recon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cfs.methods._diffusion_common import (
    KDE1D,
    build_canonical_levels,
    corrupt_from_x0,
    estimate_x0,
    get_device,
    to_minus1_1,
)


def _to_x(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty batch.")
        if torch.is_tensor(batch[0]):
            return batch[0]
        raise ValueError("Could not extract image tensor from tuple/list batch.")
    if isinstance(batch, dict):
        for k in ("image", "images", "x", "data", "input"):
            v = batch.get(k, None)
            if torch.is_tensor(v):
                return v
        raise ValueError("Could not extract image tensor from dict batch.")
    raise ValueError(f"Unsupported batch type: {type(batch)}")


@dataclass
class NaiveReconConfig:
    lambda_min: float = -8.0
    lambda_max: float = 5.0

    # number of USED canonical levels
    Kc: int = 15

    # candidate grid resolution for discrete backbones
    K_grid: int = 200
    unique_levels: bool = True

    # optional explicit canonical lambdas, e.g. [5.0]
    explicit_lambdas: Optional[Sequence[float]] = None
    match_mode: str = "logsnr"

    # pick one level using ID-only reliability unless fixed_level_idx >= 0
    fixed_level_idx: int = -1
    mc_val: int = 2
    mc_test: int = 1

    kde_bandwidth: float = 0.0
    max_fit_batches: int = 128

    internal_bs: int = 64
    use_amp: bool = True

    clamp: bool = True
    clamp_range: Tuple[float, float] = (-1.0, 1.0)

    # "clean" gives ||x0_hat - x0||^2.
    # "noisy" gives ||x0_hat - x_lambda||^2, kept only for backward compatibility.
    target: str = "clean"

    eps: float = 1e-6
    verbose: bool = False


class NaiveRecon:
    """
    Output-space reconstruction baseline.

    For each image:
        x_lambda = a(lambda) x0 + b(lambda) eps
        x0_hat   = denoise_UViT_or_UNet(x_lambda, lambda)

    Main feature:
        feature_lambda(x0) = mean((x0_hat - x0)^2)

    Then a 1D ID-only KDE is fitted on this feature.
    Score is OOD-high = -log p_ID(feature).

    This is a same-backbone, same-lambda, same-cost output baseline for CFS.
    """

    def __init__(self, **kwargs):
        self.cfg = NaiveReconConfig(**kwargs)
        self.cfg.match_mode = str(self.cfg.match_mode).lower()
        if self.cfg.match_mode not in {"logsnr", "uniform_t"}:
            raise ValueError(f"Unknown match_mode={self.cfg.match_mode!r}")

        self.cfg.target = str(self.cfg.target).lower()
        if self.cfg.target not in {"clean", "noisy"}:
            raise ValueError(f"Unknown target={self.cfg.target!r}; use 'clean' or 'noisy'.")

        self.name = "naive_recon"
        self.return_id_large = False

        self._levels: List[Dict[str, Any]] = []
        self._chosen: Optional[int] = None
        self._head: Optional[KDE1D] = None

    @staticmethod
    def _reliability(mean_x: np.ndarray, var_within_x: np.ndarray, eps: float) -> float:
        vb = float(np.var(mean_x))
        vw = float(np.mean(var_within_x))
        return float(np.sqrt(max(vb, 0.0)) / (np.sqrt(max(vw, 0.0)) + float(eps)))

    def _fmt_level(self, lvl: Dict[str, Any]) -> str:
        lam = float(lvl.get("lambda", 0.0))
        lam_tgt = float(lvl.get("lambda_target", lam))
        b = float(lvl.get("b", np.nan))
        b2 = b * b if np.isfinite(b) else np.nan

        if "t" in lvl:
            return (
                f"k={lvl.get('k')} target={lam_tgt:.4f} "
                f"lambda={lam:.4f} t={int(lvl['t'])} b2={b2:.6g}"
            )
        return (
            f"k={lvl.get('k')} target={lam_tgt:.4f} "
            f"lambda={lam:.4f} sigma={float(lvl.get('sigma', 0.0)):.4g} b2={b2:.6g}"
        )

    def _build_levels(self, adapter: Any) -> List[Dict[str, Any]]:
        cfg = self.cfg

        lvls = build_canonical_levels(
            adapter,
            lambda_min=float(cfg.lambda_min),
            lambda_max=float(cfg.lambda_max),
            Kc=int(cfg.Kc),
            K_grid=int(cfg.K_grid),
            unique=bool(cfg.unique_levels),
            explicit_lambdas=cfg.explicit_lambdas,
            match_mode=str(cfg.match_mode),
        )

        lvls = sorted(lvls, key=lambda z: float(z.get("lambda", 0.0)), reverse=True)
        return lvls

    @torch.no_grad()
    def _feature_batch(self, adapter: Any, x0: torch.Tensor, lvl: Dict[str, Any], mc: int) -> torch.Tensor:
        cfg = self.cfg
        lo, hi = cfg.clamp_range
        acc = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.float32)

        for _ in range(int(max(1, mc))):
            eps = torch.randn_like(x0)
            x_lambda = corrupt_from_x0(x0, lvl, eps=eps)

            if cfg.clamp:
                x_lambda = x_lambda.clamp(lo, hi)

            x0_hat = estimate_x0(
                adapter,
                x_lambda,
                lvl,
                internal_bs=int(cfg.internal_bs),
                use_amp=bool(cfg.use_amp),
            )

            if cfg.target == "clean":
                feat = ((x0_hat - x0) ** 2).mean(dim=(1, 2, 3))
            else:
                feat = ((x0_hat - x_lambda) ** 2).mean(dim=(1, 2, 3))

            acc += feat.float()

        return acc / float(max(1, mc))

    @torch.no_grad()
    def fit_id_train(self, adapter: Any, loader: Iterable):
        cfg = self.cfg
        dev = get_device(adapter)

        self._levels = self._build_levels(adapter)

        if len(self._levels) == 0:
            raise RuntimeError("[naive_recon] No canonical levels built.")

        if bool(cfg.verbose):
            print(
                f"[naive_recon] levels: Kc={len(self._levels)} "
                f"| K_grid={cfg.K_grid} | unique_levels={cfg.unique_levels} "
                f"| explicit_lambdas={cfg.explicit_lambdas} | target={cfg.target}"
            )
            for i, lvl in enumerate(self._levels[: min(8, len(self._levels))]):
                print(f"[naive_recon] level {i}: {self._fmt_level(lvl)}")

        # --------------------------------------------------
        # Choose level
        # --------------------------------------------------
        if cfg.fixed_level_idx >= 0:
            self._chosen = int(min(max(0, int(cfg.fixed_level_idx)), len(self._levels) - 1))
        else:
            per_level_means: List[List[np.ndarray]] = [[] for _ in range(len(self._levels))]
            per_level_within: List[List[np.ndarray]] = [[] for _ in range(len(self._levels))]

            nb = 0
            maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0

            for batch in tqdm(loader, desc="NaiveRecon fit (select level)", leave=False):
                x0 = _to_x(batch)
                x0 = to_minus1_1(x0.to(dev, non_blocking=True))

                for li, lvl in enumerate(self._levels):
                    vals = []
                    for _ in range(int(max(1, cfg.mc_val))):
                        vals.append(self._feature_batch(adapter, x0, lvl, mc=1))

                    V = torch.stack(vals, dim=0)
                    mu = V.mean(dim=0)
                    var = V.var(dim=0, unbiased=False) if V.shape[0] >= 2 else torch.zeros_like(mu)

                    per_level_means[li].append(mu.detach().cpu().numpy().astype(np.float32))
                    per_level_within[li].append(var.detach().cpu().numpy().astype(np.float32))

                nb += 1
                if maxb > 0 and nb >= maxb:
                    break

            reliabilities = []
            for li in range(len(self._levels)):
                mx = (
                    np.concatenate(per_level_means[li], axis=0)
                    if len(per_level_means[li])
                    else np.zeros((0,), dtype=np.float32)
                )
                vx = (
                    np.concatenate(per_level_within[li], axis=0)
                    if len(per_level_within[li])
                    else np.zeros((0,), dtype=np.float32)
                )
                reliabilities.append(self._reliability(mx, vx, eps=float(cfg.eps)))

            self._chosen = int(np.argmax(np.asarray(reliabilities, dtype=np.float64)))

            if bool(cfg.verbose):
                print(
                    f"[naive_recon] chosen level idx={self._chosen} | "
                    f"{self._fmt_level(self._levels[self._chosen])} "
                    f"| reliability={float(np.max(reliabilities)):.6g}"
                )

        # --------------------------------------------------
        # Fit KDE on chosen feature
        # --------------------------------------------------
        chosen = int(self._chosen)
        head = KDE1D(bandwidth=float(cfg.kde_bandwidth))
        buf: List[np.ndarray] = []

        nb2 = 0
        maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0

        for batch in tqdm(loader, desc="NaiveRecon fit (kde)", leave=False):
            x0 = _to_x(batch)
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))

            f = self._feature_batch(adapter, x0, self._levels[chosen], mc=int(cfg.mc_test))
            buf.append(f.detach().cpu().numpy().astype(np.float32))

            nb2 += 1
            if maxb > 0 and nb2 >= maxb:
                break

        arr = np.concatenate(buf, axis=0) if len(buf) else np.zeros((0,), dtype=np.float32)
        head.fit(arr)
        self._head = head

    @torch.no_grad()
    def score_loader(self, adapter: Any, loader: Iterable, tag: str = "") -> np.ndarray:
        cfg = self.cfg
        dev = get_device(adapter)

        if not self._levels:
            self._levels = self._build_levels(adapter)

        if self._chosen is None:
            self._chosen = 0 if cfg.fixed_level_idx < 0 else int(cfg.fixed_level_idx)

        if self._head is None:
            self._head = KDE1D()

        scores: List[np.ndarray] = []
        lvl = self._levels[int(self._chosen)]

        for batch in tqdm(loader, desc=f"NaiveRecon score {tag}".strip(), leave=False):
            x0 = _to_x(batch)
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))

            f = self._feature_batch(adapter, x0, lvl, mc=int(cfg.mc_test))
            s = self._head.score_ood(f.detach().cpu().numpy().astype(np.float32))
            scores.append(s.astype(np.float32))

        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(scores, axis=0)

    def level_summary(self) -> List[Dict[str, Any]]:
        rows = []
        for lvl in self._levels:
            row = dict(lvl)
            for k, v in list(row.items()):
                if isinstance(v, np.generic):
                    row[k] = v.item()
            rows.append(row)
        return rows