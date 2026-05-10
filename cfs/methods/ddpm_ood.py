# cfs/methods/ddpm_ood.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cfs.methods._diffusion_common import (
    build_canonical_levels,
    clamp_x,
    corrupt_from_x0,
    estimate_x0_eps_native,
    get_device,
    to_minus1_1,
)


@dataclass
class DDPMOODConfig:
    lambda_min: float = -8.0
    lambda_max: float = 5.0

    # Number of canonical levels used to discretize the reverse path.
    # This is NOT the total NFE for DDPM-OOD.
    Kc: int = 100
    K_grid: int = 0

    # Keep every `start_skip`-th element from the noisy->clean reverse schedule.
    start_skip: int = 16

    mc_val: int = 1
    mc_test: int = 1

    level_norm: str = "zscore"    # "zscore" | "robust"
    robust_mad_scale: float = 1.4826

    agg_mode: str = "mean"        # "mean" | "median" | "sum"
    positive_only: bool = False

    max_fit_batches: int = 128

    internal_bs: int = 64
    use_amp: bool = True

    clamp: bool = False
    clamp_range: Tuple[float, float] = (-1.0, 1.0)

    eps: float = 1e-6
    verbose: bool = False


class DDPMOOD:
    """
    BCE-harmonized DDPM-OOD-style reconstruction detector.

    Main logic:
      1) build a canonical reverse schedule,
      2) select multiple start levels along the noisy->clean path,
      3) for each start, corrupt x0 at that level,
      4) reconstruct x0 deterministically through the suffix of the reverse schedule,
      5) compute one reconstruction MSE per start,
      6) z-score each start using ID-train statistics,
      7) aggregate across starts.

    This preserves the DDPM-OOD scoring logic while remaining compatible with
    the canonical-level BCE setup and both improved / EDM adapters.
    """

    def __init__(self, **kwargs):
        self.cfg = DDPMOODConfig(**kwargs)
        self.name = "ddpm_ood"
        self.return_id_large = False

        self._levels: List[Dict[str, Any]] = []          # clean -> noisy
        self._reverse_levels: List[Dict[str, Any]] = []  # noisy -> clean
        self._suffixes: List[List[Dict[str, Any]]] = []  # one suffix per start

        self._center: Optional[torch.Tensor] = None      # (S,)
        self._scale: Optional[torch.Tensor] = None       # (S,)

        self.total_nfe_: int = 0

    @staticmethod
    def _safe_center_scale(
        x: np.ndarray,
        mode: str,
        eps: float,
        robust_mad_scale: float,
    ) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0, 1.0

        mode = (mode or "zscore").lower()
        if mode == "robust":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            s = float(robust_mad_scale * mad)
            if (not np.isfinite(s)) or s <= eps:
                s = float(np.std(x))
                if (not np.isfinite(s)) or s <= eps:
                    s = 1.0
            return med, s

        c = float(np.mean(x))
        s = float(np.std(x))
        if (not np.isfinite(s)) or s <= eps:
            s = 1.0
        return c, s

    def _aggregate(self, z: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.positive_only):
            z = z.clamp_min(0.0)

        mode = (self.cfg.agg_mode or "mean").lower()
        if mode == "sum":
            return z.sum(dim=1)
        if mode == "median":
            zs = torch.sort(z, dim=1).values
            K = zs.shape[1]
            if K % 2 == 1:
                return zs[:, K // 2]
            return 0.5 * (zs[:, K // 2 - 1] + zs[:, K // 2])
        return z.mean(dim=1)

    def _init_schedule(self, adapter: Any) -> None:
        if self._levels:
            return

        cfg = self.cfg
        self._levels = build_canonical_levels(
            adapter,
            lambda_min=float(cfg.lambda_min),
            lambda_max=float(cfg.lambda_max),
            Kc=int(cfg.Kc),
            K_grid=(int(cfg.K_grid) if int(cfg.K_grid) > 0 else None),
            unique=True,
        )
        self._reverse_levels = list(reversed(self._levels))  # noisy -> clean

        skip = int(max(1, cfg.start_skip))
        start_pos = list(range(0, len(self._reverse_levels), skip))
        self._suffixes = [self._reverse_levels[p:] for p in start_pos]

        self.total_nfe_ = int(
            sum(len(suf) for suf in self._suffixes) * max(1, int(cfg.mc_test))
        )

        if bool(cfg.verbose):
            lams = [float(l["lambda"]) for l in self._levels]
            print(
                f"[ddpm_ood] Kc={len(self._levels)} | K_grid={cfg.K_grid or cfg.Kc} "
                f"| starts={len(self._suffixes)} | total_nfe(test)={self.total_nfe_} "
                f"| lambda_eff in [{min(lams):.3f},{max(lams):.3f}] "
                f"| agg_mode={cfg.agg_mode} | positive_only={cfg.positive_only} "
                f"| clamp={cfg.clamp}"
            )

    @torch.no_grad()
    def _reconstruct_from_suffix(
        self,
        adapter: Any,
        x_start: torch.Tensor,
        suffix: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        cfg = self.cfg
        x_cur = x_start

        for i, lvl in enumerate(suffix):
            x0_hat, eps_hat = estimate_x0_eps_native(
                adapter,
                x_cur,
                lvl,
                internal_bs=int(cfg.internal_bs),
                use_amp=bool(cfg.use_amp),
            )

            # next element in suffix = cleaner level
            if i + 1 < len(suffix):
                nxt = suffix[i + 1]
                a_next = float(nxt["a"])
                b_next = float(nxt["b"])
                x_cur = a_next * x0_hat + b_next * eps_hat
                x_cur = clamp_x(x_cur, bool(cfg.clamp), cfg.clamp_range)
            else:
                x_cur = x0_hat.float()

        return x_cur

    @torch.no_grad()
    def _score_matrix_batch(self, adapter: Any, x0: torch.Tensor, mc: int) -> torch.Tensor:
        """
        Returns raw reconstruction MSE matrix of shape (B, n_starts).
        """
        if not self._suffixes:
            raise RuntimeError("DDPMOOD: schedule not initialized.")

        per_start: List[torch.Tensor] = []

        for suffix in self._suffixes:
            lvl_start = suffix[0]
            vals_mc: List[torch.Tensor] = []

            for _ in range(int(max(1, mc))):
                eps = torch.randn_like(x0)
                x_start = corrupt_from_x0(x0, lvl_start, eps=eps)
                x_start = clamp_x(x_start, bool(self.cfg.clamp), self.cfg.clamp_range)

                x_rec = self._reconstruct_from_suffix(adapter, x_start, suffix)
                mse = ((x_rec - x0) ** 2).mean(dim=(1, 2, 3))
                vals_mc.append(mse.float())

            V = torch.stack(vals_mc, dim=1)  # (B, mc)
            if int(max(1, mc)) == 1:
                mse_s = V[:, 0]
            else:
                mse_s = V.median(dim=1).values if self.cfg.agg_mode == "median" else V.mean(dim=1)

            per_start.append(mse_s[:, None])

        return torch.cat(per_start, dim=1)  # (B, n_starts)

    @torch.no_grad()
    def fit_id_train(self, adapter: Any, loader: Iterable):
        cfg = self.cfg
        dev = get_device(adapter)

        self._init_schedule(adapter)

        maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0
        nb = 0
        rows: List[np.ndarray] = []

        for batch in tqdm(loader, desc="DDPM-OOD fit", leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))
            M = self._score_matrix_batch(adapter, x0, mc=int(cfg.mc_val))
            rows.append(M.detach().cpu().numpy().astype(np.float32))
            nb += 1
            if maxb > 0 and nb >= maxb:
                break

        S = len(self._suffixes)
        X = np.concatenate(rows, axis=0) if len(rows) else np.zeros((0, S), dtype=np.float32)

        centers = np.zeros((S,), dtype=np.float32)
        scales = np.ones((S,), dtype=np.float32)

        for s in range(S):
            c, sc = self._safe_center_scale(
                X[:, s] if X.shape[0] > 0 else np.zeros((0,), dtype=np.float32),
                mode=str(cfg.level_norm),
                eps=float(cfg.eps),
                robust_mad_scale=float(cfg.robust_mad_scale),
            )
            centers[s] = float(c)
            scales[s] = float(sc)

        self._center = torch.tensor(centers, dtype=torch.float32)
        self._scale = torch.tensor(scales, dtype=torch.float32)

    @torch.no_grad()
    def score_loader(self, adapter: Any, loader: Iterable, tag: str = "") -> np.ndarray:
        cfg = self.cfg
        dev = get_device(adapter)

        self._init_schedule(adapter)

        S = len(self._suffixes)
        if self._center is None or self._scale is None:
            self._center = torch.zeros((S,), dtype=torch.float32)
            self._scale = torch.ones((S,), dtype=torch.float32)

        center = self._center.to(device=dev, dtype=torch.float32).view(1, -1)
        scale = self._scale.to(device=dev, dtype=torch.float32).view(1, -1)

        scores: List[np.ndarray] = []
        for batch in tqdm(loader, desc=f"DDPM-OOD score {tag}".strip(), leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))

            M = self._score_matrix_batch(adapter, x0, mc=int(cfg.mc_test))
            z = (M - center) / (scale + float(cfg.eps))
            s = self._aggregate(z)

            scores.append(s.detach().cpu().numpy().astype(np.float32))

        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)