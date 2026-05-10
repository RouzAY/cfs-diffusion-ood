# cfs/methods/diffpath.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from cfs.methods._diffusion_common import (
    DiagGaussian,
    KDE1D,
    build_canonical_levels,
    clamp_x,
    corrupt_from_x0,
    estimate_x0_eps_native,
    get_device,
    sort_levels_clean_to_noisy,
    to_minus1_1,
)


@dataclass
class DiffPathConfig:
    lambda_min: float = -8.0
    lambda_max: float = 5.0

    # Kc = number of levels actually used = actual NFE
    Kc: int = 15
    # K_grid = candidate grid for better canonical mapping
    K_grid: int = 0

    mc_val: int = 1
    mc_test: int = 1
    max_fit_batches: int = 128

    variant: str = "1d"         # "1d" or "6d"
    path_stat: str = "sqmean"   # "sqmean" | "absmean" | "mean"
    kde_bandwidth: float = 0.0

    internal_bs: int = 64
    use_amp: bool = True

    # Recommended for path-based probes under explicit corruption:
    clamp: bool = False
    clamp_range: tuple[float, float] = (-1.0, 1.0)

    eps: float = 1e-6
    verbose: bool = False


class DiffPathOOD:
    """
    BCE-harmonized DiffPath-style baseline (recursive version).

    Path construction:
      - levels are canonical and shared across backbone families
      - start from the cleanest canonical level
      - at each level k:
            (x0_hat_k, eps_hat_k) = native_estimator(x_k, level_k)
            q_k = scalar(eps_hat_k)
            x_{k+1} = a_{k+1} x0_hat_k + b_{k+1} eps_hat_k

    Features:
      - variant="1d":
            sqrt( mean_k (dQ/dlambda)^2 )
      - variant="6d":
            moments of Q and dQ/dlambda

    This keeps the path semantics explicit while remaining backbone-equated.
    """

    def __init__(self, **kwargs):
        self.cfg = DiffPathConfig(**kwargs)
        self.name = "diffpath"
        self.return_id_large = False

        self._levels: List[Dict[str, Any]] = []
        self._head_1d: KDE1D | None = None
        self._head_6d: DiagGaussian | None = None

    # ============================================================
    # Helpers
    # ============================================================

    def _eps_to_scalar(self, eps_hat: torch.Tensor) -> torch.Tensor:
        mode = (self.cfg.path_stat or "sqmean").lower()
        if mode == "mean":
            return eps_hat.mean(dim=(1, 2, 3))
        if mode == "absmean":
            return eps_hat.abs().mean(dim=(1, 2, 3))
        return (eps_hat * eps_hat).mean(dim=(1, 2, 3))

    def _delta_lambda(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        Returns |lambda_{k+1} - lambda_k| in clean->noisy order.
        Shape: (K-1,)
        """
        if not self._levels:
            raise RuntimeError("DiffPathOOD: levels not initialized.")

        levels = sort_levels_clean_to_noisy(self._levels)
        lam = torch.tensor(
            [float(lvl["lambda"]) for lvl in levels],
            device=device,
            dtype=dtype,
        )
        if lam.numel() <= 1:
            return None
        return (lam[1:] - lam[:-1]).abs().clamp_min(1e-12)

    @torch.no_grad()
    def _path_scalar_sequence(self, adapter: Any, x0: torch.Tensor) -> torch.Tensor:
        """
        Recursive deterministic path, returns Q of shape (B, K).
        """
        cfg = self.cfg
        if not self._levels:
            raise RuntimeError("DiffPathOOD: levels not initialized.")

        levels = sort_levels_clean_to_noisy(self._levels)  # clean -> noisy
        K = len(levels)

        # Initialize at cleanest level with one base noise draw.
        lvl0 = levels[0]
        base_eps = torch.randn_like(x0)
        x_cur = corrupt_from_x0(x0, lvl0, eps=base_eps)
        x_cur = clamp_x(x_cur, bool(cfg.clamp), cfg.clamp_range)

        q_list: List[torch.Tensor] = []

        for k, lvl in enumerate(levels):
            x0_hat, eps_hat = estimate_x0_eps_native(
                adapter,
                x_cur,
                lvl,
                internal_bs=int(cfg.internal_bs),
                use_amp=bool(cfg.use_amp),
            )

            q_list.append(self._eps_to_scalar(eps_hat))

            # Recursive propagation to the next (noisier) canonical level.
            if k + 1 < K:
                nxt = levels[k + 1]
                a_next = float(nxt["a"])
                b_next = float(nxt["b"])
                x_cur = a_next * x0_hat + b_next * eps_hat
                x_cur = clamp_x(x_cur, bool(cfg.clamp), cfg.clamp_range)

        return torch.stack(q_list, dim=1)  # (B, K)

    # ============================================================
    # Features
    # ============================================================

    @torch.no_grad()
    def _feature_batch_1d(self, adapter: Any, x0: torch.Tensor, mc: int) -> torch.Tensor:
        """
        1D feature:
            sqrt( mean_k (dQ/dlambda)^2 )
        """
        acc = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.float32)
        dlam = self._delta_lambda(device=x0.device, dtype=torch.float32)

        for _ in range(int(max(1, mc))):
            Q = self._path_scalar_sequence(adapter, x0)
            if Q.shape[1] <= 1 or dlam is None:
                feat = Q[:, 0]
            else:
                dQ = (Q[:, 1:] - Q[:, :-1]) / dlam.view(1, -1)
                feat = torch.sqrt((dQ * dQ).mean(dim=1) + 1e-12)
            acc += feat.float()

        return acc / float(max(1, mc))

    @torch.no_grad()
    def _feature_batch_6d(self, adapter: Any, x0: torch.Tensor, mc: int) -> torch.Tensor:
        """
        6D feature family using moments of Q and dQ/dlambda.
        """
        acc = torch.zeros((x0.shape[0], 6), device=x0.device, dtype=torch.float32)
        dlam = self._delta_lambda(device=x0.device, dtype=torch.float32)

        for _ in range(int(max(1, mc))):
            Q = self._path_scalar_sequence(adapter, x0)

            f1 = Q.mean(dim=1)
            f2 = (Q * Q).mean(dim=1)
            f3 = torch.pow(Q.abs().pow(3).mean(dim=1) + 1e-12, 1.0 / 3.0)

            if Q.shape[1] <= 1 or dlam is None:
                z = torch.zeros_like(f1)
                f4, f5, f6 = z, z, z
            else:
                dQ = (Q[:, 1:] - Q[:, :-1]) / dlam.view(1, -1)
                f4 = dQ.abs().mean(dim=1)
                f5 = (dQ * dQ).mean(dim=1)
                f6 = torch.pow(dQ.abs().pow(3).mean(dim=1) + 1e-12, 1.0 / 3.0)

            feat = torch.stack([f1, f2, f3, f4, f5, f6], dim=1)
            acc += feat.float()

        return acc / float(max(1, mc))

    # ============================================================
    # Fit / score
    # ============================================================

    @torch.no_grad()
    def fit_id_train(self, adapter: Any, loader: Iterable):
        cfg = self.cfg
        dev = get_device(adapter)

        self._levels = build_canonical_levels(
            adapter,
            lambda_min=float(cfg.lambda_min),
            lambda_max=float(cfg.lambda_max),
            Kc=int(cfg.Kc),
            K_grid=(int(cfg.K_grid) if int(cfg.K_grid) > 0 else None),
            unique=True,
        )

        variant = str(cfg.variant).lower()
        if bool(cfg.verbose):
            lams = [float(l["lambda"]) for l in self._levels]
            print(
                f"[diffpath] variant={variant} | Kc={len(self._levels)} | K_grid={cfg.K_grid or cfg.Kc} "
                f"| lambda_eff in [{min(lams):.3f},{max(lams):.3f}] | path_stat={cfg.path_stat} "
                f"| recursive=True | dQ/dlambda=True | clamp={cfg.clamp}"
            )

        maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0
        nb = 0

        if variant == "1d":
            buf: List[np.ndarray] = []
            for batch in tqdm(loader, desc="DiffPath fit (1d)", leave=False):
                x0, _ = batch
                x0 = to_minus1_1(x0.to(dev, non_blocking=True))
                f = self._feature_batch_1d(adapter, x0, mc=int(cfg.mc_val))
                buf.append(f.detach().cpu().numpy().astype(np.float32))
                nb += 1
                if maxb > 0 and nb >= maxb:
                    break

            arr = np.concatenate(buf, axis=0) if len(buf) else np.zeros((0,), dtype=np.float32)
            head = KDE1D(bandwidth=float(cfg.kde_bandwidth))
            head.fit(arr)
            self._head_1d = head
            self._head_6d = None
            return

        buf6: List[np.ndarray] = []
        for batch in tqdm(loader, desc="DiffPath fit (6d)", leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))
            f6 = self._feature_batch_6d(adapter, x0, mc=int(cfg.mc_val))
            buf6.append(f6.detach().cpu().numpy().astype(np.float32))
            nb += 1
            if maxb > 0 and nb >= maxb:
                break

        X = np.concatenate(buf6, axis=0) if len(buf6) else np.zeros((0, 6), dtype=np.float32)
        head6 = DiagGaussian()
        head6.fit(X)
        self._head_6d = head6
        self._head_1d = None

    @torch.no_grad()
    def score_loader(self, adapter: Any, loader: Iterable, tag: str = "") -> np.ndarray:
        cfg = self.cfg
        dev = get_device(adapter)

        if not self._levels:
            self._levels = build_canonical_levels(
                adapter,
                lambda_min=float(cfg.lambda_min),
                lambda_max=float(cfg.lambda_max),
                Kc=int(cfg.Kc),
                K_grid=(int(cfg.K_grid) if int(cfg.K_grid) > 0 else None),
                unique=True,
            )

        variant = str(cfg.variant).lower()
        scores: List[np.ndarray] = []

        if variant == "1d":
            if self._head_1d is None:
                self._head_1d = KDE1D()

            for batch in tqdm(loader, desc=f"DiffPath score {tag}".strip(), leave=False):
                x0, _ = batch
                x0 = to_minus1_1(x0.to(dev, non_blocking=True))
                f = self._feature_batch_1d(adapter, x0, mc=int(cfg.mc_test))
                s = self._head_1d.score_ood(f.detach().cpu().numpy().astype(np.float32))
                scores.append(s.astype(np.float32))

        else:
            if self._head_6d is None:
                self._head_6d = DiagGaussian()

            for batch in tqdm(loader, desc=f"DiffPath score {tag}".strip(), leave=False):
                x0, _ = batch
                x0 = to_minus1_1(x0.to(dev, non_blocking=True))
                f6 = self._feature_batch_6d(adapter, x0, mc=int(cfg.mc_test))
                s = self._head_6d.score_ood(f6.detach().cpu().numpy().astype(np.float32))
                scores.append(s.astype(np.float32))

        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)