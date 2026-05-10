# cfs/methods/msma.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cfs.methods._diffusion_common import (
    build_canonical_levels,
    clamp_x,
    corrupt_from_x0,
    estimate_eps_native,
    get_device,
    to_minus1_1,
)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
except Exception:
    GaussianMixture = None
    NearestNeighbors = None


# ============================================================
# Small heads
# ============================================================

class _Standardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("_Standardizer.fit expects shape (N, D)")
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1]), dtype=np.float64)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(np.isfinite(std) & (std > self.eps), std, 1.0)

        self.mean_ = mean.astype(np.float64, copy=True)
        self.std_ = std.astype(np.float64, copy=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        mean = self.mean_ if self.mean_ is not None else np.zeros((X.shape[1],), dtype=np.float64)
        std = self.std_ if self.std_ is not None else np.ones((X.shape[1],), dtype=np.float64)
        return ((X - mean) / (std + self.eps)).astype(np.float32)


class _DiagGaussianHead:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("_DiagGaussianHead.fit expects shape (N, D)")
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1]), dtype=np.float64)

        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        var = np.where(np.isfinite(var) & (var > 1e-8), var, 1.0)

        self.mu = mu.astype(np.float64, copy=True)
        self.var = var.astype(np.float64, copy=True)

    def score_ood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        mu = self.mu if self.mu is not None else np.zeros((X.shape[1],), dtype=np.float64)
        var = self.var if self.var is not None else np.ones((X.shape[1],), dtype=np.float64)

        z = (X - mu) ** 2 / var
        nll = 0.5 * np.sum(z + np.log(var), axis=1)
        return nll.astype(np.float32)


class _GMMHead:
    def __init__(
        self,
        n_components_grid: Tuple[int, ...],
        covariance_type: str = "full",
        max_fit_samples: int = 20000,
        random_state: int = 0,
    ):
        if GaussianMixture is None:
            raise RuntimeError("scikit-learn is required for head='gmm'.")

        self.n_components_grid = tuple(int(max(1, c)) for c in n_components_grid)
        self.covariance_type = str(covariance_type)
        self.max_fit_samples = int(max(1, max_fit_samples))
        self.random_state = int(random_state)

        self.model: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("_GMMHead.fit expects shape (N, D)")
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1]), dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        if X.shape[0] > self.max_fit_samples:
            idx = rng.choice(X.shape[0], size=self.max_fit_samples, replace=False)
            X_sel = X[idx]
        else:
            X_sel = X

        cand = [c for c in self.n_components_grid if c <= max(1, X_sel.shape[0])]
        if len(cand) == 0:
            cand = [1]

        best_bic = np.inf
        best_n = cand[0]

        for n in cand:
            try:
                g = GaussianMixture(
                    n_components=int(n),
                    covariance_type=self.covariance_type,
                    random_state=self.random_state,
                    reg_covar=1e-6,
                )
                g.fit(X_sel)
                bic = g.bic(X_sel)
                if np.isfinite(bic) and bic < best_bic:
                    best_bic = bic
                    best_n = int(n)
            except Exception:
                continue

        self.model = GaussianMixture(
            n_components=int(best_n),
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        self.model.fit(X)

    def score_ood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.model is None:
            return np.zeros((X.shape[0],), dtype=np.float32)
        ll = self.model.score_samples(X)   # inlier-high
        return (-ll).astype(np.float32)    # OOD-high


class _KNNHead:
    def __init__(self, k: int = 5):
        if NearestNeighbors is None:
            raise RuntimeError("scikit-learn is required for head='knn'.")
        self.k = int(max(1, k))
        self.nn: Optional[NearestNeighbors] = None
        self.k_eff: int = self.k

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("_KNNHead.fit expects shape (N, D)")
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1]), dtype=np.float64)

        self.k_eff = int(min(max(1, self.k), X.shape[0]))
        self.nn = NearestNeighbors(n_neighbors=self.k_eff, algorithm="auto")
        self.nn.fit(X)

    def score_ood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.nn is None:
            return np.zeros((X.shape[0],), dtype=np.float32)
        dists, _ = self.nn.kneighbors(X)
        return dists[:, self.k_eff - 1].astype(np.float32)


# ============================================================
# Config
# ============================================================

@dataclass
class MSMAConfig:
    lambda_min: float = -8.0
    lambda_max: float = 5.0

    # Kc = evaluated canonical levels = actual NFE
    Kc: int = 15
    # K_grid = denser candidate grid for mapping quality
    K_grid: int = 0

    # Head close to original MSMA spirit
    head: str = "gmm"   # "gmm" | "knn" | "diag"
    standardize: bool = True

    gmm_components: Tuple[int, ...] = (2, 4, 6, 8)
    gmm_covariance: str = "full"
    gmm_max_fit_samples: int = 20000
    random_state: int = 0

    knn_k: int = 5
    max_fit_batches: int = 128

    internal_bs: int = 64
    use_amp: bool = True

    explicit_corruption: bool = True
    shared_eps_across_levels: bool = True

    # Recommended for MSMA:
    # do not clamp after corruption, otherwise noisy levels get distorted.
    clamp: bool = False
    clamp_range: tuple[float, float] = (-1.0, 1.0)

    eps: float = 1e-6
    verbose: bool = False


# ============================================================
# Method
# ============================================================

class MSMAOOD:
    """
    MSMA harmonized under BCE / canonical-level protocol.

    Canonical feature:
        feat_k(x0) = || eps_hat_k ||_2

    with:
      - explicit corruption at canonical level k:
            x_k = a_k x0 + b_k eps
      - improved branch:
            eps_hat_k = eps_theta(x_k, t_k)    [native extraction]
      - EDM branch:
            x0_hat = denoise(x_k, sigma_k)
            eps_hat_k = (x_k - a_k x0_hat) / b_k

    Thus:
      - same semantic feature across backbone families,
      - same canonical levels,
      - same compute accounting through Kc.
    """

    def __init__(self, **kwargs):
        self.cfg = MSMAConfig(**kwargs)
        self.name = "msma"
        self.return_id_large = False

        self._levels: List[Dict[str, Any]] = []
        self._std = _Standardizer(eps=float(self.cfg.eps))
        self._head: Optional[object] = None

    @torch.no_grad()
    def _feature_batch(self, adapter: Any, x0: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not self._levels:
            raise RuntimeError("MSMAOOD: levels not initialized.")

        vals: List[torch.Tensor] = []

        shared_eps: Optional[torch.Tensor] = None
        if bool(cfg.explicit_corruption) and bool(cfg.shared_eps_across_levels):
            shared_eps = torch.randn_like(x0)

        for lvl in self._levels:
            if bool(cfg.explicit_corruption):
                eps = shared_eps if shared_eps is not None else torch.randn_like(x0)
                x_in = corrupt_from_x0(x0, lvl, eps=eps)
            else:
                x_in = x0

            x_in = clamp_x(x_in, bool(cfg.clamp), cfg.clamp_range)

            eps_hat = estimate_eps_native(
                adapter,
                x_in,
                lvl,
                internal_bs=int(cfg.internal_bs),
                use_amp=bool(cfg.use_amp),
            )

            feat = torch.linalg.vector_norm(eps_hat.flatten(start_dim=1), ord=2, dim=1)
            vals.append(feat.float())

        return torch.stack(vals, dim=1)  # (B, Kc)

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

        if bool(cfg.verbose):
            lams = [float(l["lambda"]) for l in self._levels]
            print(
                f"[msma] Kc={len(self._levels)} | K_grid={cfg.K_grid or cfg.Kc} "
                f"| lambda_eff in [{min(lams):.3f},{max(lams):.3f}] "
                f"| head={cfg.head} | standardize={cfg.standardize} "
                f"| explicit_corruption={cfg.explicit_corruption} "
                f"| shared_eps_across_levels={cfg.shared_eps_across_levels} "
                f"| clamp={cfg.clamp}"
            )

        maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0
        nb = 0
        buf: List[np.ndarray] = []

        for batch in tqdm(loader, desc="MSMA fit", leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))
            f = self._feature_batch(adapter, x0)
            buf.append(f.detach().cpu().numpy().astype(np.float32))
            nb += 1
            if maxb > 0 and nb >= maxb:
                break

        K = len(self._levels)
        X = np.concatenate(buf, axis=0) if len(buf) else np.zeros((0, K), dtype=np.float32)

        if bool(cfg.standardize):
            self._std.fit(X)
            Xh = self._std.transform(X)
        else:
            Xh = X.astype(np.float32, copy=False)

        head_name = str(cfg.head).lower()
        if head_name == "gmm":
            self._head = _GMMHead(
                n_components_grid=tuple(int(c) for c in cfg.gmm_components),
                covariance_type=str(cfg.gmm_covariance),
                max_fit_samples=int(cfg.gmm_max_fit_samples),
                random_state=int(cfg.random_state),
            )
        elif head_name == "knn":
            self._head = _KNNHead(k=int(cfg.knn_k))
        elif head_name == "diag":
            self._head = _DiagGaussianHead()
        else:
            raise ValueError(f"Unknown MSMA head={cfg.head!r}")

        self._head.fit(Xh)

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

        if self._head is None:
            self._head = _DiagGaussianHead()
            self._head.fit(np.zeros((1, len(self._levels)), dtype=np.float32))

        scores: List[np.ndarray] = []
        for batch in tqdm(loader, desc=f"MSMA score {tag}".strip(), leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))
            f = self._feature_batch(adapter, x0)
            X = f.detach().cpu().numpy().astype(np.float32)

            Xh = self._std.transform(X) if bool(cfg.standardize) else X
            s = self._head.score_ood(Xh)
            scores.append(s.astype(np.float32))

        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)
