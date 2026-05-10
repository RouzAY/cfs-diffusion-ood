# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cfs.methods._diffusion_common import build_canonical_levels, clamp_x


def _to_x(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Empty batch.")
        if torch.is_tensor(batch[0]):
            return batch[0]
    if isinstance(batch, dict):
        for k in ("image", "images", "x", "data", "input"):
            v = batch.get(k)
            if torch.is_tensor(v):
                return v
    raise ValueError(f"Could not extract image tensor from batch type: {type(batch)}")


def _first_tensor(out: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if torch.is_tensor(x):
                return x
    if isinstance(out, dict):
        for x in out.values():
            if torch.is_tensor(x):
                return x
    return None


def _stacked_feature_matrix(z_by_level: List[List[torch.Tensor]]) -> torch.Tensor:
    parts = [z for z_list in z_by_level for z in z_list]
    if not parts:
        raise RuntimeError("No CFS features to stack.")
    return torch.cat(parts, dim=1)


class _DiagRunningStat:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64)
        self.sumsq = torch.zeros(self.dim, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, D], got {tuple(x.shape)}")
        xc = x.detach().to(dtype=torch.float64, device="cpu")
        self.n += int(xc.shape[0])
        self.sum += xc.sum(dim=0)
        self.sumsq += (xc * xc).sum(dim=0)

    def finalize(self, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n <= 0:
            raise RuntimeError("Cannot finalize empty running statistics.")
        mean = self.sum / float(self.n)
        var = self.sumsq / float(self.n) - mean * mean
        var = torch.clamp(var, min=float(eps))
        return mean.float(), var.float()


@dataclass
class _LevelSpec:
    idx: int
    lam: float
    kind: str
    t: Optional[int] = None
    sigma: Optional[float] = None
    lambda_target: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    sigma_tilde: Optional[float] = None


class CFSOOD:
    """
    Canonical Feature Snapshots for diffusion OOD detection.

    The detector probes sparse internal diffusion features at canonical noise
    levels and fits an ID-only head on pooled representations.

    Supported heads:
      - diag: diagonal Gaussian score, streaming fit;
      - knn: k-NN distance in CFS feature space;
      - shrinkage: Ledoit-Wolf Mahalanobis;
      - gmm_light: diagonal Gaussian mixture.
    """

    def __init__(
        self,
        lambda_min: float = -8.0,
        lambda_max: float = 5.0,
        Kc: int = 4,
        K_grid: int = 0,
        explicit_lambdas: Optional[Sequence[float]] = None,
        match_mode: str = "logsnr",
        stage_positions: Sequence[float] = (0.20, 0.50, 0.80),
        pool_std: bool = True,
        max_feat_dim: int = 0,
        mc_fit: int = 1,
        mc_test: int = 1,
        max_fit_batches: Optional[int] = 64,
        internal_bs: int = 64,
        eps: float = 1e-6,
        hook_policy: str = "sparse_ed_id",
        id_probe_batches: int = 3,
        max_region_candidates: int = 4,
        sparse_keep_same_res: bool = True,
        id_probe_images: int = 32,
        id_probe_max_chunks: int = 4,
        transformer_hook_region: str = "late",
        transformer_stage_positions: Sequence[float] = (0.20, 0.50, 0.80),
        num_special_tokens: int = -1,
        head: str = "diag",
        head_k: int = 10,
        head_n_components: int = 4,
        head_standardize: bool = True,
        head_bank_limit: int = 0,
        region_mode: str = "both",
        enc_keep: int = 1,
        dec_keep: int = 1,
        clamp: bool = False,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        verbose: bool = False,
    ):
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.Kc = int(Kc)
        self.K_grid = int(K_grid)

        self.explicit_lambdas = tuple(float(x) for x in explicit_lambdas) if explicit_lambdas else None

        self.match_mode = str(match_mode).lower()
        if self.match_mode not in {"logsnr", "uniform_t"}:
            raise ValueError(f"Unknown match_mode={match_mode!r}")

        self.stage_positions = tuple(min(0.95, max(0.05, float(p))) for p in stage_positions)
        self.transformer_stage_positions = tuple(
            min(0.95, max(0.05, float(p))) for p in transformer_stage_positions
        )

        self.transformer_hook_region = str(transformer_hook_region).lower()
        if self.transformer_hook_region == "middle":
            self.transformer_hook_region = "mid"
        if self.transformer_hook_region not in {"early", "mid", "late", "early_late", "all"}:
            raise ValueError(f"Unknown transformer_hook_region={transformer_hook_region!r}")

        self.num_special_tokens = int(num_special_tokens)
        self.pool_std = bool(pool_std)
        self.max_feat_dim = int(max_feat_dim)

        self.mc_fit = int(mc_fit)
        self.mc_test = int(mc_test)
        self.max_fit_batches = None if max_fit_batches is None else int(max_fit_batches)
        self.internal_bs = max(1, int(internal_bs))
        self.eps = float(eps)

        self.hook_policy = str(hook_policy).lower()
        if self.hook_policy not in {"quantile", "sparse_ed", "sparse_ed_id"}:
            raise ValueError(f"Unknown hook_policy={hook_policy!r}")

        self.id_probe_batches = max(1, int(id_probe_batches))
        self.max_region_candidates = max(1, int(max_region_candidates))
        self.sparse_keep_same_res = bool(sparse_keep_same_res)
        self.id_probe_images = max(8, int(id_probe_images))
        self.id_probe_max_chunks = max(2, int(id_probe_max_chunks))

        self.head = str(head).lower()
        if self.head not in {"diag", "knn", "shrinkage", "gmm_light"}:
            raise ValueError(f"Unknown head={head!r}")

        self.head_k = max(1, int(head_k))
        self.head_n_components = max(1, int(head_n_components))
        self.head_standardize = bool(head_standardize)
        self.head_bank_limit = max(0, int(head_bank_limit))

        self.region_mode = str(region_mode).lower()
        if self.region_mode not in {"both", "enc_only", "dec_only"}:
            raise ValueError(f"Unknown region_mode={region_mode!r}")

        self.enc_keep = max(0, int(enc_keep))
        self.dec_keep = max(0, int(dec_keep))

        if self.region_mode == "enc_only":
            self.dec_keep = 0
            self.enc_keep = max(1, self.enc_keep)
        elif self.region_mode == "dec_only":
            self.enc_keep = 0
            self.dec_keep = max(1, self.dec_keep)
        elif self.enc_keep <= 0 and self.dec_keep <= 0:
            raise ValueError("At least one of enc_keep/dec_keep must be positive.")

        self.clamp = bool(clamp)
        self.clamp_range = (float(clamp_range[0]), float(clamp_range[1]))
        self.verbose = bool(verbose)

        self.name = "cfs" if self.head == "diag" else f"cfs_{self.head}"

        self._fitted = False
        self._levels: List[_LevelSpec] = []
        self._hook_names: List[str] = []
        self._hook_shapes: List[Tuple[int, ...]] = []

        self._static_stats: List[_DiagRunningStat] = []
        self._static_mean: List[torch.Tensor] = []
        self._static_var: List[torch.Tensor] = []

        self._n_stages = 0
        self._device: Optional[torch.device] = None

        self._head_model = None
        self._head_mean_np: Optional[np.ndarray] = None
        self._head_std_np: Optional[np.ndarray] = None

    @torch.no_grad()
    def fit_id_train(self, adapter, loader) -> None:
        self._device = self._get_device(adapter)
        self._levels = self._build_levels(adapter)

        probe_chunks: List[torch.Tensor] = []
        probe_count = 0
        hooks_discovered = False

        for batch_idx, batch in enumerate(loader):
            if self.max_fit_batches is not None and batch_idx >= self.max_fit_batches:
                break

            x = _to_x(batch).to(self._device, non_blocking=True)

            for xb in x.split(self.internal_bs, dim=0):
                if xb.numel() == 0:
                    continue

                if not hooks_discovered and self.hook_policy == "sparse_ed_id":
                    if probe_count < self.id_probe_max_chunks:
                        keep = min(int(xb.shape[0]), self.id_probe_images)
                        if keep > 0:
                            probe_chunks.append(xb[:keep].detach())
                            probe_count += 1

                    total_probe = sum(int(t.shape[0]) for t in probe_chunks)
                    ready = total_probe >= self.id_probe_images or probe_count >= self.id_probe_max_chunks

                    if ready:
                        x_probe = torch.cat(probe_chunks, dim=0)[: self.id_probe_images]
                        self._discover_hooks(adapter, x_probe)
                        hooks_discovered = True

                elif len(self._hook_names) == 0:
                    self._discover_hooks(adapter, xb[:1])
                    hooks_discovered = True

                if len(self._hook_names) > 0:
                    self._fit_chunk(adapter, xb)

        if len(self._hook_names) == 0 and probe_chunks:
            x_probe = torch.cat(probe_chunks, dim=0)[: self.id_probe_images]
            self._discover_hooks(adapter, x_probe)

        if len(self._static_stats) == 0:
            raise RuntimeError("[cfs] No statistics collected during fit.")

        self._static_mean, self._static_var = [], []
        for st in self._static_stats:
            m, v = st.finalize(self.eps)
            self._static_mean.append(m.to(self._device))
            self._static_var.append(v.to(self._device))

        if self.head != "diag":
            self._fit_posthoc_head(adapter, loader)

        self._fitted = True

        if self.verbose:
            logging.info("[cfs] fit complete | levels=%d | hooks=%d", len(self._levels), len(self._hook_names))
            for i, (nm, shp) in enumerate(zip(self._hook_names, self._hook_shapes)):
                logging.info("[cfs] hook[%d] %s -> %s", i, nm, tuple(shp))

    @torch.no_grad()
    def score_loader(self, adapter, loader, tag: str = "") -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("[cfs] Call fit_id_train() before score_loader().")

        all_scores: List[np.ndarray] = []

        for batch in loader:
            x = _to_x(batch).to(self._device, non_blocking=True)
            chunk_scores: List[np.ndarray] = []

            for xb in x.split(self.internal_bs, dim=0):
                if xb.numel() == 0:
                    continue
                if self.head == "diag":
                    s = self._score_chunk_diag(adapter, xb)
                else:
                    s = self._score_chunk_posthoc(adapter, xb)
                chunk_scores.append(s)

            if chunk_scores:
                all_scores.append(np.concatenate(chunk_scores, axis=0))

        if not all_scores:
            return np.zeros((0,), dtype=np.float32)

        out = np.concatenate(all_scores, axis=0).astype(np.float32, copy=False)

        if self.verbose:
            logging.info("[cfs] score tag=%s | n=%d | mean=%.4f | std=%.4f",
                         tag, out.size, float(out.mean()), float(out.std()))
        return out

    def level_summary(self) -> List[Dict[str, Any]]:
        rows = []
        for lv in self._levels:
            row: Dict[str, Any] = {"idx": lv.idx, "lambda": lv.lam, "kind": lv.kind}
            for k in ("lambda_target", "t", "sigma", "a", "b", "sigma_tilde"):
                v = getattr(lv, k)
                if v is not None:
                    row[k] = int(v) if k == "t" else float(v)
            if lv.b is not None:
                row["b2"] = float(lv.b) ** 2
            rows.append(row)
        return rows

    def hook_summary(self) -> List[Dict[str, Any]]:
        return [{"name": n, "shape": tuple(s)} for n, s in zip(self._hook_names, self._hook_shapes)]

    def _prepare_x0(self, adapter, x: torch.Tensor) -> torch.Tensor:
        fn = getattr(adapter, "preprocess_x0", None)
        if callable(fn):
            return fn(x).to(device=self._device, dtype=torch.float32)
        return x.to(device=self._device, dtype=torch.float32)

    def _fit_chunk(self, adapter, x: torch.Tensor) -> None:
        x0 = self._prepare_x0(adapter, x)

        for _ in range(self.mc_fit):
            eps_path = torch.randn_like(x0)
            z_by_level = self._extract_path_features(adapter, x0, eps_path)

            if len(self._static_stats) == 0:
                self._init_stats_from_first_path(z_by_level)

            for k, z_list in enumerate(z_by_level):
                for l, z in enumerate(z_list):
                    self._static_stats[self._static_slot(k, l)].update(z)

    def _init_stats_from_first_path(self, z_by_level: List[List[torch.Tensor]]) -> None:
        self._n_stages = len(z_by_level[0])
        self._static_stats = []
        for z_list in z_by_level:
            for z in z_list:
                self._static_stats.append(_DiagRunningStat(int(z.shape[1])))

    def _score_chunk_diag(self, adapter, x: torch.Tensor) -> np.ndarray:
        x0 = self._prepare_x0(adapter, x)
        B = int(x0.shape[0])
        total = torch.zeros(B, device=x0.device, dtype=torch.float32)

        for _ in range(self.mc_test):
            eps_path = torch.randn_like(x0)
            z_by_level = self._extract_path_features(adapter, x0, eps_path)

            s = torch.zeros(B, device=x0.device, dtype=torch.float32)
            n = 0

            for k, z_list in enumerate(z_by_level):
                for l, z in enumerate(z_list):
                    idx = self._static_slot(k, l)
                    mu = self._static_mean[idx]
                    var = self._static_var[idx]
                    s += ((z - mu) ** 2 / var).mean(dim=1)
                    n += 1

            total += s / float(max(1, n))

        total /= float(max(1, self.mc_test))
        return total.detach().cpu().numpy()

    def _extract_static_matrix_chunk(self, adapter, x: torch.Tensor, mc: int) -> torch.Tensor:
        x0 = self._prepare_x0(adapter, x)
        acc = None

        for _ in range(max(1, int(mc))):
            eps_path = torch.randn_like(x0)
            z_by_level = self._extract_path_features(adapter, x0, eps_path)
            vec = _stacked_feature_matrix(z_by_level)
            acc = vec if acc is None else acc + vec

        return acc / float(max(1, int(mc)))

    @torch.no_grad()
    def _fit_posthoc_head(self, adapter, loader) -> None:
        rows = []
        total = 0

        for batch in loader:
            x = _to_x(batch).to(self._device, non_blocking=True)
            for xb in x.split(self.internal_bs, dim=0):
                if xb.numel() == 0:
                    continue
                z = self._extract_static_matrix_chunk(adapter, xb, mc=self.mc_fit)
                rows.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
                total += int(z.shape[0])

                if self.head_bank_limit > 0 and total >= self.head_bank_limit:
                    break

            if self.head_bank_limit > 0 and total >= self.head_bank_limit:
                break

        if not rows:
            raise RuntimeError("[cfs] Empty feature bank.")

        X_fit = np.concatenate(rows, axis=0)
        if self.head_bank_limit > 0:
            X_fit = X_fit[: self.head_bank_limit]

        if self.head_standardize:
            self._head_mean_np = X_fit.mean(axis=0).astype(np.float32)
            self._head_std_np = np.maximum(X_fit.std(axis=0), self.eps).astype(np.float32)
            X_fit = (X_fit - self._head_mean_np) / self._head_std_np

        if self.head == "knn":
            from sklearn.neighbors import NearestNeighbors
            k = min(self.head_k, len(X_fit))
            self._head_model = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_fit)

        elif self.head == "shrinkage":
            from sklearn.covariance import LedoitWolf
            self._head_model = LedoitWolf().fit(X_fit)

        elif self.head == "gmm_light":
            from sklearn.mixture import GaussianMixture
            n_components = min(self.head_n_components, len(X_fit))
            self._head_model = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                reg_covar=1e-6,
                random_state=0,
                max_iter=300,
            ).fit(X_fit)

        else:
            raise ValueError(f"Unsupported head={self.head!r}")

    def _score_chunk_posthoc(self, adapter, x: torch.Tensor) -> np.ndarray:
        if self._head_model is None:
            raise RuntimeError("[cfs] Post-hoc head is not fitted.")

        X = self._extract_static_matrix_chunk(adapter, x, mc=self.mc_test)
        X = X.detach().cpu().numpy().astype(np.float32, copy=False)

        if self.head_standardize:
            X = (X - self._head_mean_np) / self._head_std_np

        if self.head == "knn":
            dist, _ = self._head_model.kneighbors(X, return_distance=True)
            s = dist.mean(axis=1)

        elif self.head == "shrinkage":
            mu = self._head_model.location_
            prec = self._head_model.precision_
            d = X - mu[None, :]
            s = np.einsum("bi,ij,bj->b", d, prec, d)

        elif self.head == "gmm_light":
            s = -self._head_model.score_samples(X)

        else:
            raise ValueError(f"Unsupported head={self.head!r}")

        return s.astype(np.float32, copy=False)

    def _extract_path_features(self, adapter, x0: torch.Tensor, eps_path: torch.Tensor) -> List[List[torch.Tensor]]:
        out: List[List[torch.Tensor]] = []
        for lv in self._levels:
            xk, cond = self._make_noisy_level(adapter, x0, eps_path, lv)
            feats = self._capture_hooked_features(adapter, xk, cond)
            out.append([self._pool_feature(adapter, f) for f in feats])
        return out

    def _make_noisy_level(
        self,
        adapter,
        x0: torch.Tensor,
        eps_path: torch.Tensor,
        lv: _LevelSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lv.kind == "edm":
            sigma_vec, a, b = adapter.sigma_to_ab(float(lv.sigma), x0)
            xk = a * x0 + b * eps_path
            xk = clamp_x(xk, self.clamp, self.clamp_range)
            return xk, sigma_vec

        t = int(lv.t)
        a = adapter.sqrt_ab[t].view(1, 1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        b = adapter.sqrt_one_minus_ab[t].view(1, 1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        xk = clamp_x(a * x0 + b * eps_path, self.clamp, self.clamp_range)

        mk_cond = getattr(adapter, "make_condition", None)
        if callable(mk_cond):
            try:
                cond = mk_cond(t=t, batch_size=int(x0.shape[0]), device=x0.device)
            except TypeError:
                cond = mk_cond(t, int(x0.shape[0]), x0.device)
        else:
            cond = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)

        return xk, cond

    def _infer_num_special_tokens(self, adapter) -> int:
        if self.num_special_tokens >= 0:
            return self.num_special_tokens
        for obj in (getattr(adapter, "model", None), getattr(adapter, "raw_model", None)):
            if obj is None:
                continue
            for target in (obj, getattr(obj, "inner", None)):
                if target is not None and hasattr(target, "extras"):
                    try:
                        return int(getattr(target, "extras"))
                    except Exception:
                        pass
        return int(getattr(adapter, "num_special_tokens", 0))

    def _pool_feature(self, adapter, feat: torch.Tensor) -> torch.Tensor:
        feat = feat.float()

        if feat.ndim == 4:
            mu = feat.mean(dim=(2, 3))
            if self.pool_std:
                sd = torch.sqrt(torch.clamp(feat.var(dim=(2, 3), unbiased=False), min=self.eps))
                z = torch.cat([mu, sd], dim=1)
            else:
                z = mu

        elif feat.ndim == 3:
            n_special = self._infer_num_special_tokens(adapter)
            if n_special > 0 and feat.shape[1] > n_special:
                feat = feat[:, n_special:, :]

            mu = feat.mean(dim=1)
            if self.pool_std:
                sd = torch.sqrt(torch.clamp(feat.var(dim=1, unbiased=False), min=self.eps))
                z = torch.cat([mu, sd], dim=1)
            else:
                z = mu

        else:
            raise ValueError(f"[cfs] Expected 3D or 4D feature map, got {tuple(feat.shape)}")

        if self.max_feat_dim > 0 and z.shape[1] > self.max_feat_dim:
            z = z[:, : self.max_feat_dim]
        return z

    @torch.no_grad()
    def _discover_hooks(self, adapter, x_ref: torch.Tensor) -> None:
        x_ref_p = self._prepare_x0(adapter, x_ref)
        lv = self._levels[len(self._levels) // 2]
        eps_ref = torch.randn_like(x_ref_p)
        xk, cond = self._make_noisy_level(adapter, x_ref_p, eps_ref, lv)

        named_modules = list(adapter.model.named_modules())
        candidate_names = self._candidate_module_names(adapter, named_modules)
        candidates = self._probe_candidate_shapes(adapter, xk, cond, candidate_names)

        if not candidates:
            raise RuntimeError("[cfs] Hook discovery failed: no internal 3D/4D feature maps captured.")

        if bool(getattr(adapter, "is_transformer_diffusion", False)):
            selected = self._select_transformer_candidates(candidates)
        elif self.hook_policy == "quantile":
            selected = self._select_stage_candidates(candidates)
        elif self.hook_policy == "sparse_ed":
            selected = self._select_sparse_ed_candidates(adapter, candidates)
        else:
            selected = self._select_sparse_ed_id_candidates(adapter, x_ref_p, candidates)

        if not selected:
            raise RuntimeError("[cfs] Hook discovery selected zero hooks.")

        self._hook_names = [nm for nm, _ in selected]
        self._hook_shapes = [shp for _, shp in selected]

    def _candidate_module_names(self, adapter, named_modules):
        if bool(getattr(adapter, "is_transformer_diffusion", False)):
            block_regex = getattr(adapter, "block_regex", None)
            out = []

            if block_regex:
                rx = re.compile(str(block_regex))
                out = [name for name, _ in named_modules if name and rx.match(name)]

            if not out:
                patterns = [
                    r"^(inner\.)?in_blocks\.\d+$",
                    r"^(inner\.)?mid_block$",
                    r"^(inner\.)?out_blocks\.\d+$",
                    r"^(inner\.)?blocks\.\d+$",
                    r"^blocks\.\d+$",
                ]
                rxs = [re.compile(p) for p in patterns]
                out = [name for name, _ in named_modules if name and any(rx.match(name) for rx in rxs)]

            return out

        if self.hook_policy == "quantile":
            out = []
            for name, mod in named_modules:
                if name and not any(True for _ in mod.children()) and any(True for _ in mod.parameters(recurse=False)):
                    out.append(name)
            return out

        if getattr(adapter, "is_edm", False):
            rx = re.compile(r"^.*\.(enc|dec)\..*_block\d+$")
            return [name for name, _ in named_modules if name and rx.match(name)]

        rx_in = re.compile(r"^input_blocks\.\d+\.0$")
        rx_out = re.compile(r"^output_blocks\.\d+\.0$")
        return [name for name, _ in named_modules if name and (rx_in.match(name) or rx_out.match(name))]

    @torch.no_grad()
    def _probe_candidate_shapes(self, adapter, x, cond, candidate_names):
        model = adapter.model
        named = dict(model.named_modules())
        captured: Dict[str, Tuple[int, ...]] = {}
        handles = []

        def make_hook(name: str):
            def hook(_m, _inp, out):
                t = _first_tensor(out)
                if t is not None and t.ndim in (3, 4) and name not in captured:
                    captured[name] = tuple(int(s) for s in t.shape)
            return hook

        for name in candidate_names:
            mod = named.get(name)
            if mod is not None:
                handles.append(mod.register_forward_hook(make_hook(name)))

        try:
            self._forward_backbone(adapter, x, cond)
        finally:
            for h in handles:
                h.remove()

        return [(name, captured[name]) for name in candidate_names if name in captured]

    def _requested_region_keeps(self) -> Tuple[int, int]:
        if self.region_mode == "enc_only":
            return self.enc_keep, 0
        if self.region_mode == "dec_only":
            return 0, self.dec_keep
        return self.enc_keep, self.dec_keep

    def _split_regions(self, adapter, candidates):
        names = [nm for nm, _ in candidates]
        shapes = [shp for _, shp in candidates]

        if getattr(adapter, "is_edm", False):
            enc_idx = [i for i, nm in enumerate(names) if ".enc." in nm]
            dec_idx = [i for i, nm in enumerate(names) if ".dec." in nm]
        else:
            enc_idx = [i for i, nm in enumerate(names) if nm.startswith("input_blocks")]
            dec_idx = [i for i, nm in enumerate(names) if nm.startswith("output_blocks")]

        return enc_idx, dec_idx, shapes

    def _region_structural_order(self, adapter, candidates, region: str) -> List[int]:
        enc_idx, dec_idx, shapes = self._split_regions(adapter, candidates)
        idxs = enc_idx if region == "enc" else dec_idx
        if not idxs:
            return []

        res = lambda i: int(shapes[i][2])

        if region == "enc":
            preferred = [i for i in idxs if res(i) > min(res(j) for j in idxs)] or list(idxs)
            levels = sorted({res(i) for i in preferred})
        else:
            preferred = [i for i in idxs if res(i) < max(res(j) for j in idxs)] or list(idxs)
            levels = sorted({res(i) for i in preferred}, reverse=True)

        order, used = [], set()
        for rr in levels:
            same = [i for i in preferred if res(i) == rr]
            for i in reversed(same):
                if i not in used:
                    order.append(i)
                    used.add(i)

        for i in idxs:
            if i not in used:
                order.append(i)

        return order

    def _select_sparse_ed_candidates(self, adapter, candidates):
        enc_keep, dec_keep = self._requested_region_keeps()
        enc_order = self._region_structural_order(adapter, candidates, "enc")
        dec_order = self._region_structural_order(adapter, candidates, "dec")

        selected_idx = []
        selected_idx.extend(enc_order[:enc_keep])
        selected_idx.extend(dec_order[:dec_keep])

        if not selected_idx:
            return self._select_stage_candidates(candidates)

        return [candidates[i] for i in selected_idx]

    @torch.no_grad()
    def _select_sparse_ed_id_candidates(self, adapter, x_ref, candidates):
        enc_keep, dec_keep = self._requested_region_keeps()
        enc_order = self._region_structural_order(adapter, candidates, "enc")
        dec_order = self._region_structural_order(adapter, candidates, "dec")

        selected_idx = []

        if enc_keep > 0 and enc_order:
            short = enc_order[: max(self.max_region_candidates, enc_keep)]
            ranked = self._rank_candidates_by_id_proxy(adapter, x_ref, candidates, short)
            selected_idx.extend(ranked[:enc_keep])

        if dec_keep > 0 and dec_order:
            short = dec_order[: max(self.max_region_candidates, dec_keep)]
            ranked = self._rank_candidates_by_id_proxy(adapter, x_ref, candidates, short)
            selected_idx.extend(ranked[:dec_keep])

        if not selected_idx:
            return self._select_sparse_ed_candidates(adapter, candidates)

        return [candidates[i] for i in selected_idx]

    @torch.no_grad()
    def _rank_candidates_by_id_proxy(self, adapter, x_ref, candidates, idx_list):
        if len(idx_list) <= 1:
            return list(idx_list)

        names = [nm for nm, _ in candidates]
        named = dict(adapter.model.named_modules())

        def capture_one(name, x, cond):
            feats: Dict[str, torch.Tensor] = {}

            def hook(_m, _inp, out):
                t = _first_tensor(out)
                if t is not None and t.ndim in (3, 4):
                    feats["t"] = t

            h = named[name].register_forward_hook(hook)
            try:
                self._forward_backbone(adapter, x, cond)
            finally:
                h.remove()

            if "t" not in feats:
                raise RuntimeError(f"[cfs] Candidate hook did not fire: {name}")
            return self._pool_feature(adapter, feats["t"])

        scored = []

        for ci in idx_list:
            nm = names[ci]
            runs = []

            for _ in range(self.id_probe_batches):
                eps_path = torch.randn_like(x_ref)
                per_level = []
                for lv in self._levels:
                    xk, cond = self._make_noisy_level(adapter, x_ref, eps_path, lv)
                    per_level.append(capture_one(nm, xk, cond))
                runs.append(torch.stack(per_level, dim=0).mean(dim=0))

            Z = torch.stack(runs, dim=0)
            z_mean = Z.mean(dim=0)

            between = z_mean.var(dim=0, unbiased=False).mean() if z_mean.shape[0] > 1 else torch.tensor(0.0, device=x_ref.device)
            within = Z.var(dim=0, unbiased=False).mean() if Z.shape[0] > 1 else torch.tensor(0.0, device=x_ref.device)
            score = float((between / (within + self.eps)).item())
            scored.append((ci, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ci for ci, _ in scored]

    def _select_stage_candidates(self, candidates):
        n = len(candidates)
        used, selected = set(), []

        for p in self.stage_positions:
            idx = int(round(p * (n - 1)))
            for radius in range(n):
                probes = [idx] if radius == 0 else [idx - radius, idx + radius]
                found = False
                for j in probes:
                    if 0 <= j < n:
                        name, shp = candidates[j]
                        if name not in used:
                            selected.append((name, shp))
                            used.add(name)
                            found = True
                            break
                if found:
                    break

        return selected or [candidates[-1]]

    def _select_transformer_candidates(self, candidates):
        n = len(candidates)
        if n == 0:
            return []

        names = [nm for nm, _ in candidates]
        has_uvit = any("in_blocks" in nm or "out_blocks" in nm or "mid_block" in nm for nm in names)

        if has_uvit:
            early = [i for i, nm in enumerate(names) if "in_blocks" in nm]
            mid = [i for i, nm in enumerate(names) if "mid_block" in nm]
            late = [i for i, nm in enumerate(names) if "out_blocks" in nm]
            early_idx = early[len(early) // 2] if early else 0
            mid_idx = mid[0] if mid else n // 2
            late_idx = late[len(late) // 2] if late else n - 1
        else:
            ps = self.transformer_stage_positions
            early_idx = int(round(ps[0] * (n - 1)))
            mid_idx = int(round(ps[1] * (n - 1)))
            late_idx = int(round(ps[2] * (n - 1)))

        if self.transformer_hook_region == "early":
            idxs = [early_idx]
        elif self.transformer_hook_region == "mid":
            idxs = [mid_idx]
        elif self.transformer_hook_region == "late":
            idxs = [late_idx]
        elif self.transformer_hook_region == "early_late":
            idxs = [early_idx, late_idx]
        else:
            idxs = list(range(n))

        out, used = [], set()
        for i in idxs:
            i = max(0, min(n - 1, int(i)))
            if i not in used:
                out.append(candidates[i])
                used.add(i)
        return out

    def _capture_hooked_features(self, adapter, x, cond):
        model = adapter.model
        named = dict(model.named_modules())
        feats: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(_m, _inp, out):
                t = _first_tensor(out)
                if t is not None and t.ndim in (3, 4):
                    feats[name] = t
            return hook

        for name in self._hook_names:
            mod = named.get(name)
            if mod is None:
                raise RuntimeError(f"[cfs] Hook module not found: {name}")
            handles.append(mod.register_forward_hook(make_hook(name)))

        try:
            self._forward_backbone(adapter, x, cond)
        finally:
            for h in handles:
                h.remove()

        out = []
        for name in self._hook_names:
            if name not in feats:
                raise RuntimeError(f"[cfs] Selected hook produced no tensor: {name}")
            out.append(feats[name])

        return out

    def _forward_backbone(self, adapter, x, cond):
        if hasattr(adapter, "forward_model") and callable(adapter.forward_model):
            return adapter.forward_model(x, cond)

        if getattr(adapter, "is_edm", False):
            x = x.to(dtype=torch.float32)
            cond = cond.to(dtype=torch.float32)
            try:
                return adapter.model(x, cond, class_labels=None, force_fp32=True)
            except TypeError:
                return adapter.model(x, cond, class_labels=None)

        return adapter.model(x, cond)

    def _build_levels(self, adapter) -> List[_LevelSpec]:
        if self.Kc <= 0 and not self.explicit_lambdas:
            raise ValueError("Kc must be >= 1 unless explicit_lambdas is provided.")

        lvls = build_canonical_levels(
            adapter,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            Kc=self.Kc,
            K_grid=self.K_grid if self.K_grid > 0 else None,
            unique=True,
            explicit_lambdas=self.explicit_lambdas,
            match_mode=self.match_mode,
        )

        out: List[_LevelSpec] = []

        for d in lvls:
            if getattr(adapter, "is_edm", False):
                out.append(_LevelSpec(
                    idx=int(d["k"]),
                    lam=float(d["lambda"]),
                    lambda_target=float(d.get("lambda_target", d["lambda"])),
                    kind="edm",
                    sigma=float(d["sigma"]),
                    a=float(d.get("a", np.nan)),
                    b=float(d.get("b", np.nan)),
                    sigma_tilde=float(d.get("sigma_tilde", np.nan)),
                ))
            else:
                out.append(_LevelSpec(
                    idx=int(d["k"]),
                    lam=float(d["lambda"]),
                    lambda_target=float(d.get("lambda_target", d["lambda"])),
                    kind="improved",
                    t=int(d["t"]),
                    a=float(d.get("a", np.nan)),
                    b=float(d.get("b", np.nan)),
                    sigma_tilde=float(d.get("sigma_tilde", np.nan)),
                ))

        return out

    def _static_slot(self, k: int, l: int) -> int:
        return k * self._n_stages + l

    def _get_device(self, adapter) -> torch.device:
        dev = getattr(adapter, "device", None)
        if dev is not None:
            return dev if isinstance(dev, torch.device) else torch.device(str(dev))
        return next(adapter.model.parameters()).device