# dtd/methods/gepc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cfs.methods._diffusion_common import (
    KDE1D,
    build_canonical_levels,
    clamp_x,
    corrupt_from_x0,
    estimate_eps_from_x0hat,
    estimate_x0,
    get_device,
    to_minus1_1,
)


# ============================================================
#  Group ops
# ============================================================

def _build_group_ops(
    H: int,
    W: int,
    group_set: str = "flip180",
    use_shifts: bool = False,
    shift_px: int = 1,
):
    """
    Returns a list of non-identity ops:
      [(g, ginv, name), ...]
    all acting on BCHW tensors.
    """
    ops = []

    def _h(x): return torch.flip(x, dims=[3])
    def _v(x): return torch.flip(x, dims=[2])
    def _rot90(x): return torch.rot90(x, k=1, dims=(2, 3))
    def _rot180(x): return torch.rot90(x, k=2, dims=(2, 3))
    def _rot270(x): return torch.rot90(x, k=3, dims=(2, 3))

    gset = str(group_set).lower()

    if gset in {"flip", "flip180", "full90"}:
        ops.append((_h, _h, "hflip"))
        ops.append((_v, _v, "vflip"))

    if gset in {"flip180", "full90"} and H == W:
        ops.append((_rot180, _rot180, "rot180"))

    if gset == "full90" and H == W:
        ops.append((_rot90, _rot270, "rot90"))
        ops.append((_rot270, _rot90, "rot270"))

    if use_shifts:
        s = int(shift_px)

        def _rollx(x): return torch.roll(x, shifts=s, dims=3)
        def _unrollx(x): return torch.roll(x, shifts=-s, dims=3)
        def _rolly(x): return torch.roll(x, shifts=s, dims=2)
        def _unrolly(x): return torch.roll(x, shifts=-s, dims=2)

        ops.append((_rollx, _unrollx, f"shiftx{s}"))
        ops.append((_rolly, _unrolly, f"shifty{s}"))

    if len(ops) == 0:
        ops.append((_h, _h, "hflip"))

    return ops


# ============================================================
#  Small helpers
# ============================================================

def _cosine_batch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    af = a.flatten(start_dim=1)
    bf = b.flatten(start_dim=1)
    na = torch.linalg.vector_norm(af, ord=2, dim=1).clamp_min(eps)
    nb = torch.linalg.vector_norm(bf, ord=2, dim=1).clamp_min(eps)
    dot = (af * bf).sum(dim=1)
    return dot / (na * nb)


def _pool_spatial(v: torch.Tensor, mode: str = "mean", rho: float = 0.10) -> torch.Tensor:
    """
    Input:
      - [B,C,H,W] or [B,H,W]
    Output:
      - [B]
    """
    if v.ndim == 4:
        v = v.mean(dim=1)  # [B,H,W]
    if v.ndim != 3:
        raise ValueError(f"_pool_spatial expects [B,C,H,W] or [B,H,W], got {tuple(v.shape)}")

    B = v.shape[0]
    z = v.reshape(B, -1)

    mode = str(mode).lower()
    if mode == "mean":
        return z.mean(dim=1)

    if mode == "topk":
        k = max(1, int(float(rho) * z.shape[1]))
        topk = torch.topk(z.abs(), k=k, dim=1).values
        return topk.mean(dim=1)

    raise ValueError(f"Unknown spatial_pool={mode!r}")


# ============================================================
#  Config
# ============================================================

@dataclass
class GEPCConfig:
    lambda_min: float = -10.0
    lambda_max: float = 5.0

    # Kc = evaluated levels (cost), K_grid = mapping quality
    Kc: int = 2
    K_grid: int = 0

    # Monte Carlo
    mc_fit: int = 1
    mc_test: int = 1
    shared_eps_across_levels: bool = True

    # features
    features: Tuple[str, ...] = ("gepc_s", "gepc_s_cos", "gepc_x0")
    metric_default: str = "gepc_s"
    agg_feat: str = "sum"       # "sum" | "mean" | "default"

    # across-level aggregation
    agg_t: str = "wmean"        # "mean" | "max" | "wmean" | "trimmean"
    trim_alpha: float = 0.10
    weight_t: str = "inv_cv"    # "inv_cv" | "uniform"
    keep_k: int = 2

    # calibration
    density_mode: str = "kde"   # "none" | "zscore" | "kde"
    bandwidth: float = 0.0

    # group
    group_set: str = "flip180"  # "flip" | "flip180" | "full90"
    group_shifts: bool = False
    shift_px: int = 1

    # pooling
    spatial_pool: str = "mean"  # "mean" | "topk"
    topk_rho: float = 0.10

    # compute
    max_fit_batches: int = 64
    internal_bs: int = 64
    use_amp: bool = True

    # input handling
    clamp: bool = True
    clamp_range: Tuple[float, float] = (-1.0, 1.0)

    eps: float = 1e-6
    verbose: bool = False


# ============================================================
#  Method
# ============================================================

class GEPC:
    """
    BCE-harmonized GEPC:
      - canonical logSNR levels
      - explicit corruption x_k = a_k x0 + b_k eps
      - adapter-agnostic x0_hat
      - score proxy from eps_hat = (x_k - a_k x0_hat)/b_k
      - group consistency on output space
      - ID-only calibration
    Returned score is always OOD-high.
    """

    def __init__(self, **kwargs):
        self.cfg = GEPCConfig(**kwargs)
        self.name = "gepc"
        self.return_id_large = False

        self._levels: List[Dict[str, Any]] = []

        # calibration
        self._mu: Dict[Tuple[int, str], float] = {}
        self._sig: Dict[Tuple[int, str], float] = {}
        self._kde: Dict[Tuple[int, str], KDE1D] = {}

        # level selection / weighting
        self._w_t: Dict[int, float] = {}
        self._t_kept: Optional[set[int]] = None

    # --------------------------------------------------------
    # raw GEPC features at one level
    # --------------------------------------------------------

    @torch.no_grad()
    def _features_at_level(
        self,
        adapter: Any,
        x0: torch.Tensor,
        lvl: Dict[str, Any],
        eps_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        if eps_noise is None:
            eps_noise = torch.randn_like(x0)

        xk = corrupt_from_x0(x0, lvl, eps=eps_noise)
        xk = clamp_x(xk, bool(cfg.clamp), cfg.clamp_range)

        # base prediction
        x0_hat = estimate_x0(
            adapter,
            xk,
            lvl,
            internal_bs=int(cfg.internal_bs),
            use_amp=bool(cfg.use_amp),
        )
        eps_hat = estimate_eps_from_x0hat(xk, x0_hat, lvl, eps=float(cfg.eps))
        b = float(lvl["b"])
        score_ref = -eps_hat / (b + float(cfg.eps))

        B, C, H, W = xk.shape
        ops = _build_group_ops(
            H=H,
            W=W,
            group_set=str(cfg.group_set),
            use_shifts=bool(cfg.group_shifts),
            shift_px=int(cfg.shift_px),
        )

        # transformed noisy samples
        xg = torch.cat([g(xk) for (g, _ginv, _nm) in ops], dim=0)

        x0_hat_g = estimate_x0(
            adapter,
            xg,
            lvl,
            internal_bs=int(cfg.internal_bs),
            use_amp=bool(cfg.use_amp),
        )
        eps_hat_g = estimate_eps_from_x0hat(xg, x0_hat_g, lvl, eps=float(cfg.eps))
        score_g = -eps_hat_g / (b + float(cfg.eps))

        # back to canonical frame
        score_back_list: List[torch.Tensor] = []
        x0_back_list: List[torch.Tensor] = []

        for i, (_g, ginv, _nm) in enumerate(ops):
            sl = slice(i * B, (i + 1) * B)
            score_back_list.append(ginv(score_g[sl]))
            x0_back_list.append(ginv(x0_hat_g[sl]))

        score_back = torch.stack(score_back_list, dim=0)   # [G,B,C,H,W]
        x0_back = torch.stack(x0_back_list, dim=0)         # [G,B,C,H,W]

        denom_s = _pool_spatial(score_ref.square(), cfg.spatial_pool, cfg.topk_rho).clamp_min(float(cfg.eps))
        denom_x0 = _pool_spatial(x0_hat.square(), cfg.spatial_pool, cfg.topk_rho).clamp_min(float(cfg.eps))

        out: Dict[str, torch.Tensor] = {}

        if "gepc_s" in cfg.features:
            vals = []
            for i in range(score_back.shape[0]):
                diff_map = (score_back[i] - score_ref).square()
                vals.append(_pool_spatial(diff_map, cfg.spatial_pool, cfg.topk_rho) / denom_s)
            out["gepc_s"] = torch.stack(vals, dim=0).mean(dim=0)

        if "gepc_s_cos" in cfg.features:
            vals = []
            for i in range(score_back.shape[0]):
                vals.append(1.0 - _cosine_batch(score_back[i], score_ref, eps=float(cfg.eps)))
            out["gepc_s_cos"] = torch.stack(vals, dim=0).mean(dim=0)

        if "gepc_x0" in cfg.features:
            vals = []
            for i in range(x0_back.shape[0]):
                diff_map = (x0_back[i] - x0_hat).square()
                vals.append(_pool_spatial(diff_map, cfg.spatial_pool, cfg.topk_rho) / denom_x0)
            out["gepc_x0"] = torch.stack(vals, dim=0).mean(dim=0)

        if "cycle" in cfg.features:
            # same-level reconstruction cycle with the same eps realization
            xk_rt = float(lvl["a"]) * x0_hat + float(lvl["b"]) * eps_noise
            xk_rt = clamp_x(xk_rt, bool(cfg.clamp), cfg.clamp_range)
            cyc = _pool_spatial((xk_rt - xk).square(), cfg.spatial_pool, cfg.topk_rho)
            denom_xt = _pool_spatial(xk.square(), cfg.spatial_pool, cfg.topk_rho).clamp_min(float(cfg.eps))
            out["cycle"] = cyc / denom_xt

        return out

    @torch.no_grad()
    def _features_all_levels(
        self,
        adapter: Any,
        x0: torch.Tensor,
        mc: int,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Returns:
          slot_id -> { feature_name -> [B] }
        averaged across MC repeats.
        """
        cfg = self.cfg
        acc: Dict[int, Dict[str, torch.Tensor]] = {}

        for _ in range(int(max(1, mc))):
            shared_eps = torch.randn_like(x0) if bool(cfg.shared_eps_across_levels) else None

            for lvl in self._levels:
                slot = int(lvl["k"])
                eps_use = shared_eps if shared_eps is not None else torch.randn_like(x0)
                feats = self._features_at_level(adapter, x0, lvl, eps_noise=eps_use)

                if slot not in acc:
                    acc[slot] = {k: v.clone() for k, v in feats.items()}
                else:
                    for k in feats:
                        acc[slot][k] = acc[slot][k] + feats[k]

        scale = float(max(1, mc))
        for slot in acc:
            for k in acc[slot]:
                acc[slot][k] = acc[slot][k] / scale

        return acc

    # --------------------------------------------------------
    # weighting / level-keep
    # --------------------------------------------------------

    def _compute_t_weights_from_id(self, base_buf: Dict[int, torch.Tensor]) -> None:
        cfg = self.cfg
        keys = sorted(base_buf.keys())

        if str(cfg.weight_t).lower() == "uniform":
            w = {k: 1.0 for k in keys}
        else:
            w = {}
            for k in keys:
                v = base_buf[k].detach().float().view(-1)
                if v.numel() == 0:
                    w[k] = 1.0
                    continue
                mu = float(v.mean())
                sd = float(v.std().clamp_min(1e-6))
                cv = sd / (abs(mu) + 1e-6)
                w[k] = 1.0 / (cv + 1e-6)

        s = sum(float(v) for v in w.values()) + 1e-12
        self._w_t = {k: float(w[k] / s) for k in keys}

    def _stable_t_mask(self, base_buf: Dict[int, torch.Tensor], keep_k: int) -> set[int]:
        stats = []
        for k, v in base_buf.items():
            vv = v.detach().float().view(-1)
            if vv.numel() == 0:
                stats.append((k, float("inf")))
                continue
            mu = float(vv.mean())
            sd = float(vv.std().clamp_min(1e-8))
            cv = sd / (abs(mu) + 1e-6)
            stats.append((k, cv))

        stats.sort(key=lambda z: z[1])
        kept = [k for (k, _cv) in stats[:min(int(keep_k), len(stats))]]
        return set(kept)

    # --------------------------------------------------------
    # calibration
    # --------------------------------------------------------

    def _fit_calibration(self, buf: Dict[Tuple[int, str], List[torch.Tensor]]) -> None:
        cfg = self.cfg
        mode = str(cfg.density_mode).lower()

        self._mu.clear()
        self._sig.clear()
        self._kde.clear()

        if mode == "none":
            return

        for key, lst in buf.items():
            data = torch.cat(lst, dim=0) if len(lst) else torch.zeros((1,), dtype=torch.float32)
            arr = data.detach().cpu().numpy().astype(np.float32)

            if mode == "kde":
                kde = KDE1D(bandwidth=float(cfg.bandwidth))
                kde.fit(arr)
                self._kde[key] = kde
            elif mode == "zscore":
                mu = float(np.mean(arr)) if arr.size > 0 else 0.0
                sd = float(np.std(arr)) if arr.size > 0 else 1.0
                if (not np.isfinite(sd)) or sd <= float(cfg.eps):
                    sd = 1.0
                self._mu[key] = mu
                self._sig[key] = sd
            else:
                raise ValueError(f"Unknown density_mode={cfg.density_mode!r}")

    def _calibrate_feature(self, slot: int, feat_name: str, raw: torch.Tensor) -> torch.Tensor:
        """
        Returns an OOD-high calibrated score.
        """
        cfg = self.cfg
        mode = str(cfg.density_mode).lower()

        if mode == "none":
            return raw

        if mode == "zscore":
            mu = float(self._mu[(slot, feat_name)])
            sd = float(self._sig[(slot, feat_name)])
            z = (raw - mu) / (sd + float(cfg.eps))
            return 0.5 * (z * z)

        if mode == "kde":
            x = raw.detach().cpu().numpy().astype(np.float32)
            s = self._kde[(slot, feat_name)].score_ood(x)  # already OOD-high
            return torch.from_numpy(s).to(device=raw.device, dtype=torch.float32)

        raise ValueError(f"Unknown density_mode={cfg.density_mode!r}")

    # --------------------------------------------------------
    # aggregation
    # --------------------------------------------------------

    def _agg_features_at_t(self, slot: int, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg

        vals = []
        for f in cfg.features:
            if f not in feats:
                continue
            vals.append(self._calibrate_feature(slot, f, feats[f]))

        if len(vals) == 0:
            raise RuntimeError(f"[gepc] no available feature among {cfg.features}")

        mode = str(cfg.agg_feat).lower()
        if mode == "mean":
            return torch.stack(vals, dim=0).mean(dim=0)
        if mode == "sum":
            return torch.stack(vals, dim=0).sum(dim=0)

        # default => chosen metric only
        f0 = str(cfg.metric_default)
        if f0 not in feats:
            f0 = next(iter(feats.keys()))
        return self._calibrate_feature(slot, f0, feats[f0])

    def _agg_across_t(self, per_t_scores: torch.Tensor, t_used: List[int]) -> torch.Tensor:
        """
        per_t_scores: [T,B], OOD-high
        """
        cfg = self.cfg
        T, B = per_t_scores.shape
        if T == 0:
            return torch.zeros((B,), device=per_t_scores.device, dtype=torch.float32)

        mode = str(cfg.agg_t).lower()

        if mode == "max":
            return per_t_scores.max(dim=0).values

        if mode == "wmean":
            w = torch.tensor(
                [float(self._w_t.get(t, 1.0 / max(1, T))) for t in t_used],
                device=per_t_scores.device,
                dtype=torch.float32,
            ).view(T, 1)
            w = w / (w.sum() + 1e-12)
            return (w * per_t_scores).sum(dim=0)

        if mode == "trimmean":
            alpha = float(cfg.trim_alpha)
            k = max(0, int(alpha * T / 2.0))
            S = torch.sort(per_t_scores, dim=0).values
            if k > 0 and (2 * k) < T:
                S = S[k:T - k, :]
            return S.mean(dim=0)

        return per_t_scores.mean(dim=0)

    # --------------------------------------------------------
    # fit / score
    # --------------------------------------------------------

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
                f"[gepc] Kc={len(self._levels)} | K_grid={cfg.K_grid or cfg.Kc} "
                f"| lambda_eff in [{min(lams):.3f}, {max(lams):.3f}] "
                f"| features={cfg.features} | density_mode={cfg.density_mode} "
                f"| group_set={cfg.group_set} | spatial_pool={cfg.spatial_pool}"
            )

        buf: Dict[Tuple[int, str], List[torch.Tensor]] = {
            (int(lvl["k"]), f): [] for lvl in self._levels for f in cfg.features
        }

        base_feature = str(cfg.metric_default)
        if base_feature not in cfg.features:
            base_feature = cfg.features[0]

        maxb = int(cfg.max_fit_batches) if cfg.max_fit_batches else 0
        nb = 0

        for batch in tqdm(loader, desc="GEPC fit", leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))

            feats_all = self._features_all_levels(adapter, x0, mc=int(cfg.mc_fit))
            for slot, feats in feats_all.items():
                for f in cfg.features:
                    if f in feats:
                        buf[(slot, f)].append(feats[f].detach().cpu().float())

            nb += 1
            if maxb > 0 and nb >= maxb:
                break

        base_buf = {}
        for lvl in self._levels:
            slot = int(lvl["k"])
            lst = buf.get((slot, base_feature), [])
            base_buf[slot] = torch.cat(lst, dim=0) if len(lst) else torch.zeros((1,), dtype=torch.float32)

        self._compute_t_weights_from_id(base_buf)
        self._t_kept = self._stable_t_mask(base_buf, keep_k=int(cfg.keep_k))
        self._fit_calibration(buf)

        if bool(cfg.verbose):
            kept = sorted(list(self._t_kept)) if self._t_kept is not None else [int(l["k"]) for l in self._levels]
            print(f"[gepc] kept levels={kept}")
            print(f"[gepc] weights={{{', '.join([f'{k}: {self._w_t[k]:.3f}' for k in sorted(self._w_t)])}}}")

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

        if self._t_kept is None:
            self._t_kept = set(int(lvl["k"]) for lvl in self._levels)
        if len(self._w_t) == 0:
            n = max(1, len(self._levels))
            self._w_t = {int(lvl["k"]): 1.0 / n for lvl in self._levels}

        scores: List[np.ndarray] = []

        for batch in tqdm(loader, desc=f"GEPC score {tag}".strip(), leave=False):
            x0, _ = batch
            x0 = to_minus1_1(x0.to(dev, non_blocking=True))

            feats_all = self._features_all_levels(adapter, x0, mc=int(cfg.mc_test))

            t_used = []
            per_t = []
            for lvl in self._levels:
                slot = int(lvl["k"])
                if slot not in self._t_kept:
                    continue
                s_t = self._agg_features_at_t(slot, feats_all[slot])  # OOD-high
                t_used.append(slot)
                per_t.append(s_t)

            if len(per_t) == 0:
                s = torch.zeros((x0.shape[0],), device=dev, dtype=torch.float32)
            else:
                S = torch.stack(per_t, dim=0)  # [T,B]
                s = self._agg_across_t(S, t_used)  # OOD-high

            scores.append(s.detach().cpu().numpy().astype(np.float32))

        if len(scores) == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)