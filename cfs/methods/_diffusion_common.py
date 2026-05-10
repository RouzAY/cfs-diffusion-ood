# cfs/methods/_diffusion_common.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import torch


# ============================================================
# Generic helpers
# ============================================================

def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure image tensor is in [-1, 1].
    Supports inputs already in [-1,1], [0,1], or [0,255]-ish.
    """
    x = x.detach()
    xmin, xmax = float(x.min()), float(x.max())
    if xmin >= -1.01 and xmax <= 1.01 and xmin < -1e-6:
        return x
    if xmin >= 0.0 and xmax <= 1.01:
        return (x * 2.0 - 1.0).clamp(-1.0, 1.0)
    return (x / 127.5 - 1.0).clamp(-1.0, 1.0)


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(start_dim=1)


def is_edm_adapter(adapter: Any) -> bool:
    """Conservative EDM detection."""
    if bool(getattr(adapter, "is_edm", False)):
        return True
    cls_name = type(adapter).__name__.lower()
    if "edm" in cls_name:
        return True
    if hasattr(adapter, "precond_type") and hasattr(adapter, "denoise") and callable(getattr(adapter, "denoise")):
        if not hasattr(adapter, "diffusion"):
            return True
    return False


def get_device(adapter: Any) -> torch.device:
    m = getattr(adapter, "model", None)
    if m is not None and hasattr(m, "parameters"):
        try:
            return next(m.parameters()).device
        except StopIteration:
            pass
    dev = getattr(adapter, "device", None)
    if isinstance(dev, str):
        return torch.device(dev)
    if isinstance(dev, torch.device):
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp_x(x: torch.Tensor, clamp: bool, clamp_range: Tuple[float, float]) -> torch.Tensor:
    if not clamp:
        return x
    lo, hi = float(clamp_range[0]), float(clamp_range[1])
    return x.clamp(lo, hi)


def _first_tensor(out: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v):
                return v
    return None


def _pick_first_tensor4d(out: Any) -> torch.Tensor:
    t = _first_tensor(out)
    if t is None:
        raise RuntimeError(f"Unsupported model output type: {type(out)}")
    if t.ndim != 4:
        raise RuntimeError(f"Expected 4D tensor output, got shape={tuple(t.shape)}")
    return t


# ============================================================
# Canonical level helpers
# ============================================================

def level_lambda(lvl: Dict[str, Any]) -> float:
    if "lambda_eff" in lvl:
        return float(lvl["lambda_eff"])
    return float(lvl.get("lambda", 0.0))


def sort_levels_clean_to_noisy(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort from cleaner to noisier. Larger lambda/logSNR = cleaner."""
    return sorted(levels, key=level_lambda, reverse=True)


def _select_uniform_indices(n_total: int, n_keep: int) -> List[int]:
    n_total = int(n_total)
    n_keep = int(n_keep)
    if n_total <= 0:
        return []
    if n_keep >= n_total:
        return list(range(n_total))
    raw = np.linspace(0, n_total - 1, n_keep)
    idx = np.round(raw).astype(np.int64)
    idx = np.clip(idx, 0, n_total - 1)
    out: List[int] = []
    used = set()
    for i in idx.tolist():
        if i not in used:
            out.append(int(i))
            used.add(int(i))
    if len(out) < n_keep:
        for j in range(n_total):
            if j not in used:
                out.append(int(j))
                used.add(int(j))
                if len(out) >= n_keep:
                    break
    return out[:n_keep]


# ============================================================
# Small heads
# ============================================================

class KDE1D:
    def __init__(self, bandwidth: float = 0.0):
        self.bw = float(bandwidth)
        self.ref: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            x = np.zeros((1,), dtype=np.float64)
        if self.bw <= 0.0:
            q25, q75 = np.percentile(x, [25, 75])
            iqr = float(q75 - q25)
            std = float(np.std(x))
            sigma = min(std, iqr / 1.34 if iqr > 0 else std)
            N = max(1, int(x.size))
            bw = 0.9 * max(sigma, 1e-8) * (N ** (-1.0 / 5.0))
            if (not np.isfinite(bw)) or bw <= 1e-6:
                bw = 0.1
            self.bw = float(bw)
        self.ref = x.astype(np.float64, copy=True)

    def score_ood(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.full_like(z, fill_value=np.inf, dtype=np.float64)
        mask = np.isfinite(z)
        if not np.any(mask):
            return out.astype(np.float32)
        X = self.ref if self.ref is not None else np.zeros((1,), dtype=np.float64)
        bw2 = float(self.bw * self.bw)
        cst = -0.5 * np.log(2.0 * np.pi * bw2)
        zz = z[mask][:, None]
        diff2 = (zz - X[None, :]) ** 2
        lp = np.log(np.mean(np.exp(cst - diff2 / (2.0 * bw2)), axis=1) + 1e-300)
        out[mask] = -lp
        return out.astype(np.float32)


class DiagGaussian:
    def __init__(self):
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("DiagGaussian.fit expects shape (N, D).")
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1]), dtype=np.float64)
        self.mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        var = np.where(np.isfinite(var) & (var > 1e-8), var, 1.0)
        self.var = var

    def score_ood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("DiagGaussian.score_ood expects shape (N, D).")
        mu = self.mu if self.mu is not None else np.zeros((X.shape[1],), dtype=np.float64)
        var = self.var if self.var is not None else np.ones((X.shape[1],), dtype=np.float64)
        z = (X - mu) ** 2 / var
        nll = 0.5 * np.sum(z + np.log(var), axis=1)
        return nll.astype(np.float32)


# ============================================================
# Improved / VP schedule helpers
# ============================================================

def get_alphas_cumprod(adapter: Any) -> np.ndarray:
    ab = getattr(adapter, "alphas_cumprod", None)
    if torch.is_tensor(ab):
        return ab.detach().float().cpu().numpy()
    if ab is not None:
        return np.asarray(ab, dtype=np.float64)
    diff = getattr(adapter, "diffusion", None)
    if diff is not None:
        ab = getattr(diff, "alphas_cumprod", None)
        if ab is not None:
            return np.asarray(ab, dtype=np.float64)
    raise RuntimeError("alphas_cumprod not found on adapter/diffusion.")


def as_torch_scalar(arr, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(arr):
        v = arr[int(t)]
        return v.to(device=device, dtype=dtype).view(1, 1, 1, 1)
    v = np.asarray(arr)[int(t)]
    return torch.tensor(float(v), device=device, dtype=dtype).view(1, 1, 1, 1)


def sqrt_ab_and_sigma_improved(adapter: Any, t: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = getattr(adapter, "diffusion", None)
    sqrt_ab = getattr(diff, "sqrt_alphas_cumprod", None) if diff is not None else None
    if sqrt_ab is None:
        sqrt_ab = getattr(adapter, "sqrt_ab", None)
    if sqrt_ab is None:
        ab = get_alphas_cumprod(adapter)
        sqrt_ab = np.sqrt(np.maximum(ab, 0.0))

    sig = getattr(diff, "sqrt_one_minus_alphas_cumprod", None) if diff is not None else None
    if sig is None:
        sig = getattr(adapter, "sqrt_one_minus_ab", None)
    if sig is None:
        ab = get_alphas_cumprod(adapter)
        sig = np.sqrt(np.maximum(1.0 - ab, 0.0))
    return as_torch_scalar(sqrt_ab, t, device, dtype), as_torch_scalar(sig, t, device, dtype)


def improved_lambdas_from_timesteps(adapter: Any, timesteps: Sequence[int]) -> List[float]:
    ab = get_alphas_cumprod(adapter).astype(np.float64)
    ab = np.clip(ab, 1e-12, 1.0 - 1e-12)
    logsnr_t = np.log(ab) - np.log(1.0 - ab)
    T = len(logsnr_t)
    out = []
    for t in timesteps:
        tt = int(np.clip(int(t), 0, T - 1))
        out.append(float(logsnr_t[tt]))
    return out


def summarize_canonical_levels(adapter: Any, explicit_lambdas: Sequence[float], match_mode: str = "logsnr") -> List[Dict[str, Any]]:
    levels = build_canonical_levels(
        adapter,
        lambda_min=min(explicit_lambdas),
        lambda_max=max(explicit_lambdas),
        Kc=len(explicit_lambdas),
        K_grid=len(explicit_lambdas),
        unique=False,
        explicit_lambdas=explicit_lambdas,
        match_mode=match_mode,
    )
    rows = []
    for lv in levels:
        row = {
            "k": int(lv["k"]),
            "lambda_target": float(lv["lambda_target"]),
            "lambda_eff": float(lv["lambda_eff"]),
            "a": float(lv["a"]),
            "b": float(lv["b"]),
            "b2": float(lv["b"]) ** 2,
            "sigma_tilde": float(lv["sigma_tilde"]),
            "match_mode": lv.get("match_mode", None),
        }
        if "t" in lv:
            row["native_t"] = int(lv["t"])
        if "sigma" in lv:
            row["sigma"] = float(lv["sigma"])
            row["precond"] = lv.get("precond", None)
        rows.append(row)
    return rows


def make_discrete_condition(adapter: Any, t: int, batch_size: int, device: torch.device) -> torch.Tensor:
    mk_cond = getattr(adapter, "make_condition", None)
    if callable(mk_cond):
        try:
            return mk_cond(t=int(t), batch_size=int(batch_size), device=device)
        except TypeError:
            return mk_cond(int(t), int(batch_size), device)

    tt_raw = torch.full((int(batch_size),), int(t), device=device, dtype=torch.long)
    return _scale_timesteps_improved(adapter, tt_raw)


@torch.no_grad()
def forward_discrete_model(
    adapter: Any,
    x: torch.Tensor,
    t: int,
    use_amp: bool = True,
) -> torch.Tensor:
    dev = x.device
    cond = make_discrete_condition(adapter, int(t), int(x.shape[0]), dev)

    if hasattr(adapter, "forward_model") and callable(getattr(adapter, "forward_model")):
        return adapter.forward_model(x, cond)

    model = getattr(adapter, "model", None)
    if model is None:
        raise RuntimeError("adapter.model not found.")

    use_amp = bool(use_amp) and (dev.type == "cuda")
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            return model(x, cond)
    return model(x, cond)


@torch.no_grad()
def forward_eps_improved(adapter: Any, x: torch.Tensor, t: int, internal_bs: int, use_amp: bool) -> torch.Tensor:
    model = getattr(adapter, "model", None)
    if model is None:
        raise RuntimeError("adapter.model not found.")
    B, C, _, _ = x.shape
    outs: List[torch.Tensor] = []
    dev = x.device
    use_amp = bool(use_amp) and (dev.type == "cuda")
    for j in range(0, B, int(internal_bs)):
        xj = x[j:j + int(internal_bs)]
        bj = xj.shape[0]
        # tt = torch.full((bj,), int(t), device=dev, dtype=torch.long)
        # if use_amp:
        #     with torch.amp.autocast("cuda", dtype=torch.float16):
        #         out = model(xj, tt)
        # else:
        #     out = model(xj, tt)
        out = forward_discrete_model(adapter, xj, t, use_amp=use_amp)
        out_t = _pick_first_tensor4d(out)
        eps_pred = out_t[:, :C].float()
        outs.append(eps_pred)
    return torch.cat(outs, dim=0)


def _scale_timesteps_improved(adapter: Any, t: torch.Tensor) -> torch.Tensor:
    diff = getattr(adapter, "diffusion", None)
    fn = getattr(diff, "_scale_timesteps", None) if diff is not None else None
    if callable(fn):
        return fn(t)
    fn2 = getattr(adapter, "scale_timesteps", None)
    if callable(fn2):
        return fn2(t)
    return t


# ============================================================
# Canonical levels
# ============================================================

def build_canonical_levels(
    adapter: Any,
    lambda_min: float,
    lambda_max: float,
    Kc: int,
    K_grid: int | None = None,
    unique: bool = True,
    explicit_lambdas: Optional[Sequence[float]] = None,
    match_mode: str = "logsnr",
) -> List[Dict[str, Any]]:
    """
    Build canonical logSNR levels.

    Discrete VP/DDPM-style backbones are matched to nearest native timestep in
    logSNR. EDM-style backbones are continuous in sigma.
    """
    match_mode = str(match_mode).lower()
    if match_mode not in {"logsnr", "uniform_t"}:
        raise ValueError(f"Unknown match_mode={match_mode!r}")

    if explicit_lambdas is not None and len(explicit_lambdas) > 0:
        lam_grid = np.asarray([float(x) for x in explicit_lambdas], dtype=np.float64)
        K_target = int(len(lam_grid))
    else:
        Kc = int(max(1, Kc))
        if K_grid is None or int(K_grid) <= 0:
            K_grid = Kc
        K_grid = int(max(1, K_grid))
        lam_grid = np.linspace(float(lambda_min), float(lambda_max), num=K_grid).astype(np.float64)
        K_target = Kc

    # EDM / continuous sigma side.
    if is_edm_adapter(adapter):
        precond = str(getattr(adapter, "precond_type", "unknown")).lower()
        raw: List[Dict[str, Any]] = []
        for i, lam_tgt in enumerate(lam_grid):
            sigma = float(np.exp(-0.5 * float(lam_tgt)))
            if precond == "vp":
                a = 1.0 / np.sqrt(1.0 + sigma * sigma)
                b = sigma * a
            else:
                a = 1.0
                b = sigma
            raw.append({
                "grid_k": i,
                "lambda_target": float(lam_tgt),
                "lambda_eff": float(lam_tgt),
                "lambda": float(lam_tgt),
                "sigma": sigma,
                "a": float(a),
                "b": float(b),
                "sigma_tilde": float(b / max(a, 1e-12)),
                "precond": precond,
                "match_mode": "continuous",
            })
        raw = sort_levels_clean_to_noisy(raw)
        out = raw if explicit_lambdas is not None and len(explicit_lambdas) > 0 else [raw[j] for j in _select_uniform_indices(len(raw), K_target)]
        for k, lvl in enumerate(out):
            lvl["k"] = int(k)
        return out

    # Discrete VP/DDPM-style side, including improved-diffusion, DiT and U-ViT adapters.
    ab = get_alphas_cumprod(adapter).astype(np.float64)
    ab = np.clip(ab, 1e-12, 1.0 - 1e-12)
    logsnr_t = np.log(ab) - np.log(1.0 - ab)
    T = int(len(logsnr_t))

    if match_mode == "logsnr":
        t_by_i = [int(np.argmin(np.abs(logsnr_t - float(lam_tgt)))) for lam_tgt in lam_grid]
    else:
        lam_clean = float(np.max(logsnr_t))
        lam_noisy = float(np.min(logsnr_t))
        denom = max(lam_clean - lam_noisy, 1e-12)
        t_by_i = []
        for lam_tgt in lam_grid:
            r_noisy = (lam_clean - float(lam_tgt)) / denom
            r_noisy = float(np.clip(r_noisy, 0.0, 1.0))
            tk = int(np.round(r_noisy * (T - 1)))
            tk = int(np.clip(tk, 0, T - 1))
            t_by_i.append(tk)

    raw2: List[Dict[str, Any]] = []
    for i, lam_tgt in enumerate(lam_grid):
        tk = int(np.clip(t_by_i[i], 0, T - 1))
        a = float(np.sqrt(ab[tk]))
        b = float(np.sqrt(1.0 - ab[tk]))
        lam_eff = float(logsnr_t[tk])
        raw2.append({
            "grid_k": i,
            "lambda_target": float(lam_tgt),
            "lambda_eff": lam_eff,
            "lambda": lam_eff,
            "t": tk,
            "a": a,
            "b": b,
            "sigma_tilde": float(b / max(a, 1e-12)),
            "match_mode": match_mode,
        })

    if unique:
        best_by_t: Dict[int, Dict[str, Any]] = {}
        for lvl in raw2:
            t = int(lvl["t"])
            if t not in best_by_t:
                best_by_t[t] = lvl
            elif match_mode == "logsnr":
                err = abs(float(lvl["lambda_target"]) - float(lvl["lambda_eff"]))
                err_prev = abs(float(best_by_t[t]["lambda_target"]) - float(best_by_t[t]["lambda_eff"]))
                if err < err_prev:
                    best_by_t[t] = lvl
        uniq = list(best_by_t.values())
    else:
        uniq = raw2

    uniq = sort_levels_clean_to_noisy(uniq)
    out = uniq if explicit_lambdas is not None and len(explicit_lambdas) > 0 else [uniq[j] for j in _select_uniform_indices(len(uniq), K_target)]
    for k, lvl in enumerate(out):
        lvl["k"] = int(k)
    return out


# ============================================================
# Corruption / x0 / eps / score helpers
# ============================================================

def _broadcast_ab(lvl: Dict[str, Any], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.tensor(float(lvl["a"]), device=x.device, dtype=x.dtype).view(1, 1, 1, 1)
    b = torch.tensor(float(lvl["b"]), device=x.device, dtype=x.dtype).view(1, 1, 1, 1)
    return a, b


@torch.no_grad()
def corrupt_from_x0(x0: torch.Tensor, lvl: Dict[str, Any], eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    if eps is None:
        eps = torch.randn_like(x0)
    a, b = _broadcast_ab(lvl, x0)
    return a * x0 + b * eps


@torch.no_grad()
def estimate_x0(adapter: Any, x: torch.Tensor, lvl: Dict[str, Any], internal_bs: int = 64, use_amp: bool = True) -> torch.Tensor:
    if ("t" in lvl) and (lvl["t"] is not None):
        model = getattr(adapter, "model", None)
        if model is None:
            raise RuntimeError("estimate_x0: adapter.model not found for discrete branch.")
        t = int(lvl["t"])
        B, C, _, _ = x.shape
        dev = x.device
        use_amp = bool(use_amp) and (dev.type == "cuda")
        ab = get_alphas_cumprod(adapter)
        ab_t = torch.as_tensor(ab, device=dev, dtype=torch.float32)[t].view(1, 1, 1, 1)
        sqrt_ab = torch.sqrt(ab_t.clamp_min(1e-12))
        sqrt_1mab = torch.sqrt((1.0 - ab_t).clamp_min(1e-12))
        outs: List[torch.Tensor] = []
        for j in range(0, B, int(internal_bs)):
            xj = x[j:j + int(internal_bs)]
            bj = xj.shape[0]
            tt = make_discrete_condition(adapter, t, bj, dev)
            if hasattr(adapter, "forward_model") and callable(getattr(adapter, "forward_model")):
                out = adapter.forward_model(xj, tt)
            else:
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(xj, tt)
                else:
                    out = model(xj, tt)
            out = _pick_first_tensor4d(out)
            eps_pred = out[:, :C].float()
            x0_hat = (xj.float() - sqrt_1mab * eps_pred) / sqrt_ab
            outs.append(x0_hat)
        return torch.cat(outs, dim=0)

    if ("sigma" in lvl) and (lvl["sigma"] is not None):
        if not hasattr(adapter, "denoise") or not callable(getattr(adapter, "denoise")):
            raise RuntimeError("estimate_x0: lvl contains 'sigma' but adapter has no callable denoise().")
        sigma = float(lvl["sigma"])
        B = x.shape[0]
        outs: List[torch.Tensor] = []
        for j in range(0, B, int(internal_bs)):
            xj = x[j:j + int(internal_bs)]
            x0_hat = adapter.denoise(xj, sigma).to(device=x.device, dtype=torch.float32)
            outs.append(x0_hat)
        return torch.cat(outs, dim=0)

    raise ValueError(f"estimate_x0: malformed level dict. Expected key 't' or 'sigma', got keys={list(lvl.keys())}")


@torch.no_grad()
def estimate_eps_from_x0hat(x: torch.Tensor, x0_hat: torch.Tensor, lvl: Dict[str, Any], eps: float = 1e-6) -> torch.Tensor:
    a = float(lvl["a"])
    b = float(lvl["b"])
    b_safe = max(abs(b), float(eps))
    return ((x - a * x0_hat) / b_safe).float()


@torch.no_grad()
def estimate_score_from_x0hat(x: torch.Tensor, x0_hat: torch.Tensor, lvl: Dict[str, Any], eps: float = 1e-6) -> torch.Tensor:
    eps_hat = estimate_eps_from_x0hat(x, x0_hat, lvl, eps=eps)
    b = float(lvl["b"])
    b_safe = max(abs(b), float(eps))
    return (-eps_hat / b_safe).float()


@torch.no_grad()
def estimate_x0_eps_native(adapter: Any, x: torch.Tensor, lvl: Dict[str, Any], internal_bs: int = 64, use_amp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    x0_hat = estimate_x0(adapter, x, lvl, internal_bs=internal_bs, use_amp=use_amp)
    eps_hat = estimate_eps_from_x0hat(x, x0_hat, lvl)
    return x0_hat.float(), eps_hat.float()


@torch.no_grad()
def estimate_eps_native(adapter: Any, x: torch.Tensor, lvl: Dict[str, Any], internal_bs: int = 64, use_amp: bool = True) -> torch.Tensor:
    _, eps_hat = estimate_x0_eps_native(adapter, x, lvl, internal_bs=internal_bs, use_amp=use_amp)
    return eps_hat