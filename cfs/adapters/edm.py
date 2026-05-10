# dtd/adapters/edm.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import pickle
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

"""
EDMAdapter (NVLabs EDM .pkl)
----------------------------
Patch notes:
- Detects preconditioning family from loaded checkpoint:
    * VPPrecond  -> precond_type = "vp"
    * VEPrecond  -> precond_type = "ve"
    * EDMPrecond -> precond_type = "edm"
- Exposes sigma_to_ab(): converts model input sigma (interpreted as sigma_tilde for VP)
  into corruption coefficients a,b used by coordinate-free methods.
- denoise(x, sigma) remains ONE-SHOT (single network forward), not a sampler.
- score_src() kept as a lightweight proxy; coordinate-free methods should prefer
  sigma_to_ab() + denoise() for exact canonical score computation.

API:
- self.model
- self.precond_type in {"vp", "ve", "edm", "unknown"}
- denoise(x, sigma) -> x0_hat
- sigma_to_ab(sigma, x) -> (sigma_vec, a, b)
- score_src(x, sigma) -> legacy proxy (not the canonical score for all preconds)
- sigma_t(t_idx) schedule Karras (not used by CFSTEIN directly)
"""

# ---------------------------------------------------------------------
# Locate edm-main repo (for dnnlib)
# ---------------------------------------------------------------------
_here = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    os.environ.get("EDM_REPO_DIR"),
    os.path.abspath(os.path.join(_here, "../../..", "repos", "edm-main")),
    os.path.abspath(os.path.join(_here, "../../..", "edm-main")),
    os.path.abspath(os.path.join(_here, "..", "..", "repos", "edm-main")),
]
for c in _candidates:
    if c and os.path.isdir(c) and c not in sys.path:
        sys.path.append(c)

try:
    import dnnlib  # type: ignore
except Exception:
    dnnlib = None


class EDMAdapter:
    def __init__(
        self,
        checkpoint_path: str,
        prediction_type: str = "x0",
        in_channels: int = 3,
        data_range: Tuple[float, float] = (-1.0, 1.0),
        device: Union[str, torch.device, None] = None,
        edm_repo_dir: Optional[str] = None,
    ):
        # --------------------- Device ---------------------
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(str(device))

        self.prediction_type = str(prediction_type).lower()
        self.in_channels = int(in_channels)
        self.data_lo, self.data_hi = float(data_range[0]), float(data_range[1])

        self.is_edm = True  # flag for methods

        self._sigma_schedule: Optional[torch.Tensor] = None
        self.sampler_cfg: Dict[str, Any] = {}

        # --------------------- Repo EDM ---------------------
        if edm_repo_dir is not None and os.path.isdir(edm_repo_dir) and edm_repo_dir not in sys.path:
            sys.path.insert(0, edm_repo_dir)

        global dnnlib
        if dnnlib is None:
            try:
                import dnnlib as _dnnlib  # type: ignore
                dnnlib = _dnnlib
            except Exception as e:
                raise RuntimeError(
                    "[EDMAdapter] Cannot import dnnlib. "
                    "Set EDM_REPO_DIR or place edm-main in repos/edm-main."
                ) from e

        # --------------------- Load checkpoint ---------------------
        ckpt_path = checkpoint_path
        if not ckpt_path.startswith("http"):
            ckpt_path = os.path.abspath(ckpt_path)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"[EDMAdapter] Checkpoint not found: {ckpt_path}")
        self.checkpoint_path = ckpt_path

        ext = os.path.splitext(ckpt_path)[1].lower()
        if ext in [".pkl", ".pickle"]:
            with dnnlib.util.open_url(ckpt_path, verbose=False) as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "ema" in obj:
                net = obj["ema"]
            elif hasattr(obj, "forward"):
                net = obj
            else:
                raise RuntimeError("[EDMAdapter] .pkl format not recognized (no 'ema' and not nn.Module).")
            self.model = net.to(self.device).eval()
        else:
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=self.device)
            if isinstance(ckpt, torch.nn.Module):
                self.model = ckpt.to(self.device).eval()
            elif isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
                self.model = ckpt["model"].to(self.device).eval()
            else:
                raise RuntimeError(
                    "[EDMAdapter] .pt/.pth unsupported (looks like a raw state_dict). "
                    "Instantiate EDM net then load state_dict upstream."
                )

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.img_resolution = getattr(self.model, "img_resolution", None)
        self.img_channels = getattr(self.model, "img_channels", None)
        self.sigma_min = float(getattr(self.model, "sigma_min", 0.0) or 0.0)
        self.sigma_max = float(getattr(self.model, "sigma_max", 0.0) or 0.0)

        # --------------------- Preconditioning family ---------------------
        self.model_class_name = self.model.__class__.__name__
        cls = self.model_class_name.lower()
        if "vpprecond" in cls:
            self.precond_type = "vp"
        elif "veprecond" in cls:
            self.precond_type = "ve"
        elif "edmprecond" in cls:
            self.precond_type = "edm"
        else:
            self.precond_type = "unknown"

        name = os.path.basename(ckpt_path)
        print(
            f"[EDMAdapter] Loaded EDM network '{name}' on device={self.device} "
            f"(class={self.model_class_name}, precond={self.precond_type})"
        )
        print(
            f"[EDMAdapter] img_resolution={self.img_resolution}, img_channels={self.img_channels}, "
            f"sigma_min={self.sigma_min:.4g}, sigma_max={self.sigma_max:.4g}"
        )

    # -----------------------------------------------------------------
    # sigma utilities
    # -----------------------------------------------------------------
    def _sigma_vec(self, sigma: Any, x: torch.Tensor) -> torch.Tensor:
        """
        Return sigma as 1D float32 tensor (B,) on x.device.
        Accepts: float, scalar tensor, (1,), (B,), etc.
        """
        B = int(x.shape[0])
        if torch.is_tensor(sigma):
            s = sigma.to(device=x.device, dtype=torch.float32).reshape(-1)
        else:
            s = torch.tensor([float(sigma)], device=x.device, dtype=torch.float32)

        if s.numel() == 1:
            s = s.expand(B)
        elif s.numel() != B:
            raise ValueError(f"[EDMAdapter] sigma must have 1 or B elements, got {s.numel()} (B={B}).")
        return s

    def sigma_to_ab(self, sigma: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the model input sigma into corruption coefficients a,b.

        Returns:
            sigma_vec : (B,) float32
            a        : (B,1,1,1) float32
            b        : (B,1,1,1) float32

        Convention:
        - For VPPrecond, sigma is interpreted as sigma_tilde = b / a
          => a = 1 / sqrt(1 + sigma^2), b = sigma / sqrt(1 + sigma^2)
        - For VEPrecond and EDMPrecond, we use VE-like corruption
          => a = 1, b = sigma
        - For unknown, fallback to VE-like.
        """
        sigma_vec = self._sigma_vec(sigma, x)  # (B,)
        sig = sigma_vec.view(-1, 1, 1, 1)

        if self.precond_type == "vp":
            a = 1.0 / torch.sqrt(1.0 + sig * sig)
            b = sig * a
            return sigma_vec, a, b

        # VEPrecond, EDMPrecond, unknown -> VE-like
        a = torch.ones_like(sig)
        b = sig
        return sigma_vec, a, b

    @torch.no_grad()
    def denoise(self, x: torch.Tensor, sigma: Any) -> torch.Tensor:
        """
        Return x0_hat at noise level sigma.
        ONE-SHOT network forward (not a sampler).

        HARD FIX: force x to float32 (EDM preconditioners often assert fp32).
        """
        x = x.to(self.device, dtype=torch.float32)
        sigma_vec = self._sigma_vec(sigma, x)  # (B,) float32

        try:
            out = self.model(x, sigma_vec, class_labels=None, force_fp32=True)
        except TypeError:
            out = self.model(x, sigma_vec, class_labels=None)

        if self.prediction_type == "x0":
            return out

        # Fallback (rare): if model output is epsilon-like or v-like
        sig = sigma_vec.view(-1, 1, 1, 1)
        if self.prediction_type in ("epsilon", "eps", "v"):
            return x - sig * out

        raise ValueError(f"[EDMAdapter] Unknown prediction_type={self.prediction_type!r}")

    def score_src(self, x: torch.Tensor, sigma: Any, allow_grad: bool = False) -> torch.Tensor:
        """
        Legacy score proxy retained for compatibility:
            (x0_hat - x) / sigma

        WARNING:
        This is NOT the exact canonical data-space score for VP checkpoints.
        Coordinate-free methods should instead use:
            sigma_to_ab() + denoise() and compute (a * x0_hat - x) / b^2.
        """
        x = x.to(self.device, dtype=torch.float32)
        sigma_vec = self._sigma_vec(sigma, x)

        if allow_grad:
            x0_hat = self.denoise(x, sigma_vec)
        else:
            with torch.no_grad():
                x0_hat = self.denoise(x, sigma_vec)

        sig = sigma_vec.view(-1, 1, 1, 1)
        return (x0_hat - x) / (sig + 1e-8)

    # -----------------------------------------------------------------
    # sigma schedule (Karras) -- not used by CFSTEIN directly
    # -----------------------------------------------------------------
    def _ensure_sigma_schedule(self) -> None:
        """
        Build Karras schedule. Robust to sigma_min=0 / sigma_max=inf from pickles.
        """
        if self._sigma_schedule is not None:
            return

        cfg = getattr(self, "sampler_cfg", {}) or {}
        steps = int(cfg.get("steps", 32))
        rho = float(cfg.get("rho", 7.0))

        sigma_min_cfg = cfg.get("sigma_min", None)
        sigma_max_cfg = cfg.get("sigma_max", None)

        sigma_min = float(sigma_min_cfg) if sigma_min_cfg is not None else float(self.sigma_min)
        sigma_max = float(sigma_max_cfg) if sigma_max_cfg is not None else float(self.sigma_max)

        if (not np.isfinite(sigma_min)) or (sigma_min <= 0.0):
            sigma_min = 0.002
        if (not np.isfinite(sigma_max)) or (sigma_max <= 0.0) or (sigma_max > 1e6):
            sigma_max = 80.0

        ramp = torch.linspace(0.0, 1.0, steps, device=self.device, dtype=torch.float32)
        min_inv = sigma_min ** (1.0 / rho)
        max_inv = sigma_max ** (1.0 / rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho

        self._sigma_schedule = sigmas

    def sigma_t(self, t_idx: int) -> torch.Tensor:
        self._ensure_sigma_schedule()
        assert self._sigma_schedule is not None
        t = int(t_idx)
        t = max(0, min(t, int(self._sigma_schedule.numel()) - 1))
        return self._sigma_schedule[t]

    def get_sigma_schedule(self) -> torch.Tensor:
        self._ensure_sigma_schedule()
        assert self._sigma_schedule is not None
        return self._sigma_schedule


# # dtd/adapters/edm.py
# # -*- coding: utf-8 -*-
# from __future__ import annotations

# import os
# import sys
# import pickle
# from typing import Any, Dict, Optional, Tuple, Union

# import numpy as np
# import torch

# """
# EDMAdapter (NVLabs EDM .pkl)
# ----------------------------
# Fixes:
# - EDM nets assert often require x.dtype == float32 and sigma.dtype == float32.
# - If autocast is enabled upstream, x may arrive as fp16 -> AssertionError in EDMPrecond.forward.
# => We force fp32 inside denoise/score_src and (best effort) call forward with force_fp32=True.

# API:
# - self.model
# - denoise(x, sigma) -> x0_hat
# - score_src(x, sigma) -> (x0_hat - x)/sigma
# - sigma_t(t_idx) schedule Karras
# """

# # ---------------------------------------------------------------------
# # Locate edm-main repo (for dnnlib)
# # ---------------------------------------------------------------------
# _here = os.path.abspath(os.path.dirname(__file__))
# _candidates = [
#     os.environ.get("EDM_REPO_DIR"),
#     os.path.abspath(os.path.join(_here, "../../..", "repos", "edm-main")),
#     os.path.abspath(os.path.join(_here, "../../..", "edm-main")),
#     os.path.abspath(os.path.join(_here, "..", "..", "repos", "edm-main")),
# ]
# for c in _candidates:
#     if c and os.path.isdir(c) and c not in sys.path:
#         sys.path.append(c)

# try:
#     import dnnlib  # type: ignore
# except Exception:
#     dnnlib = None


# class EDMAdapter:
#     def __init__(
#         self,
#         checkpoint_path: str,
#         prediction_type: str = "x0",
#         in_channels: int = 3,
#         data_range: Tuple[float, float] = (-1.0, 1.0),
#         device: Union[str, torch.device, None] = None,
#         edm_repo_dir: Optional[str] = None,
#     ):
#         # --------------------- Device ---------------------
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         elif isinstance(device, torch.device):
#             self.device = device
#         else:
#             self.device = torch.device(str(device))

#         self.prediction_type = str(prediction_type).lower()
#         self.in_channels = int(in_channels)
#         self.data_lo, self.data_hi = float(data_range[0]), float(data_range[1])

#         self.is_edm = True  # flag for methods

#         self._sigma_schedule: Optional[torch.Tensor] = None
#         self.sampler_cfg: Dict[str, Any] = {}

#         # --------------------- Repo EDM ---------------------
#         if edm_repo_dir is not None and os.path.isdir(edm_repo_dir) and edm_repo_dir not in sys.path:
#             sys.path.insert(0, edm_repo_dir)

#         global dnnlib
#         if dnnlib is None:
#             try:
#                 import dnnlib as _dnnlib  # type: ignore
#                 dnnlib = _dnnlib
#             except Exception as e:
#                 raise RuntimeError(
#                     "[EDMAdapter] Cannot import dnnlib. "
#                     "Set EDM_REPO_DIR or place edm-main in repos/edm-main."
#                 ) from e

#         # --------------------- Load checkpoint ---------------------
#         ckpt_path = checkpoint_path
#         if not ckpt_path.startswith("http"):
#             ckpt_path = os.path.abspath(ckpt_path)
#             if not os.path.exists(ckpt_path):
#                 raise FileNotFoundError(f"[EDMAdapter] Checkpoint not found: {ckpt_path}")
#         self.checkpoint_path = ckpt_path

#         ext = os.path.splitext(ckpt_path)[1].lower()
#         if ext in [".pkl", ".pickle"]:
#             with dnnlib.util.open_url(ckpt_path, verbose=False) as f:
#                 obj = pickle.load(f)
#             if isinstance(obj, dict) and "ema" in obj:
#                 net = obj["ema"]
#             elif hasattr(obj, "forward"):
#                 net = obj
#             else:
#                 raise RuntimeError("[EDMAdapter] .pkl format not recognized (no 'ema' and not nn.Module).")
#             self.model = net.to(self.device).eval()
#         else:
#             # Rare: pickled nn.Module
#             try:
#                 ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
#             except TypeError:
#                 ckpt = torch.load(ckpt_path, map_location=self.device)
#             if isinstance(ckpt, torch.nn.Module):
#                 self.model = ckpt.to(self.device).eval()
#             elif isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
#                 self.model = ckpt["model"].to(self.device).eval()
#             else:
#                 raise RuntimeError(
#                     "[EDMAdapter] .pt/.pth unsupported (looks like a raw state_dict). "
#                     "Instantiate EDM net then load state_dict upstream."
#                 )

#         for p in self.model.parameters():
#             p.requires_grad_(False)

#         self.img_resolution = getattr(self.model, "img_resolution", None)
#         self.img_channels = getattr(self.model, "img_channels", None)
#         self.sigma_min = float(getattr(self.model, "sigma_min", 0.0) or 0.0)
#         self.sigma_max = float(getattr(self.model, "sigma_max", 0.0) or 0.0)

#         name = os.path.basename(ckpt_path)
#         print(
#             f"[EDMAdapter] Loaded EDM network '{name}' on device={self.device} "
#             f"(class={self.model.__class__.__name__})"
#         )
#         print(
#             f"[EDMAdapter] img_resolution={self.img_resolution}, img_channels={self.img_channels}, "
#             f"sigma_min={self.sigma_min:.4g}, sigma_max={self.sigma_max:.4g}"
#         )

#     # -----------------------------------------------------------------
#     # sigma utilities
#     # -----------------------------------------------------------------
#     def _sigma_vec(self, sigma: Any, x: torch.Tensor) -> torch.Tensor:
#         """
#         Return sigma as 1D float32 tensor (B,) on x.device.
#         Accept: float, scalar tensor, (1,), (B,), etc.
#         """
#         B = int(x.shape[0])
#         if torch.is_tensor(sigma):
#             s = sigma.to(device=x.device, dtype=torch.float32).reshape(-1)
#         else:
#             s = torch.tensor([float(sigma)], device=x.device, dtype=torch.float32)

#         if s.numel() == 1:
#             s = s.expand(B)
#         elif s.numel() != B:
#             raise ValueError(f"[EDMAdapter] sigma must have 1 or B elements, got {s.numel()} (B={B}).")
#         return s

#     @torch.no_grad()
#     def denoise(self, x: torch.Tensor, sigma: Any) -> torch.Tensor:
#         """
#         Return x0_hat at noise level sigma.
#         HARD FIX: force x to float32 (EDMPrecond asserts often require fp32).
#         """
#         x = x.to(self.device, dtype=torch.float32)
#         sigma_vec = self._sigma_vec(sigma, x)  # (B,) float32

#         # Many NVLabs EDM nets accept force_fp32 kwarg. Use best-effort.
#         try:
#             out = self.model(x, sigma_vec, class_labels=None, force_fp32=True)
#         except TypeError:
#             out = self.model(x, sigma_vec, class_labels=None)

#         if self.prediction_type == "x0":
#             return out

#         # fallback (rare)
#         sig = sigma_vec.view(-1, 1, 1, 1)
#         if self.prediction_type in ("epsilon", "eps", "v"):
#             return x - sig * out

#         raise ValueError(f"[EDMAdapter] Unknown prediction_type={self.prediction_type!r}")

#     def score_src(self, x: torch.Tensor, sigma: Any, allow_grad: bool = False) -> torch.Tensor:
#         """
#         Score proxy: (x0_hat - x) / sigma.
#         HARD FIX: force x float32.
#         """
#         x = x.to(self.device, dtype=torch.float32)
#         sigma_vec = self._sigma_vec(sigma, x)  # (B,) float32

#         if allow_grad:
#             x0_hat = self.denoise(x, sigma_vec)
#         else:
#             with torch.no_grad():
#                 x0_hat = self.denoise(x, sigma_vec)

#         sig = sigma_vec.view(-1, 1, 1, 1)
#         return (x0_hat - x) / (sig + 1e-8)

#     # -----------------------------------------------------------------
#     # sigma schedule (Karras)
#     # -----------------------------------------------------------------
#     def _ensure_sigma_schedule(self) -> None:
#         """
#         Build Karras schedule. Robust to sigma_min=0 / sigma_max=inf from pickles.
#         """
#         if self._sigma_schedule is not None:
#             return

#         cfg = getattr(self, "sampler_cfg", {}) or {}
#         steps = int(cfg.get("steps", 32))
#         rho = float(cfg.get("rho", 7.0))

#         sigma_min_cfg = cfg.get("sigma_min", None)
#         sigma_max_cfg = cfg.get("sigma_max", None)

#         sigma_min = float(sigma_min_cfg) if sigma_min_cfg is not None else float(self.sigma_min)
#         sigma_max = float(sigma_max_cfg) if sigma_max_cfg is not None else float(self.sigma_max)

#         if (not np.isfinite(sigma_min)) or (sigma_min <= 0.0):
#             sigma_min = 0.002
#         if (not np.isfinite(sigma_max)) or (sigma_max <= 0.0) or (sigma_max > 1e6):
#             sigma_max = 80.0

#         ramp = torch.linspace(0.0, 1.0, steps, device=self.device, dtype=torch.float32)
#         min_inv = sigma_min ** (1.0 / rho)
#         max_inv = sigma_max ** (1.0 / rho)
#         sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho  # decreasing sigma_max -> sigma_min

#         self._sigma_schedule = sigmas

#     def sigma_t(self, t_idx: int) -> torch.Tensor:
#         self._ensure_sigma_schedule()
#         assert self._sigma_schedule is not None
#         t = int(t_idx)
#         t = max(0, min(t, int(self._sigma_schedule.numel()) - 1))
#         return self._sigma_schedule[t]

#     def get_sigma_schedule(self) -> torch.Tensor:
#         self._ensure_sigma_schedule()
#         assert self._sigma_schedule is not None
#         return self._sigma_schedule



# # dtd/adapters/edm.py
# # -*- coding: utf-8 -*-
# import os, sys, pickle, torch, numpy as np

# """
# EDMAdapter
# ----------
# Adaptateur unifié pour les modèles EDM (NVLabs) au format .pkl.

# - Charge les checkpoints NVLabs (edm-main) via dnnlib + pickle.
# - Accepte aussi un nn.Module picklé (.pt/.pth) directement (cas rare).
# - Expose une API minimale compatible avec nos méthodes OOD basées score/denoise :

#     - self.model         : réseau EDM (EDMPrecond / VEPrecond / VPPrecond / iDDPMPrecond)
#     - self.device        : device torch
#     - denoise(x, sigma)  : x0_hat ~ x débruité conditionnellement à sigma
#     - score_src(x, sigma): (x0_hat - x) / sigma (approx. score, signe pas critique pour GEPC)

# Les pas en sigma (schedule) ou les solveurs ODE (Heun, UniPC, etc.) restent gérés
# dans les méthodes (DiffPath, SLIDPC, etc.) qui utiliseront cette API.
# """

# # ---------------------------------------------------------------------
# # Localiser le repo EDM (NVLabs) pour importer dnnlib / torch_utils
# # ---------------------------------------------------------------------

# _here = os.path.abspath(os.path.dirname(__file__))
# _candidates = [
#     os.environ.get("EDM_REPO_DIR"),
#     os.path.abspath(os.path.join(_here, "../../..", "repos", "edm-main")),
#     os.path.abspath(os.path.join(_here, "../../..", "edm-main")),
#     os.path.abspath(os.path.join(_here, "..", "..", "repos", "edm-main")),
# ]
# for c in _candidates:
#     if c and os.path.isdir(c) and c not in sys.path:
#         sys.path.append(c)

# try:
#     import dnnlib  # type: ignore
# except Exception:
#     dnnlib = None


# class EDMAdapter:
#     """
#     Adaptateur unifié pour EDM NVLabs.

#     Paramètres attendus (comme dans le YAML) :
#       - checkpoint_path : chemin vers le .pkl NVLabs (edm-cifar10-32x32-*.pkl, etc.)
#       - prediction_type : "x0" (par défaut) ; "epsilon"/"v" possible mais moins utile ici
#       - in_channels     : canaux image (3 pour RGB, 1 pour grayscale)
#       - data_range      : tuple (lo, hi) pour info ([-1,1] en général)
#       - device          : "cuda:0", "cpu", etc.
#       - edm_repo_dir    : chemin explicite du repo edm-main (optionnel)
#     """

#     def __init__(
#         self,
#         checkpoint_path: str,
#         prediction_type: str = "x0",
#         in_channels: int = 3,
#         data_range=(-1.0, 1.0),
#         device=None,
#         edm_repo_dir: str = None,
#     ):
#         # --------------------- Device ---------------------
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         elif isinstance(device, torch.device):
#             self.device = device
#         else:
#             self.device = torch.device(device)
#         self.prediction_type = str(prediction_type)
#         self.in_channels = int(in_channels)
#         self.data_lo, self.data_hi = float(data_range[0]), float(data_range[1])
#         # Flag pour les méthodes (SLIDPC, etc.)
#         self.is_edm = True
#         # Schedule sigma (rempli à la demande)
#         self._sigma_schedule = None

#         # --------------------- Repo EDM ---------------------
#         if edm_repo_dir is not None and os.path.isdir(edm_repo_dir) and edm_repo_dir not in sys.path:
#             sys.path.insert(0, edm_repo_dir)

#         global dnnlib
#         if dnnlib is None:
#             try:
#                 import dnnlib as _dnnlib  # type: ignore
#                 dnnlib = _dnnlib
#             except Exception as e:
#                 raise RuntimeError(
#                     "[EDMAdapter] Impossible d'importer dnnlib. "
#                     "Vérifie que le repo 'edm-main' est présent (EDM_REPO_DIR ou repos/edm-main)."
#                 ) from e

#         # --------------------- Charger checkpoint ---------------------
#         ckpt_path = os.path.abspath(checkpoint_path)
#         if not os.path.exists(ckpt_path) and not ckpt_path.startswith("http"):
#             raise FileNotFoundError(f"[EDMAdapter] Checkpoint introuvable: {ckpt_path}")
#         self.checkpoint_path = ckpt_path

#         ext = os.path.splitext(ckpt_path)[1].lower()
#         if ext in [".pkl", ".pickle"]:
#             # Format NVLabs standard
#             with dnnlib.util.open_url(ckpt_path, verbose=False) as f:
#                 obj = pickle.load(f)

#             if isinstance(obj, dict) and "ema" in obj:
#                 net = obj["ema"]
#             elif hasattr(obj, "forward"):
#                 net = obj
#             else:
#                 raise RuntimeError(
#                     "[EDMAdapter] Checkpoint .pkl non reconnu (pas de clé 'ema' ni module nn.Module)."
#                 )
#             self.model = net.to(self.device).eval()
#         else:
#             # .pt / .pth contenant directement un nn.Module
#             try:
#                 ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
#             except TypeError:
#                 ckpt = torch.load(ckpt_path, map_location=self.device)
#             if isinstance(ckpt, torch.nn.Module):
#                 self.model = ckpt.to(self.device).eval()
#             elif isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), torch.nn.Module):
#                 self.model = ckpt["model"].to(self.device).eval()
#             else:
#                 raise RuntimeError(
#                     "[EDMAdapter] Checkpoint .pt/.pth non supporté : on dirait un state_dict pur. "
#                     "Instancie l'archi EDM depuis le repo puis charge le state_dict en amont."
#                 )

#         for p in self.model.parameters():
#             p.requires_grad_(False)

#         # Quelques méta-infos utiles
#         self.img_resolution = getattr(self.model, "img_resolution", None)
#         self.img_channels = getattr(self.model, "img_channels", None)
#         self.sigma_min = float(getattr(self.model, "sigma_min", 0.0))
#         self.sigma_max = float(getattr(self.model, "sigma_max", 0.0))

#         # Log de sanity
#         name = os.path.basename(ckpt_path)
#         print(
#             f"[EDMAdapter] Loaded EDM network '{name}' "
#             f"on device={self.device} (class={self.model.__class__.__name__})"
#         )
#         print(
#             f"[EDMAdapter] img_resolution={self.img_resolution}, img_channels={self.img_channels}, "
#             f"sigma_min={self.sigma_min:.4g}, sigma_max={self.sigma_max:.4g}"
#         )

#     # -----------------------------------------------------------------
#     # Helpers internes
#     # -----------------------------------------------------------------
#     def _as_sigma_tensor(self, sigma, x: torch.Tensor) -> torch.Tensor:
#         if not torch.is_tensor(sigma):
#             sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
#         if sigma.ndim == 0:
#             sigma = sigma.view(1)
#         return sigma

#     # -----------------------------------------------------------------
#     # API : denoise / score_src
#     # -----------------------------------------------------------------
#     @torch.no_grad()
#     def denoise(self, x: torch.Tensor, sigma):
#         """
#         Retourne x0_hat depuis le réseau EDM au niveau de bruit sigma.

#         Convention NVLabs EDM :
#           denoised = net(x, sigma, class_labels)
#           -> D(x, sigma) ~ x débruité (préconditionné)

#         Ici, on l'interprète comme un estimateur de x0 pour construire le score.
#         """
#         x = x.to(self.device)
#         sigma = self._as_sigma_tensor(sigma, x)

#         # broadcast sigma -> [B,1,1,1]
#         if sigma.ndim == 1:
#             sigma_in = sigma.view(-1, 1, 1, 1)
#         else:
#             sigma_in = sigma

#         # Réseau EDM *préconditionné* : net(x, sigma) -> "denoised"
#         out = self.model(x, sigma_in, class_labels=None)

#         # Cas standard : le réseau renvoie déjà un x0_hat-like
#         if self.prediction_type == "x0":
#             return out

#         # Cas "epsilon" ou "v" (peu utilisé ici, fallback simple)
#         if self.prediction_type in ("epsilon", "v"):
#             return x - sigma_in * out

#         raise ValueError(f"[EDMAdapter] Unknown prediction_type={self.prediction_type!r}")

#     def score_src(self, x: torch.Tensor, sigma, allow_grad: bool = False) -> torch.Tensor:
#         """
#         Approximation du score s(x, sigma).

#         On prend la convention simple :
#             s(x, sigma) = (x0_hat - x) / sigma
#         Le signe et la normalisation exacte importent peu pour GEPC (on regarde
#         surtout ∥s∥^2 et des gaps ID vs OOD).
#         """
#         x = x.to(self.device)
#         sigma = self._as_sigma_tensor(sigma, x)

#         if allow_grad:
#             x0_hat = self.denoise(x, sigma)
#         else:
#             with torch.no_grad():
#                 x0_hat = self.denoise(x, sigma)

#         sig = sigma
#         # broadcast sigma sur toutes les dims de x
#         while sig.ndim < x.ndim:
#             sig = sig.view(-1, *([1] * (x.ndim - 1)))

#         s = (x0_hat - x) / (sig + 1e-8)
#         return s

#     # -----------------------------------------------------------------
#     # Schedule en sigma pour index discret t_idx
#     # -----------------------------------------------------------------
#     def _ensure_sigma_schedule(self):
#         """
#         Construit un schedule en sigma de type Karras à partir de sampler_cfg
#         (si présent) ou des bornes du réseau EDM.
#         """
#         if getattr(self, "_sigma_schedule", None) is not None:
#             return

#         cfg = getattr(self, "sampler_cfg", {}) or {}
#         steps = int(cfg.get("steps", 32))
#         rho = float(cfg.get("rho", 7.0))

#         # Bornes sigma : sampler > réseau > fallback
#         sigma_min_cfg = cfg.get("sigma_min", None)
#         sigma_max_cfg = cfg.get("sigma_max", None)

#         if sigma_min_cfg is not None:
#             sigma_min = float(sigma_min_cfg)
#         else:
#             sigma_min = self.sigma_min if self.sigma_min > 0 else 0.002

#         if sigma_max_cfg is not None:
#             sigma_max = float(sigma_max_cfg)
#         else:
#             sigma_max = self.sigma_max if self.sigma_max > 0 else 80.0

#         ramp = torch.linspace(0.0, 1.0, steps, device=self.device, dtype=torch.float32)
#         min_inv = sigma_min ** (1.0 / rho)
#         max_inv = sigma_max ** (1.0 / rho)
#         # décroissant de sigma_max -> sigma_min
#         sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
#         self._sigma_schedule = sigmas

#     def sigma_t(self, t_idx: int) -> torch.Tensor:
#         """
#         Retourne sigma_t (scalaire) pour un index entier t_idx, selon
#         le schedule construit dans _ensure_sigma_schedule.
#         """
#         self._ensure_sigma_schedule()
#         if self._sigma_schedule is None:
#             # fallback ultra simple
#             return torch.tensor(self.sigma_min, device=self.device, dtype=torch.float32)
#         t = int(t_idx)
#         t = max(0, min(t, self._sigma_schedule.shape[0] - 1))
#         return self._sigma_schedule[t]

#     def get_sigma_schedule(self) -> torch.Tensor:
#         """
#         Schedule complet des sigmas utilisés par sigma_t (shape [T]).
#         """
#         self._ensure_sigma_schedule()
#         return self._sigma_schedule
