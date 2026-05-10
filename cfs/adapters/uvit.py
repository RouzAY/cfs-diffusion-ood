from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Optional

import numpy as np
import torch


# ============================================================
# U-ViT utilities
# ============================================================

def _load_config_from_py(config_path: str):
    config_path = os.path.abspath(config_path)
    spec = importlib.util.spec_from_file_location("uvit_external_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import U-ViT config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_config"):
        raise RuntimeError(f"U-ViT config {config_path} has no get_config().")
    return mod.get_config()


def _config_to_dict(x: Any) -> dict:
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    if isinstance(x, dict):
        return dict(x)
    return dict(x)


def _vp_sde_alphas_cumprod(T: int = 1000, beta_min: float = 0.1, beta_max: float = 20.0) -> np.ndarray:
    """
    U-ViT's pixel-space eval.py wraps the network in sde.VPSDE(beta_min=0.1, beta_max=20)
    and calls nnet(xt, t * 999). We therefore expose a discrete t=0..999 grid whose
    alpha_bar matches the continuous VP-SDE marginal at tau=t/(T-1).
    """
    T = int(T)
    tau = np.linspace(0.0, 1.0, T, dtype=np.float64)
    integral = beta_min * tau + 0.5 * (beta_max - beta_min) * tau * tau
    ab = np.exp(-integral)
    return np.clip(ab, 1e-12, 1.0 - 1e-12)


class UViTForwardWrapper(torch.nn.Module):
    """Wrap official U-ViT so MBE can call model(x, t)."""

    def __init__(self, inner: torch.nn.Module, class_label: Optional[int] = None):
        super().__init__()
        self.inner = inner
        self.class_label = None if class_label is None else int(class_label)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B = int(x.shape[0])
        if getattr(self.inner, "num_classes", -1) is not None and int(getattr(self.inner, "num_classes", -1)) > 0:
            # Conditional U-ViT checkpoints require a label. For OOD sanity checks,
            # prefer uncond CIFAR/CelebA checkpoints. If a conditional checkpoint is used,
            # pass class_label explicitly and treat it as a fixed-condition probe.
            if self.class_label is None:
                raise ValueError(
                    "This U-ViT checkpoint is conditional but class_label=None. "
                    "For CFS sanity checks use an unconditional checkpoint, or set a fixed class_label."
                )
            y = torch.full((B,), self.class_label, device=x.device, dtype=torch.long)
            return self.inner(x, t, y=y)
        return self.inner(x, t)


class UViTAdapter:
    """
    Adapter for the official U-ViT repository.

    Recommended for the NeurIPS appendix sanity check: use the unconditional
    CIFAR-10 or CelebA64 pixel-space checkpoints first. This avoids VAE/latent
    complications and directly tests whether sparse CFS snapshots transfer beyond
    convolutional U-Nets.

    Expected YAML fields:
      repo_dir: path/to/U-ViT-main
      config_path: path/to/U-ViT-main/configs/cifar10_uvit_small.py
      nnet_path: path/to/cifar10_uvit_small.pth
      class_label: null   # only needed for conditional checkpoints
    """

    is_edm = False
    is_transformer_diffusion = True
    # model is UViTForwardWrapper(inner=official UViT); named modules are inner.in_blocks.*, etc.
    block_regex = r"^(inner\.)?(in_blocks\.\d+|mid_block|out_blocks\.\d+)$"
    num_special_tokens = -1

    def __init__(
        self,
        repo_dir: str,
        config_path: str,
        nnet_path: str,
        device: str | torch.device = "cuda",
        class_label: Optional[int] = None,
        T: int = 1000,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        self.repo_dir = os.path.abspath(repo_dir)
        self.config_path = os.path.abspath(config_path)
        self.nnet_path = os.path.abspath(nnet_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.T = int(T)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        config = _load_config_from_py(self.config_path)
        self.config = config
        nnet_cfg = _config_to_dict(config.nnet)

        # Official repo utility.
        import utils as uvit_utils  # type: ignore

        raw_model = uvit_utils.get_nnet(**nnet_cfg)
        state = torch.load(self.nnet_path, map_location="cpu")
        if isinstance(state, dict) and "nnet_ema" in state:
            state = state["nnet_ema"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        raw_model.load_state_dict(state, strict=True)
        raw_model.eval().to(self.device)

        self.raw_model = raw_model
        self.model = UViTForwardWrapper(raw_model, class_label=class_label).eval().to(self.device)

        ab = _vp_sde_alphas_cumprod(T=self.T, beta_min=self.beta_min, beta_max=self.beta_max)
        self.alphas_cumprod = ab
        self.sqrt_ab = torch.tensor(np.sqrt(ab), device=self.device, dtype=torch.float32)
        self.sqrt_one_minus_ab = torch.tensor(np.sqrt(1.0 - ab), device=self.device, dtype=torch.float32)

        self.pred = str(getattr(config, "pred", "noise_pred"))
        self.image_size = int(nnet_cfg.get("img_size", 0))
        self.patch_size = int(nnet_cfg.get("patch_size", 0))
        self.num_special_tokens = int(getattr(raw_model, "extras", 0))

    def make_condition(self, t: int, batch_size: int, device: torch.device) -> torch.Tensor:
        # U-ViT expects timesteps in the same 0..999 scale used by ScoreModel.predict.
        return torch.full((int(batch_size),), int(t), device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward_model(self, x: torch.Tensor, cond: torch.Tensor):
        x = x.to(self.device, dtype=torch.float32)
        cond = cond.to(self.device)
        return self.model(x, cond)
