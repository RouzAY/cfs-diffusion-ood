# dtd/adapters/improved.py
# -*- coding: utf-8 -*-
"""
Adapter pour le repo OpenAI 'improved-diffusion'.

Objectifs:
  - Localiser automatiquement le repo improved-diffusion (IMPROVED_DIFFUSION_DIR
    ou quelques chemins candidats type repos/improved-diffusion).
  - Créer (model, diffusion) à partir de:
        * args.image_size        -> taille backbone (UNet)
        * args.improved_args     -> dict d'override (num_channels, etc.)
        * defaults de model_and_diffusion_defaults()
        * ET SURTOUT, pour les checkpoints officiels:
              - cifar10_uncond_50M_500K.pt
              - imagenet64_uncond_100M_1500K.pt
              - lsun_uncond_100M_2400K_bs64.pt
          on applique EXACTEMENT les MODEL_FLAGS / DIFFUSION_FLAGS
          du README OpenAI.
  - Détecter automatiquement learn_sigma à partir du checkpoint (3 vs 6 canaux).
  - Corriger la confusion classique num_channels (largeur UNet) vs in_channels.
  - Exposer:
        * self.model                UNet eps-predictor
        * self.diffusion            GaussianDiffusion
        * self.alphas_cumprod       [T]
        * self.sqrt_ab              sqrt(alpha_bar_t)
        * self.sqrt_one_minus_ab    sqrt(1 - alpha_bar_t)
        * sigma_t(t), score_from_eps(eps, t)
        * ddim_inversion_eps(x0)    (liste des eps_t, x_t) pour SLIDPC/GEPC
"""

import os
import sys
from typing import Any, Dict

import numpy as np
import torch

# ------------------------------------------------------------------
# Localiser le repo improved-diffusion et l'ajouter au PYTHONPATH
# ------------------------------------------------------------------
_here = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    os.environ.get("IMPROVED_DIFFUSION_DIR"),
    os.path.abspath(os.path.join(_here, "../../..", "repos", "improved-diffusion")),
    os.path.abspath(os.path.join(_here, "../../..", "improved-diffusion")),
    os.path.abspath(os.path.join(_here, "..", "..", "repos", "improved-diffusion")),
]
for c in _candidates:
    if c and os.path.isdir(c) and c not in sys.path:
        # priorité haute au repo trouvé
        sys.path.insert(0, c)

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


# ------------------------------------------------------------------
# Helpers internes
# ------------------------------------------------------------------

def _parse_mult_list(s):
    if s is None:
        return [1, 2, 3, 4]
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _detect_learn_sigma(sd: Dict[str, torch.Tensor]) -> bool | None:
    """
    Essaye de deviner si le UNet a été entraîné avec learn_sigma:
      - 3 canaux -> eps uniquement
      - 6 canaux -> (eps, sigma) concat
    """
    # cas le plus fréquent: module "out.2.weight"
    for k, v in sd.items():
        if k.endswith("out.2.weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
            return v.shape[0] == 6

    # fallback: chercher dans "out_layers"
    for k, v in sd.items():
        if "out_layers" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
            if v.shape[0] in (3, 6):
                return v.shape[0] == 6

    return None


def _validate_width(eff: Dict[str, Any]) -> None:
    """
    Assure que toutes les largeurs de blocs UNet sont divisibles par 32 (GroupNorm32).
    """
    base = int(eff.get("num_channels", 128))
    mults = _parse_mult_list(eff.get("channel_mult", "1,2,3,4"))
    bad = [base * m for m in mults if (base * m) % 32 != 0]
    if bad:
        raise ValueError(
            f"[ImprovedAdapter] Invalid UNet width: num_channels={base} with channel_mult={mults} "
            f"produces channels {bad} not divisible by 32 (GroupNorm32). "
            f"Use a base width divisible by 32 (e.g., 128)."
        )


# ------------------------------------------------------------------
# Adapter principal
# ------------------------------------------------------------------

class ImprovedDiffusionAdapter:
    """
    Adapter improved-diffusion utilisé par SLIDPC/GEPC.

    Args attendus dans `args`:
      - model_path: chemin vers le checkpoint .pt
      - image_size: taille cible du backbone (UNet), ex: 32 / 64 / 256
      - improved_args: dict facultatif pour surcharger les defaults
          (num_channels, num_res_blocks, channel_mult, etc.)

    Pour les ckpts officiels OpenAI:
      * cifar10_uncond_50M_500K.pt
        MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
        DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

      * imagenet64_uncond_100M_1500K.pt
        MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
        DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

      * lsun_uncond_100M_2400K_bs64.pt
        MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 "
                    "--learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
        DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear "
                        "--rescale_learned_sigmas False --rescale_timesteps False "
                        "--use_scale_shift_norm False"
    """

    def __init__(self, args):
        self.args = args

        # ----------------- setup distrib + device -----------------
        try:
            dist_util.setup_dist()
        except TypeError:
            try:
                dev_id = getattr(args, "device", 0)
                dist_util.setup_dist(dev_id)
            except TypeError:
                dist_util.setup_dist()

        self.device = dist_util.dev()

        # ----------------- charger le checkpoint ------------------
        ckpt_path = str(getattr(self.args, "model_path", ""))
        if not ckpt_path:
            raise ValueError("[ImprovedAdapter] args.model_path doit être spécifié.")

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
        else:
            # compatibilité avec la fonction utilitaire du repo
            state = dist_util.load_state_dict(ckpt_path, map_location="cpu")

        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        forced_learn_sigma = _detect_learn_sigma(state_dict)

        # ----------------- config effective (eff) -----------------
        cfg_defaults = model_and_diffusion_defaults()

        # 1) args.improved_args (YAML) si présent
        user: Dict[str, Any] = {}
        if hasattr(self.args, "improved_args") and self.args.improved_args:
            user = dict(self.args.improved_args)

        # 2) alias éventuel: model_channels -> num_channels
        if "model_channels" in user and "num_channels" not in user:
            user["num_channels"] = user.pop("model_channels")

        # 3) détecter preset *officiel* en fonction du nom de ckpt
        ckpt_name = os.path.basename(ckpt_path).lower()
        preset: Dict[str, Any] = {}

        # --- CIFAR-10 uncond 32x32 ---
        if "cifar10_uncond_50m_500k" in ckpt_name or (
            "cifar10" in ckpt_name and "uncond" in ckpt_name and "50m_500k" in ckpt_name
        ):
            preset = {
                # MODEL_FLAGS
                "image_size": 32,
                "num_channels": 128,
                "num_res_blocks": 3,
                "learn_sigma": True,
                "dropout": 0.3,
                # DIFFUSION_FLAGS
                "diffusion_steps": 4000,
                "noise_schedule": "cosine",
            }
            print("[ImprovedAdapter] Detected official CIFAR-10 uncond checkpoint; using OpenAI MODEL/DIFFUSION flags.")

        # --- ImageNet-64 uncond 64x64 ---
        elif "imagenet64_uncond_100m_1500k" in ckpt_name or (
            "imagenet64" in ckpt_name and "uncond" in ckpt_name
        ):
            preset = {
                "image_size": 64,
                "num_channels": 128,
                "num_res_blocks": 3,
                "learn_sigma": True,
                "diffusion_steps": 4000,
                "noise_schedule": "cosine",
            }
            print("[ImprovedAdapter] Detected official ImageNet-64 uncond checkpoint; using OpenAI MODEL/DIFFUSION flags.")

        # --- LSUN 256x256 uncond ---
        elif "lsun_uncond_100m_2400k" in ckpt_name or "lsun" in ckpt_name:
            preset = {
                "image_size": 256,
                "num_channels": 128,
                "num_res_blocks": 2,
                "num_heads": 1,
                "learn_sigma": True,
                "use_scale_shift_norm": False,
                "attention_resolutions": "16",
                "diffusion_steps": 1000,
                "noise_schedule": "linear",
                "rescale_learned_sigmas": False,
                "rescale_timesteps": False,
            }
            print("[ImprovedAdapter] Detected LSUN 256x256 uncond checkpoint; using OpenAI MODEL/DIFFUSION flags.")

        else:
            # Fallback: preset léger selon image_size si on connaît, sinon rien
            backbone_size = getattr(self.args, "image_size", None)
            if backbone_size is None:
                backbone_size = user.get("image_size", cfg_defaults["image_size"])
            backbone_size = int(backbone_size)

            if backbone_size == 32:
                # preset "raisonnable" style CIFAR/CelebA; pourra être override par improved_args.
                preset = {
                    "image_size": 32,
                    "num_channels": 128,
                    "num_res_blocks": 3,
                    "learn_sigma": True,
                    "dropout": 0.3,
                    "diffusion_steps": 4000,
                    "noise_schedule": "cosine",
                }
            elif backbone_size == 64:
                preset = {
                    "image_size": 64,
                    "num_channels": 128,
                    "num_res_blocks": 3,
                    "learn_sigma": True,
                    "diffusion_steps": 4000,
                    "noise_schedule": "cosine",
                }
            elif backbone_size in (224, 256):
                preset = {
                    "image_size": backbone_size,
                    "num_channels": 128,
                    "num_res_blocks": 2,
                    "learn_sigma": True,
                    "diffusion_steps": 1000,
                    "noise_schedule": "linear",
                }
            # sinon: pas de preset → seuls defaults + improved_args s'appliquent.

        # 4) fusion defaults <- preset <- user
        eff: Dict[str, Any] = dict(cfg_defaults)
        eff.update(preset)
        eff.update(user)

        # 5) override ultime: args.image_size top-level si non None
        if getattr(self.args, "image_size", None) is not None:
            eff["image_size"] = int(self.args.image_size)

        # 6) heuristique anti-confusion num_channels vs in_channels
        if int(eff.get("num_channels", 128)) <= 4:
            nc = int(eff["num_channels"])
            if "in_channels" not in eff:
                eff["in_channels"] = nc
            # largeur UNet par défaut si pas de preset
            eff["num_channels"] = int(preset.get("num_channels", 128) if preset else 128)
            print(
                f"[ImprovedAdapter][fix] Detected num_channels={nc} (likely image channels). "
                f"Using in_channels={eff['in_channels']} and UNet width num_channels={eff['num_channels']}."
            )

        # 7) learn_sigma forcé par le checkpoint si détecté
        if forced_learn_sigma is not None:
            eff["learn_sigma"] = bool(forced_learn_sigma)

        # 8) validation largeur UNet / GroupNorm
        _validate_width(eff)

        # 9) log de la config effective
        def _gv(k, d=None):
            return eff.get(k, d)

        print(
            "[ImprovedAdapter] Effective UNet config -> "
            f"image_size={_gv('image_size')}, width(num_channels)={_gv('num_channels')}, "
            f"num_res_blocks={_gv('num_res_blocks')}, channel_mult={_gv('channel_mult')}, "
            f"attn={_gv('attention_resolutions')}, heads={_gv('num_heads')}, "
            f"learn_sigma={_gv('learn_sigma')}, class_cond={_gv('class_cond')}"
        )

        # ----------------- création model + diffusion --------------
        # On réutilise args_to_dict pour s'aligner avec le code origine.
        cfg_obj = type("Config", (object,), eff)()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(cfg_obj, model_and_diffusion_defaults().keys())
        )

        # ----------------- chargement des poids --------------------
        # Pour les ckpts officiels: on veut 0 missing / 0 unexpected,
        # donc on met strict=True. Pour d'autres ckpts, si ça casse,
        # il faudra ajuster improved_args.
        r = model.load_state_dict(state_dict, strict=True)
        miss, unexp = len(r.missing_keys), len(r.unexpected_keys)
        print(f"[ImprovedAdapter] state_dict loaded. missing={miss} unexpected={unexp}")
        if miss > 0:
            print("[ImprovedAdapter][warn] Missing keys sample (first 5):", r.missing_keys[:5])
        if unexp > 0:
            print("[ImprovedAdapter][warn] Unexpected keys sample (first 5):", r.unexpected_keys[:5])

        self.model = model.to(self.device).eval()
        self.diffusion = diffusion

        # ----------------- cache du schedule -----------------------
        self.betas = np.asarray(self.diffusion.betas, dtype=np.float32)
        alphas = 1.0 - self.betas
        a_bar = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = torch.from_numpy(a_bar).float().to(self.device)
        self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alphas_cumprod)

        raw_ut = getattr(self.diffusion, "use_timesteps", None)
        if raw_ut is None:
            self.use_timesteps = np.arange(len(self.betas), dtype=np.int64)
        else:
            self.use_timesteps = np.array(sorted(list(raw_ut)), dtype=np.int64)
        self.inner_steps = np.arange(len(self.betas), dtype=np.int64)

    # ------------------------------------------------------------------
    # Helpers SLIDPC
    # ------------------------------------------------------------------

    def n_steps(self) -> int:
        return len(self.betas)

    def sigma_t(self, t_idx: int) -> torch.Tensor:
        """
        sigma_t = sqrt(1 - alpha_bar_t) (renvoyé comme Tensor[ ] sur self.device)
        """
        return self.sqrt_one_minus_ab[int(t_idx)]

    def score_from_eps(self, eps: torch.Tensor, t_idx: int) -> torch.Tensor:
        """
        Convertit eps en score (grad log p) en divisant par sigma_t.
        """
        sigma = self.sigma_t(t_idx).view(1, 1, 1, 1)
        return -eps / (sigma + 1e-12)

    # ------------------------------ inversion DDIM ------------------------------

    @torch.no_grad()
    def _ddim_reverse_loop_builtin(self, x0: torch.Tensor):
        """
        Utilise directement diffusion.ddim_reverse_sample_loop si disponible
        (API custom de certains forks).
        """
        try:
            out = self.diffusion.ddim_reverse_sample_loop(
                self.model,
                x0.shape,
                x0,
                clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
                model_kwargs=None,
                return_eps=True,
                return_xt=True,
            )
            eps_list, xt_list = out
        except TypeError:
            out = self.diffusion.ddim_reverse_sample_loop(
                self.model,
                x0.shape,
                x0,
                clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
                model_kwargs=None,
                return_eps=True,
            )
            eps_list, xt_list = out, None
        return eps_list, xt_list

    @torch.no_grad()
    def _ddim_reverse_loop_stepwise(self, x0: torch.Tensor):
        """
        Fallback step-by-step si le repo expose diffusion.ddim_reverse_sample.
        """
        if not hasattr(self.diffusion, "ddim_reverse_sample"):
            return None

        dev = x0.device
        B = x0.shape[0]
        T = self.n_steps()
        eps_list, xt_list, x = [], [], x0

        for i in range(T):
            t_scalar = int(self.inner_steps[i])
            t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)
            x_in = x

            out = self.diffusion.ddim_reverse_sample(
                self.model,
                x_in,
                t,
                clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
            )
            if isinstance(out, dict):
                x = out.get("sample", x_in)
                pred_x0 = out.get("pred_xstart", None)
            elif isinstance(out, (tuple, list)):
                x = out[0]
                pred_x0 = out[1] if len(out) > 1 else None
            else:
                x = out
                pred_x0 = None

            if pred_x0 is None:
                eps_t = self.model(x_in, t)
            else:
                a_bar = self.alphas_cumprod[i].view(1, 1, 1, 1)
                denom = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
                eps_t = (x_in - torch.sqrt(a_bar) * pred_x0) / denom

            eps_list.append(eps_t)
            xt_list.append(x)

        return eps_list, xt_list

    @torch.no_grad()
    def _ddim_reverse_loop_eps_only(self, x0: torch.Tensor):
        """
        Fallback minimaliste: pas d'API DDIM fournie, on reconstruit
        (x_t, eps_t) en utilisant le forward noising avec eps_t = eps(x_t, t).
        """
        dev = x0.device
        B = x0.shape[0]
        T = self.n_steps()
        eps_list, xt_list, x = [], [], x0

        for i in range(T):
            t_scalar = int(self.inner_steps[i])
            t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)
            x_in = x

            eps_t = self.model(x_in, t)
            eps_list.append(eps_t)

            a_bar_next = self.alphas_cumprod[i].view(1, 1, 1, 1)
            x = torch.sqrt(a_bar_next) * x0 + torch.sqrt(1.0 - a_bar_next) * eps_t
            xt_list.append(x)

        return eps_list, xt_list

    @torch.no_grad()
    def ddim_inversion_eps(self, x0: torch.Tensor):
        """
        Interface unique pour SLIDPC: renvoie (eps_list, xt_list).
        """
        if hasattr(self.diffusion, "ddim_reverse_sample_loop"):
            return self._ddim_reverse_loop_builtin(x0)

        stepwise = self._ddim_reverse_loop_stepwise(x0)
        if stepwise is not None:
            return stepwise

        return self._ddim_reverse_loop_eps_only(x0)





# # dtd/adapters/improved.py
# # -*- coding: utf-8 -*-
# import os, sys, torch, numpy as np

# # --- localiser le repo improved-diffusion sur le PYTHONPATH ---
# _here = os.path.abspath(os.path.dirname(__file__))
# _candidates = [
#     os.environ.get("IMPROVED_DIFFUSION_DIR"),
#     os.path.abspath(os.path.join(_here, "../../..", "repos", "improved-diffusion")),
#     os.path.abspath(os.path.join(_here, "../../..", "improved-diffusion")),
#     os.path.abspath(os.path.join(_here, "..", "..", "repos", "improved-diffusion")),
# ]
# for c in _candidates:
#     if c and os.path.isdir(c) and c not in sys.path:
#         sys.path.append(c)

# from improved_diffusion import dist_util
# from improved_diffusion.script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
# )


# def _parse_mult_list(s):
#     if s is None:
#         return [1, 2, 3, 4]
#     if isinstance(s, (list, tuple)):
#         return [int(x) for x in s]
#     return [int(x.strip()) for x in str(s).split(",") if x.strip()]


# def _detect_learn_sigma(sd):
#     # cherche la conv finale et infère 3 vs 6 canaux
#     for k, v in sd.items():
#         if k.endswith("out.2.weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
#             return v.shape[0] == 6
#     for k, v in sd.items():
#         if "out_layers" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
#             if v.shape[0] in (3, 6):
#                 return v.shape[0] == 6
#     return None


# def _validate_width(eff):
#     """Assure que toutes les largeurs de blocs sont divisibles par 32 (GroupNorm32)."""
#     base = int(eff.get("num_channels", 128))
#     mults = _parse_mult_list(eff.get("channel_mult", "1,2,3,4"))
#     bad = [base * m for m in mults if (base * m) % 32 != 0]
#     if bad:
#         raise ValueError(
#             f"[Adapter] Invalid UNet width: num_channels={base} with channel_mult={mults} "
#             f"produces channels {bad} not divisible by 32 (GroupNorm32). "
#             f"Use a base width divisible by 32 (e.g., 128)."
#         )


# class ImprovedDiffusionAdapter:
#     """
#     Adapter OpenAI improved-diffusion
#       - crée model+diffusion avec des args *compatibles checkpoint*
#       - corrige l'erreur classique num_channels (UNet width) vs in_channels (image)
#       - force learn_sigma si déductible du ckpt (6 canaux => True)
#       - presets stables pour:
#           * imagenet64_uncond_100M_1500K.pt
#           * celeba32 (UNet 128w, 3 resblocks, mult 1,2,2,2)
#       - expose helpers & DDIM inversion (optionnel)
#     """
#     def __init__(self, args):
#         self.args = args

#         # setup_dist robuste
#         try:
#             dist_util.setup_dist()
#         except TypeError:
#             try:
#                 dist_util.setup_dist(getattr(args, "device", 0))
#             except TypeError:
#                 dist_util.setup_dist()

#         # ---- 0) Charger le state_dict en amont pour introspection learn_sigma
#         ckpt_path = str(getattr(self.args, "model_path", ""))
#         if os.path.exists(ckpt_path):
#             state = torch.load(ckpt_path, map_location="cpu")
#         else:
#             state = dist_util.load_state_dict(ckpt_path, map_location="cpu")
#         forced_learn_sigma = _detect_learn_sigma(state)

#         # ---- 1) Récupérer args utilisateur ("improved_args" style DiffPath)
#         cfg_defaults = model_and_diffusion_defaults()
#         user = {}
#         if hasattr(self.args, "improved_args") and self.args.improved_args:
#             user = dict(self.args.improved_args)

#         # # fallback : si l'utilisateur a mis des clés au top-level, on les reprend
#         # for k in [
#         #     "image_size","num_channels","num_res_blocks","channel_mult",
#         #     "attention_resolutions","num_heads","class_cond","dropout",
#         #     "learn_sigma","diffusion_steps","noise_schedule","clip_denoised",
#         #     "use_zero_module","resblock_updown","use_scale_shift_norm",
#         #     "rescale_timesteps","rescale_learned_sigmas"
#         # ]:
#         #     if hasattr(self.args, k) and getattr(self.args, k) is not None:
#         #         user[k] = getattr(self.args, k)


#         # normaliser alias: model_channels -> num_channels (UNet width)
#         if "model_channels" in user and "num_channels" not in user:
#             user["num_channels"] = user.pop("model_channels")

#         # ---- 2) Presets robustes selon ckpt (si détectable)
#         name = os.path.basename(ckpt_path).lower()
#         preset = {}
#         if "imagenet64_uncond_100m_1500k" in name:
#             preset = {
#                 "image_size": 64,
#                 "num_channels": 128,       # width
#                 "num_res_blocks": 3,
#                 "channel_mult": "1,2,3,4",
#                 "attention_resolutions": "8,16",
#                 "num_heads": 4,
#                 "dropout": 0.0,
#                 "class_cond": False,
#                 "resblock_updown": True,
#                 "use_scale_shift_norm": True,
#                 "learn_sigma": True,       # FORCÉ (ckpt 6 canaux)
#                 "use_zero_module": True,
#                 "clip_denoised": True,
#                 "diffusion_steps": 4000,
#                 "noise_schedule": "cosine",
#             }
#         elif int(user.get("image_size", self.args.__dict__.get("image_size", 0)) or 0) == 32:
#             # CelebA32 par défaut (style DiffPath)
#             preset = {
#                 "image_size": 32,
#                 "num_channels": 128,
#                 "num_res_blocks": 3,
#                 "channel_mult": "1,2,2,2",
#                 "attention_resolutions": "8",
#                 "num_heads": 4,
#                 "dropout": 0.3,
#                 "class_cond": False,
#                 "resblock_updown": True,
#                 "use_scale_shift_norm": True,
#                 "learn_sigma": True,
#                 "use_zero_module": True,
#                 "clip_denoised": True,
#                 "diffusion_steps": 4000,
#                 "noise_schedule": "cosine",
#             }

#         # ---- 3) Fusion preset <- user <- CLI/args
#         eff = dict(cfg_defaults)
#         eff.update(preset)
#         eff.update(user)

#         # CLI override
#         if hasattr(self.args, "image_size") and self.args.image_size:
#             eff["image_size"] = int(self.args.image_size)

#         # Heuristique anti-confusion: si num_channels est tout petit (1–4),
#         # on considère que l'utilisateur a mis "image channels" ici.
#         # On le déplace vers in_channels, et on remet la width à 128 (ou preset).
#         if int(eff.get("num_channels", 128)) <= 4:
#             nc = int(eff["num_channels"])
#             if "in_channels" not in eff:
#                 eff["in_channels"] = nc
#             eff["num_channels"] = int(preset.get("num_channels", 128))
#             print(f"[Adapter][fix] Detected num_channels={nc} (likely image channels). "
#                   f"Using in_channels={eff['in_channels']} and UNet width num_channels={eff['num_channels']}.")

#         # appliquer détection learn_sigma (ckpt): priorité ultime
#         if forced_learn_sigma is not None:
#             eff["learn_sigma"] = bool(forced_learn_sigma)

#         # Validation width/GN
#         _validate_width(eff)

#         # log résumé
#         def _gv(k, d=None): return eff.get(k, d)
#         print(
#             "[Adapter] Effective UNet config -> "
#             f"image_size={_gv('image_size')}, width(num_channels)={_gv('num_channels')}, "
#             f"num_res_blocks={_gv('num_res_blocks')}, channel_mult={_gv('channel_mult')}, "
#             f"attn={_gv('attention_resolutions')}, heads={_gv('num_heads')}, "
#             f"learn_sigma={_gv('learn_sigma')}, class_cond={_gv('class_cond')}"
#         )

#         # ---- 4) Créer model+diffusion
#         model, diffusion = create_model_and_diffusion(
#             **args_to_dict(type("X",(object,),eff)(), model_and_diffusion_defaults().keys())
#         )

#         # ---- 5) Charger weights
#         r = model.load_state_dict(state, strict=False)
#         miss, unexp = len(r.missing_keys), len(r.unexpected_keys)
#         print(f"[Adapter] state_dict loaded. missing={miss} unexpected={unexp}")
#         if miss > 0:
#             print("[Adapter][warn] Missing keys sample (first 5):", r.missing_keys[:5])
#         if unexp > 0:
#             print("[Adapter][warn] Unexpected keys sample (first 5):", r.unexpected_keys[:5])

#         self.model, self.diffusion = model, diffusion
#         self.model.to(dist_util.dev()).eval()

#         # ---- 6) Cache schedule
#         self.betas = np.asarray(self.diffusion.betas, dtype=np.float32)
#         alphas = 1.0 - self.betas
#         a_bar = np.cumprod(alphas, axis=0)
#         self.alphas_cumprod = torch.from_numpy(a_bar).float().to(dist_util.dev())
#         self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alphas_cumprod)

#         raw_ut = getattr(self.diffusion, "use_timesteps", None)
#         if raw_ut is None:
#             self.use_timesteps = np.arange(len(self.betas), dtype=np.int64)
#         else:
#             self.use_timesteps = np.array(sorted(list(raw_ut)), dtype=np.int64)

#         self.inner_steps = np.arange(len(self.betas), dtype=np.int64)

#     # ------------------------------ helpers ------------------------------
#     def n_steps(self) -> int:
#         return len(self.betas)

#     def sigma_t(self, t_idx: int):
#         return self.sqrt_one_minus_ab[t_idx]

#     def score_from_eps(self, eps: torch.Tensor, t_idx: int):
#         sigma = self.sigma_t(t_idx).view(1, 1, 1, 1)
#         return -eps / (sigma + 1e-12)

#     # ------------------------------ inversion DDIM ------------------------------
#     @torch.no_grad()
#     def _ddim_reverse_loop_builtin(self, x0: torch.Tensor):
#         try:
#             out = self.diffusion.ddim_reverse_sample_loop(
#                 self.model, x0.shape, x0,
#                 clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
#                 model_kwargs=None, return_eps=True, return_xt=True,
#             )
#             eps_list, xt_list = out
#         except TypeError:
#             out = self.diffusion.ddim_reverse_sample_loop(
#                 self.model, x0.shape, x0,
#                 clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
#                 model_kwargs=None, return_eps=True,
#             )
#             eps_list, xt_list = out, None
#         return eps_list, xt_list

#     @torch.no_grad()
#     def _ddim_reverse_loop_stepwise(self, x0: torch.Tensor):
#         if not hasattr(self.diffusion, "ddim_reverse_sample"):
#             return None
#         dev = x0.device; B = x0.shape[0]; T = self.n_steps()
#         eps_list, xt_list, x = [], [], x0
#         for i in range(T):
#             t_scalar = int(self.inner_steps[i])
#             t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)
#             x_in = x
#             out = self.diffusion.ddim_reverse_sample(
#                 self.model, x_in, t, clip_denoised=bool(getattr(self.args, "clip_denoised", True)),
#             )
#             if isinstance(out, dict):
#                 x = out.get("sample", x_in); pred_x0 = out.get("pred_xstart", None)
#             elif isinstance(out, (tuple, list)):
#                 x = out[0]; pred_x0 = out[1] if len(out) > 1 else None
#             else:
#                 x = out; pred_x0 = None

#             if pred_x0 is None:
#                 eps_t = self.model(x_in, t)
#             else:
#                 a_bar = self.alphas_cumprod[i].view(1,1,1,1)
#                 denom = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
#                 eps_t = (x_in - torch.sqrt(a_bar) * pred_x0) / denom

#             eps_list.append(eps_t); xt_list.append(x)
#         return eps_list, xt_list

#     @torch.no_grad()
#     def _ddim_reverse_loop_eps_only(self, x0: torch.Tensor):
#         dev = x0.device; B = x0.shape[0]; T = self.n_steps()
#         eps_list, xt_list, x = [], [], x0
#         for i in range(T):
#             t_scalar = int(self.inner_steps[i])
#             t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)
#             x_in = x
#             eps_t = self.model(x_in, t)
#             eps_list.append(eps_t)
#             a_bar_next = self.alphas_cumprod[i].view(1,1,1,1)
#             x = torch.sqrt(a_bar_next) * x0 + torch.sqrt(1.0 - a_bar_next) * eps_t
#             xt_list.append(x)
#         return eps_list, xt_list

#     @torch.no_grad()
#     def ddim_inversion_eps(self, x0: torch.Tensor):
#         if hasattr(self.diffusion, "ddim_reverse_sample_loop"):
#             return self._ddim_reverse_loop_builtin(x0)
#         stepwise = self._ddim_reverse_loop_stepwise(x0)
#         if stepwise is not None:
#             return stepwise
#         return self._ddim_reverse_loop_eps_only(x0)







# # dtd/adapters/improved.py
# # -*- coding: utf-8 -*-
# import os
# import sys
# import torch
# import numpy as np

# # --- localiser le repo improved-diffusion sur le PYTHONPATH ---
# _here = os.path.abspath(os.path.dirname(__file__))
# _candidates = [
#     os.environ.get("IMPROVED_DIFFUSION_DIR"),
#     os.path.abspath(os.path.join(_here, "../../..", "repos", "improved-diffusion")),
#     os.path.abspath(os.path.join(_here, "../../..", "improved-diffusion")),
#     os.path.abspath(os.path.join(_here, "..", "..", "repos", "improved-diffusion")),
# ]
# for c in _candidates:
#     if c and os.path.isdir(c) and c not in sys.path:
#         sys.path.append(c)

# from improved_diffusion import dist_util
# from improved_diffusion.script_util import (
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     args_to_dict,
# )

# class ImprovedDiffusionAdapter:
#     """
#     Adapter for OpenAI improved-diffusion:
#       - load model + SpacedDiffusion
#       - expose DDIM inversion that returns eps per step (DiffPath-style)
#       - small helpers (n_steps, sigma_t, etc.)
#     """
#     def __init__(self, args):
#         self.args = args

#         # Device hint (compatible avec improved-diffusion vanilla)
#         if hasattr(args, "device"):
#             os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))

#         # setup_dist : tolérant selon les forks
#         try:
#             dist_util.setup_dist()
#         except TypeError:
#             try:
#                 dist_util.setup_dist(args.device if hasattr(args, "device") else 0)
#             except TypeError:
#                 dist_util.setup_dist()

#         # --- 0) Aplatir les overrides du YAML: improved_args -> self.args ---
#         # (doit être fait AVANT d'injecter les defaults)
#         if hasattr(self.args, "improved_args") and self.args.improved_args:
#             src = self.args.improved_args
#             # dict, SimpleNamespace ou autre
#             if isinstance(src, dict):
#                 items = src.items()
#             else:
#                 items = vars(src).items()
#             for k, v in items:
#                 setattr(self.args, k, v)


#         # --- IMPORTANT : injecter les defaults manquants AVANT create_model_and_diffusion ---
#         _defaults = model_and_diffusion_defaults()
#         missing = []
#         for k, v in _defaults.items():
#             if not hasattr(self.args, k):
#                 setattr(self.args, k, v)
#                 missing.append(k)
#         if missing:
#             print(f"[Adapter] Filled missing improved-diffusion args with defaults: {missing}")

#         # Respacing DDIM (après les defaults)
#         self.args.timestep_respacing = f"ddim{args.n_ddim_steps}"

#         # Création model + diffusion
#         self.model, self.diffusion = create_model_and_diffusion(
#             **args_to_dict(self.args, model_and_diffusion_defaults().keys())
#         )

#         # Chargement checkpoint
#         ckpt = self.args.model_path
#         print(f"[Adapter] Loading checkpoint: {ckpt}")
#         if os.path.exists(ckpt):
#             state = torch.load(ckpt, map_location="cpu")
#         else:
#             state = dist_util.load_state_dict(ckpt, map_location="cpu")
#         # strict=False pour tolérer de légères divergences de clés si besoin
#         self.model.load_state_dict(state, strict=False)
#         self.model.to(dist_util.dev())
#         self.model.eval()

#         # Cache schedule (SpacedDiffusion betas = chaîne respacée)
#         self.betas = np.asarray(self.diffusion.betas, dtype=np.float32)  # [T]
#         alphas = 1.0 - self.betas
#         a_bar = np.cumprod(alphas, axis=0)  # [T]
#         self.alphas_cumprod = torch.from_numpy(a_bar).float().to(dist_util.dev())  # [T]
#         self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alphas_cumprod)

#         raw_ut = getattr(self.diffusion, "use_timesteps", None)
#         if raw_ut is None:
#             self.use_timesteps = np.arange(len(self.betas), dtype=np.int64)
#         else:
#             if isinstance(raw_ut, (set, frozenset)):
#                 self.use_timesteps = np.array(sorted(list(raw_ut)), dtype=np.int64)
#             elif isinstance(raw_ut, list):
#                 self.use_timesteps = np.array(raw_ut, dtype=np.int64)
#             elif isinstance(raw_ut, np.ndarray):
#                 self.use_timesteps = raw_ut.astype(np.int64)
#             else:
#                 # objet itérable quelconque
#                 self.use_timesteps = np.array(list(raw_ut), dtype=np.int64)

#         # Indices internes 0..T-1 pour piloter SpacedDiffusion (à passer à .ddim_reverse_sample)
#         self.inner_steps = np.arange(len(self.betas), dtype=np.int64)

#         # # Indices de timestep utilisés par SpacedDiffusion (= indices d'origine)
#         # self.use_timesteps = getattr(self.diffusion, "use_timesteps", np.arange(len(self.betas)))

#     # ------------------------------ helpers ------------------------------
#     def n_steps(self) -> int:
#         """Nombre de pas DDIM de la chaîne respacée."""
#         return len(self.betas)

#     def sigma_t(self, t_idx: int):
#         """sigma(t) = sqrt(1 - alpha_bar_t) pour paramétrisation VP."""
#         return self.sqrt_one_minus_ab[t_idx]

#     def score_from_eps(self, eps: torch.Tensor, t_idx: int):
#         """s_theta ≈ - eps / sigma_t (VP-SDE)."""
#         sigma = self.sigma_t(t_idx).view(1, 1, 1, 1)
#         return -eps / (sigma + 1e-12)

#     # ------------------------------ inversion DDIM ------------------------------
#     @torch.no_grad()
#     def _ddim_reverse_loop_builtin(self, x0: torch.Tensor):
#         """
#         Utilise diffusion.ddim_reverse_sample_loop si disponible.
#         On tente return_eps/return_xt, sinon on fallback sur return_eps seul.
#         """
#         try:
#             out = self.diffusion.ddim_reverse_sample_loop(
#                 self.model,
#                 x0.shape,
#                 x0,
#                 clip_denoised=getattr(self.args, "clip_denoised", True),
#                 model_kwargs=None,
#                 return_eps=True,
#                 return_xt=True,
#             )
#             eps_list, xt_list = out
#         except TypeError:
#             out = self.diffusion.ddim_reverse_sample_loop(
#                 self.model,
#                 x0.shape,
#                 x0,
#                 clip_denoised=getattr(self.args, "clip_denoised", True),
#                 model_kwargs=None,
#                 return_eps=True,
#             )
#             eps_list, xt_list = out, None
#         return eps_list, xt_list

#     @torch.no_grad()
#     def _ddim_reverse_loop_stepwise(self, x0: torch.Tensor):
#         """
#         Fallback avec diffusion.ddim_reverse_sample (step unique) si loop absente.
#         Reconstitue eps_t via pred_xstart :
#           eps_t = (x_t - sqrt(ab_t) * pred_xstart) / sqrt(1 - ab_t)
#         """
#         if not hasattr(self.diffusion, "ddim_reverse_sample"):
#             return None

#         dev = x0.device
#         B = x0.shape[0]
#         T = self.n_steps()

#         eps_list, xt_list = [], []
#         x = x0

#         for i in range(T):
#             # t_scalar = int(self.use_timesteps[i])
#             t_scalar = int(self.inner_steps[i])  # indice interne 0..T-1
#             t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)

#             x_in = x
#             out = self.diffusion.ddim_reverse_sample(
#                 self.model,
#                 x_in,
#                 t,
#                 clip_denoised=getattr(self.args, "clip_denoised", True),
#             )
#             if isinstance(out, dict):
#                 x = out.get("sample", x_in)            # x_{t+1}
#                 pred_x0 = out.get("pred_xstart", None) # x0 prédite
#             elif isinstance(out, (tuple, list)):
#                 x = out[0]
#                 pred_x0 = out[1] if len(out) > 1 else None
#             else:
#                 x = out
#                 pred_x0 = None

#             if pred_x0 is None:
#                 # Mode EPS direct si pas de pred_xstart
#                 eps_t = self.model(x_in, t)
#             else:
#                 a_bar = self.alphas_cumprod[i].view(1, 1, 1, 1)
#                 denom = torch.sqrt(torch.clamp(1.0 - a_bar, min=1e-12))
#                 eps_t = (x_in - torch.sqrt(a_bar) * pred_x0) / denom

#             eps_list.append(eps_t)
#             xt_list.append(x)

#         return eps_list, xt_list

#     @torch.no_grad()
#     def _ddim_reverse_loop_eps_only(self, x0: torch.Tensor):
#         """
#         Dernier fallback : prédire eps_t directement et avancer de manière déterministe
#         via x_{t+1} = sqrt(ab_{t+1}) x0 + sqrt(1 - ab_{t+1}) eps_t.
#         """
#         dev = x0.device
#         B = x0.shape[0]
#         T = self.n_steps()

#         eps_list, xt_list = [], []
#         x = x0

#         for i in range(T):
#             # t_scalar = int(self.use_timesteps[i])
#             t_scalar = int(self.inner_steps[i])  # indice interne 0..T-1
#             t = torch.full((B,), t_scalar, device=dev, dtype=torch.long)

#             x_in = x
#             eps_t = self.model(x_in, t)
#             eps_list.append(eps_t)

#             a_bar_next = self.alphas_cumprod[i].view(1, 1, 1, 1)
#             x = torch.sqrt(a_bar_next) * x0 + torch.sqrt(1.0 - a_bar_next) * eps_t
#             xt_list.append(x)

#         return eps_list, xt_list

#     @torch.no_grad()
#     def ddim_inversion_eps(self, x0: torch.Tensor):
#         """
#         Retourne:
#           - eps_list: liste de T tenseurs (B,C,H,W), un par step DDIM
#           - xt_list:  liste optionnelle de T tenseurs (B,C,H,W) (ou None)
#         """
#         if hasattr(self.diffusion, "ddim_reverse_sample_loop"):
#             return self._ddim_reverse_loop_builtin(x0)

#         stepwise = self._ddim_reverse_loop_stepwise(x0)
#         if stepwise is not None:
#             return stepwise

#         return self._ddim_reverse_loop_eps_only(x0)
