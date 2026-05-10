# repos/edm-main/bridge_example.py
# Bridge pour EDM (NVIDIA). Compatible avec generate.py/training/network.py.
# Expose:
#   make_denoiser(ckpt, device=None, class_labels=None) -> obj avec:
#       - denoise(x, sigma) -> x0_hat
#       - edm_drift(x, sigma) -> (x - x0_hat)/sigma
#       - sigma_min / sigma_max / round_sigma (si présents dans le net)

import os, sys, pickle, torch

# 1) Assurer que le dossier EDM est dans le sys.path AVANT le pickle.load
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))  # .../repos/edm-main
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 2) Importer dnnlib et les modules EDM (facultatif, mais aide l'unpickling)
try:
    import dnnlib  # fourni dans le repo EDM
except Exception:
    dnnlib = None

# Essayons aussi d'importer torch_utils.persistence (souvent requis par les pickles)
try:
    from torch_utils import persistence as _persistence  # noqa: F401
except Exception as e:
    # L'import échouera si REPO_ROOT est incorrect; on laisse continuer, le pickle peut encore marcher
    pass

class _EDMDenoiserWrapper(torch.nn.Module):
    def __init__(self, net, device=None, class_labels=None):
        super().__init__()
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.net.to(self.device)
        except Exception:
            pass

        # Métadonnées utiles
        self.sigma_min = float(getattr(self.net, "sigma_min", 0.0))
        self.sigma_max = float(getattr(self.net, "sigma_max", float("inf")))
        self.round_sigma = getattr(self.net, "round_sigma", lambda s: torch.as_tensor(s))

        # Labels (unconditional par défaut)
        self.label_dim = int(getattr(self.net, "label_dim", 0))
        if class_labels is not None:
            self.class_labels = class_labels.to(self.device)
        else:
            self.class_labels = None  # construit on-the-fly si besoin

    @torch.no_grad()
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True).to(torch.float32)
        sigma = sigma.to(self.device, non_blocking=True).to(torch.float32)

        # Construire des labels neutres si nécessaire
        if self.label_dim > 0:
            if self.class_labels is None:
                cls = torch.zeros(x.shape[0], self.label_dim, device=self.device, dtype=torch.float32)
            else:
                cls = self.class_labels
                if cls.shape[0] != x.shape[0]:
                    if cls.shape[0] == 1:
                        cls = cls.expand(x.shape[0], -1).contiguous()
                    else:
                        raise RuntimeError("Batch mismatch for class_labels")
        else:
            cls = None

        out = self.net(x, sigma, cls)
        if isinstance(out, dict):
            if "x0" in out:
                return out["x0"].to(torch.float32)
            if "denoised" in out:
                return out["denoised"].to(torch.float32)
            # sinon on prend la première valeur
            return next(iter(out.values())).to(torch.float32)
        return out.to(torch.float32)

    @torch.no_grad()
    def edm_drift(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True).to(torch.float32)
        sigma = sigma.to(self.device, non_blocking=True).to(torch.float32)
        x0_hat = self.denoise(x, sigma)
        return (x - x0_hat) / (sigma + 1e-12)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Rend le wrapper callable: out = wrapper(x, sigma)."""
        return self.denoise(x, sigma)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # on laisse nn.Module.__call__ gérer hooks, mais on garde la signature explicite
        return super().__call__(x, sigma, *args, **kwargs)

def _load_net_from_pkl(path: str, device: torch.device):
    # Identique à generate.py: pickle.load(...)[ 'ema' ]
    # IMPORTANT: REPO_ROOT est déjà dans sys.path, donc les imports dans le pickle peuvent résoudre.
    if dnnlib is not None:
        with dnnlib.util.open_url(path, verbose=True) as f:
            obj = pickle.load(f)
    else:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    if isinstance(obj, dict) and "ema" in obj:
        net = obj["ema"]
    else:
        net = obj

    try:
        net.to(device)
    except Exception:
        pass
    return net

def make_denoiser(checkpoint_path: str, device: torch.device | None = None, class_labels: torch.Tensor | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (not os.path.exists(checkpoint_path)) and (dnnlib is None):
        raise FileNotFoundError(f"Checkpoint not found and dnnlib unavailable: {checkpoint_path}")
    net = _load_net_from_pkl(checkpoint_path, device)
    return _EDMDenoiserWrapper(net, device=device, class_labels=class_labels)




# # repos/edm-main/bridge_example.py
# # Bridge pour EDM (NVIDIA). Fonctionne avec generate.py / training/network.py
# # API attendue par proxenergy:
# #   make_denoiser(checkpoint_path, device=None, class_labels=None) ->
# #       objet avec méthodes:
# #         - denoise(x, sigma) -> x0_hat (EDM precond output)
# #         - edm_drift(x, sigma) -> (x - x0_hat)/sigma
# #       attributs utiles (facultatif): sigma_min, sigma_max, round_sigma

# import os
# import pickle
# import torch

# # dnnlib est utilisé par generate.py pour charger des pkl depuis URL/chemin
# try:
#     import dnnlib
# except Exception:
#     dnnlib = None

# class _EDMDenoiserWrapper(torch.nn.Module):
#     def __init__(self, net, device=None, class_labels=None):
#         super().__init__()
#         self.net = net.eval()
#         for p in self.net.parameters():
#             p.requires_grad_(False)
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.net.to(self.device)

#         # Info EDM utile
#         self.sigma_min = getattr(self.net, "sigma_min", 0.0)
#         self.sigma_max = getattr(self.net, "sigma_max", float("inf"))
#         self.round_sigma = getattr(self.net, "round_sigma", lambda s: s)

#         # Gestion labels (unconditional/conditional)
#         self.label_dim = int(getattr(self.net, "label_dim", 0))
#         if class_labels is not None:
#             # utilisateur fournit explicitement des labels (tensor one-hot ou soft)
#             self.class_labels = class_labels.to(self.device)
#         else:
#             if self.label_dim > 0:
#                 # par défaut: vecteur zéro (classe "nulle")
#                 self.class_labels = torch.zeros(1, self.label_dim, device=self.device, dtype=torch.float32)
#             else:
#                 self.class_labels = None

#     @torch.no_grad()
#     def denoise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
#         """
#         Retourne x0_hat = net(x, sigma, class_labels), comme dans edm_sampler.
#         - x: [B,C,H,W] float32
#         - sigma: scalaire ou [B,1,1,1] float32
#         """
#         x = x.to(self.device, non_blocking=True).to(torch.float32)
#         sigma = sigma.to(self.device, non_blocking=True).to(torch.float32)
#         if self.class_labels is None and self.label_dim > 0:
#             # construit des labels neutres à la bonne taille B
#             B = x.shape[0]
#             cls = torch.zeros(B, self.label_dim, device=self.device, dtype=torch.float32)
#         else:
#             cls = self.class_labels
#             if cls is not None and cls.shape[0] != x.shape[0]:
#                 # broadcast si besoin
#                 if cls.shape[0] == 1:
#                     cls = cls.expand(x.shape[0], -1).contiguous()
#                 else:
#                     raise RuntimeError("class_labels batch mismatch with x")
#         # forward EDM: renvoie "denoised" (x0-like) selon la précondition EDM
#         out = self.net(x, sigma, cls)
#         # Certains nets peuvent renvoyer un dict; on gère x0 via clé "x0" si besoin
#         if isinstance(out, dict):
#             if "x0" in out:
#                 return out["x0"].to(torch.float32)
#             elif "denoised" in out:
#                 return out["denoised"].to(torch.float32)
#             else:
#                 # suppose sortie directe
#                 # (EDM officiel renvoie directement le tenseur débruité)
#                 return next(iter(out.values())).to(torch.float32)
#         return out.to(torch.float32)

#     @torch.no_grad()
#     def edm_drift(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
#         """
#         dx/dsigma = (x - D(x, sigma)) / sigma
#         """
#         x = x.to(self.device, non_blocking=True).to(torch.float32)
#         sigma = sigma.to(self.device, non_blocking=True).to(torch.float32)
#         x0_hat = self.denoise(x, sigma)
#         return (x - x0_hat) / (sigma + 1e-12)

# def _load_net_from_pkl(path: str, device: torch.device):
#     # même logique que generate.py : pickle.load(...)[ 'ema' ]
#     if dnnlib is not None:
#         with dnnlib.util.open_url(path, verbose=True) as f:
#             obj = pickle.load(f)
#     else:
#         with open(path, "rb") as f:
#             obj = pickle.load(f)
#     if isinstance(obj, dict) and "ema" in obj:
#         net = obj["ema"]
#     else:
#         # fallback: certains dumps stockent directement le réseau
#         net = obj
#     # déplace sur device (le wrapper s'en charge aussi, mais on normalise ici)
#     try:
#         net.to(device)
#     except Exception:
#         pass
#     return net

# def make_denoiser(checkpoint_path: str, device: torch.device | None = None, class_labels: torch.Tensor | None = None):
#     """
#     Fabrique un wrapper "denoiser" compatible proxenergy:
#       - .denoise(x, sigma) -> x0_hat
#       - .edm_drift(x, sigma) -> (x - x0_hat)/sigma
#       - .sigma_min, .sigma_max, .round_sigma (si dispo)
#     """
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if not os.path.exists(checkpoint_path) and (dnnlib is None):
#         raise FileNotFoundError(f"Checkpoint not found and dnnlib unavailable: {checkpoint_path}")
#     net = _load_net_from_pkl(checkpoint_path, device)
#     return _EDMDenoiserWrapper(net, device=device, class_labels=class_labels)
