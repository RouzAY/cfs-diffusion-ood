# scripts/benchmark_images.py
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from cfs.adapters import ImprovedDiffusionAdapter, EDMAdapter, UViTAdapter
from cfs.datasets.images import load_data
from cfs.methods import CFSOOD, MSMAOOD, DiffPathOOD, DDPMOOD, GEPC, NaiveRecon
from cfs.utils.metrics import auroc_ood_high, aupr_ood_high, fpr95_ood_high


METHODS = {
    "cfs": CFSOOD,
    "msma": MSMAOOD,
    "diffpath": DiffPathOOD,
    "ddpm_ood": DDPMOOD,
    "gepc": GEPC,
    "naive_recon": NaiveRecon,
}


def set_global_determinism(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rebuild_loader_with_generator(loader, shuffle: bool, seed: int, num_workers: Optional[int] = None):
    gen = torch.Generator()
    gen.manual_seed(seed)

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers if num_workers is None else num_workers,
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def clamp_loader(loader, limit):
    if not limit or limit <= 0:
        return loader

    n = min(int(limit), len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, np.arange(n)),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=getattr(loader, "pin_memory", True),
        drop_last=False,
    )


def normalize_ood_cfg(ood_cfg):
    if ood_cfg is None:
        return []
    if isinstance(ood_cfg, str):
        return [{"name": ood_cfg, "split": "test", "limit": None, "download": True}]
    if isinstance(ood_cfg, dict):
        return [ood_cfg]

    out = []
    for item in ood_cfg:
        if isinstance(item, str):
            out.append({"name": item, "split": "test", "limit": None, "download": True})
        else:
            out.append(dict(item))
    return out


def active_adapter_type(cfg: dict, cli_adapter: Optional[str]) -> str:
    return str(cli_adapter or cfg.get("adapter", "improved")).lower()


def get_active_backbone_cfg(cfg: dict, adapter_type: str) -> dict:
    backbones = dict(cfg.get("backbones", {}) or {})
    sub = dict(backbones.get(adapter_type, {}) or {})

    if adapter_type == "improved":
        if "model_path" not in sub and cfg.get("model_path") is not None:
            sub["model_path"] = cfg["model_path"]
        if "improved_args" not in sub and cfg.get("improved_args") is not None:
            sub["improved_args"] = cfg.get("improved_args", {})

    elif adapter_type == "edm":
        legacy_edm = dict(cfg.get("edm", {}) or {})
        if "model_path" not in sub:
            sub["model_path"] = legacy_edm.get("checkpoint_path", cfg.get("model_path"))

        for k in ("repo_dir", "prediction_type", "in_channels", "data_range"):
            if k not in sub and legacy_edm.get(k) is not None:
                sub[k] = legacy_edm[k]

        if "sampler" not in sub and cfg.get("sampler") is not None:
            sub["sampler"] = cfg["sampler"]

    return sub


def backbone_summary(adapter_type: str, bb_cfg: dict) -> dict:
    out = {"adapter": adapter_type, "model_path": bb_cfg.get("model_path")}

    if adapter_type == "improved":
        out["improved_args"] = bb_cfg.get("improved_args", {})
    elif adapter_type == "edm":
        out.update({
            "prediction_type": bb_cfg.get("prediction_type", "x0"),
            "in_channels": bb_cfg.get("in_channels", 3),
            "data_range": bb_cfg.get("data_range", [-1.0, 1.0]),
            "repo_dir": bb_cfg.get("repo_dir"),
            "sampler": bb_cfg.get("sampler", {}),
        })
    elif adapter_type == "uvit":
        out.update({
            "repo_dir": bb_cfg.get("repo_dir"),
            "config_path": bb_cfg.get("config_path"),
            "nnet_path": bb_cfg.get("nnet_path"),
            "class_label": bb_cfg.get("class_label"),
        })

    return out


def build_adapter_from_cfg(cfg: dict, device_index: int, backbone_size: int, data_size: int):
    adapter_type = active_adapter_type(cfg, None)
    bb_cfg = get_active_backbone_cfg(cfg, adapter_type)

    if adapter_type == "improved":
        model_path = bb_cfg.get("model_path")
        if not model_path:
            raise ValueError("Missing backbones.improved.model_path.")

        class Args:
            pass

        args = Args()
        args.model_path = model_path
        args.device = device_index
        args.image_size = backbone_size
        args.data_image_size = data_size
        args.n_ddim_steps = int(cfg.get("n_ddim_steps", 10))
        args.improved_args = bb_cfg.get("improved_args", {}) or {}

        adapter = ImprovedDiffusionAdapter(args)
        return adapter, bb_cfg

    if adapter_type == "edm":
        model_path = bb_cfg.get("model_path")
        if not model_path:
            raise ValueError("Missing backbones.edm.model_path.")

        device_str = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

        adapter = EDMAdapter(
            checkpoint_path=model_path,
            prediction_type=bb_cfg.get("prediction_type", "x0"),
            in_channels=int(bb_cfg.get("in_channels", 3)),
            data_range=tuple(bb_cfg.get("data_range", [-1.0, 1.0])),
            device=device_str,
            edm_repo_dir=bb_cfg.get("repo_dir"),
        )
        adapter.sampler_cfg = dict(bb_cfg.get("sampler", {}) or {})
        return adapter, bb_cfg

    if adapter_type == "uvit":
        device_str = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

        adapter = UViTAdapter(
            repo_dir=bb_cfg["repo_dir"],
            config_path=bb_cfg["config_path"],
            nnet_path=bb_cfg["nnet_path"],
            device=device_str,
            class_label=bb_cfg.get("class_label"),
        )
        return adapter, bb_cfg

    raise ValueError(f"Unknown adapter type: {adapter_type}")


def compute_ood_metrics(id_scores, ood_scores) -> dict:
    return {
        "auroc": float(auroc_ood_high(id_scores, ood_scores)),
        "aupr": float(aupr_ood_high(id_scores, ood_scores)),
        "fpr95": float(fpr95_ood_high(id_scores, ood_scores, tpr_target=0.95)),
    }


def int_or_none(x):
    try:
        return None if x is None else int(x)
    except Exception:
        return None


def format_cost(F, J):
    if F is None and J is None:
        return "unknown"
    return f"{int(F or 0)}F + {int(J or 0)}J"


def infer_gepc_group_size(mcfg: dict) -> int:
    group_set = str(mcfg.get("group_set", "flip180")).lower()
    use_shifts = bool(mcfg.get("group_shifts", False))

    g = 0
    if group_set in {"flip", "flip180", "full90"}:
        g += 2
    if group_set in {"flip180", "full90"}:
        g += 1
    if group_set == "full90":
        g += 2
    if use_shifts:
        g += 2

    return max(1, g)


def infer_logical_cost(method_obj, method_arg: str, cfg: dict) -> dict:
    F = int_or_none(getattr(method_obj, "total_nfe_", None))
    J = int_or_none(getattr(method_obj, "total_nfj_", None))
    source = "method_attr" if F is not None or J is not None else "inferred"

    if F is None:
        mcfg = dict(cfg.get(method_arg, {}) or {})

        if method_arg in {"cfs", "msma", "diffpath"}:
            F = int(max(1, mcfg.get("Kc", 1))) * int(max(1, mcfg.get("mc_test", 1)))
            J = 0

        elif method_arg == "naive_recon":
            F = int(max(1, mcfg.get("mc_test", 1)))
            J = 0

        elif method_arg == "ddpm_ood":
            if hasattr(method_obj, "_suffixes") and getattr(method_obj, "_suffixes", None):
                F = sum(len(suf) for suf in method_obj._suffixes) * int(max(1, mcfg.get("mc_test", 1)))
            else:
                K = int(max(1, mcfg.get("Kc", 1)))
                skip = int(max(1, mcfg.get("start_skip", 16)))
                starts = list(range(0, K, skip))
                F = sum(max(0, K - s) for s in starts) * int(max(1, mcfg.get("mc_test", 1)))
            J = 0

        elif method_arg == "gepc":
            mc_test = int(max(1, mcfg.get("mc_test", 1)))
            K_eff = len(getattr(method_obj, "_levels", [])) or int(max(1, mcfg.get("Kc", 1)))
            G = infer_gepc_group_size(mcfg)
            F = K_eff * mc_test * (1 + G)
            J = 0

    return {
        "F_per_image": F,
        "J_per_image": J,
        "logical_cost": format_cost(F, J),
        "source": source,
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", required=True)
    pre_args, remaining = pre.parse_known_args()

    with open(pre_args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--in_dist", default=None)
    parser.add_argument("--out_dist", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", choices=sorted(METHODS.keys()), default="cfs")
    parser.add_argument("--adapter", choices=["improved", "edm", "uvit"], default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--strict_determinism", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(remaining)

    if args.device is None:
        args.device = int(cfg.get("device", 0))
    if args.seed is None:
        args.seed = int(cfg.get("seed", 1337))
    if args.adapter is not None:
        cfg["adapter"] = args.adapter.lower()
    if not args.strict_determinism:
        args.strict_determinism = bool(cfg.get("strict_determinism", False))

    adapter_type = active_adapter_type(cfg, None)

    if args.model_path is not None:
        cfg.setdefault("backbones", {})
        cfg["backbones"].setdefault(adapter_type, {})
        cfg["backbones"][adapter_type]["model_path"] = args.model_path

    set_global_determinism(args.seed, deterministic=args.strict_determinism)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    backbone_size = int(cfg.get("image_size", 32))
    data_size = int(cfg.get("data_image_size", backbone_size))
    batch_size = int(cfg.get("batch_size", 128))
    data_dir = args.data_dir or cfg.get("data_root", "./data")

    adapter, bb_cfg = build_adapter_from_cfg(cfg, args.device, backbone_size, data_size)

    ev = cfg.get("eval", {}) or {}
    idtr_cfg = dict(ev.get("id_train", {}) or {})
    idte_cfg = dict(ev.get("id_test", {}) or {})
    ood_list = normalize_ood_cfg(ev.get("ood", []))

    if args.in_dist is not None:
        idtr_cfg["name"] = args.in_dist
        idte_cfg["name"] = args.in_dist

    id_name = idtr_cfg.get("name")
    if not id_name:
        raise ValueError("Set eval.id_train.name or use --in_dist.")

    if args.out_dist is not None:
        ood_list = [o for o in ood_list if o.get("name") == args.out_dist]
        if not ood_list:
            raise ValueError(f"--out_dist={args.out_dist} not found in eval.ood.")

    id_train = load_data(
        name=id_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=data_size,
        split=idtr_cfg.get("split", "train"),
        limit=idtr_cfg.get("limit"),
        download=idtr_cfg.get("download", True),
        shuffle=True,
        model_image_size=backbone_size,
        val_split_ratio=idtr_cfg.get("val_split_ratio", 0.5),
        val_split_seed=idtr_cfg.get("val_split_seed", 0),
        benchmark_name=idtr_cfg.get("benchmark_name", None),
    )

    id_test = load_data(
        name=id_name,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=data_size,
        split=idte_cfg.get("split", "test"),
        limit=idte_cfg.get("limit"),
        download=idte_cfg.get("download", True),
        shuffle=False,
        model_image_size=backbone_size,
        val_split_ratio=idte_cfg.get("val_split_ratio", 0.5),
        val_split_seed=idte_cfg.get("val_split_seed", 0),
        benchmark_name=idte_cfg.get("benchmark_name", None),
    )

    id_train = rebuild_loader_with_generator(clamp_loader(id_train, idtr_cfg.get("limit")), True, args.seed)
    id_test = rebuild_loader_with_generator(clamp_loader(id_test, idte_cfg.get("limit")), False, args.seed)

    mcfg = dict(cfg.get(args.method, {}) or {})
    mcfg.setdefault("verbose", bool(args.verbose))
    method = METHODS[args.method](**mcfg)

    t0 = time.time()
    method.fit_id_train(adapter, id_train)
    fit_sec = time.time() - t0

    cost_info = infer_logical_cost(method, args.method, cfg)

    t0 = time.time()
    sid = method.score_loader(adapter, id_test, tag=f"ID_{id_name}")
    id_score_sec = time.time() - t0

    results = {
        "method": method.name,
        "method_arg": args.method,
        "id": id_name,
        "seed": int(args.seed),
        "strict_determinism": bool(args.strict_determinism),
        "adapter": adapter_type,
        "backbone": backbone_summary(adapter_type, bb_cfg),
        "cost": cost_info,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_index": int(args.device),
        },
        "timing": {
            "fit_sec": float(fit_sec),
            "id_score_sec": float(id_score_sec),
            "id_n": int(sid.size),
            "id_ms_per_img": float(1000.0 * id_score_sec / max(1, sid.size)),
        },
        "pairs": [],
    }

    out_root = os.path.join(
        os.path.dirname(args.config),
        "results",
        args.method,
        adapter_type,
        id_name,
    )
    os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(out_root, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    for ood_cfg in ood_list:
        ood_name = ood_cfg.get("name")

        ood_test = load_data(
            name=ood_name,
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=data_size,
            split=ood_cfg.get("split", "test"),
            limit=ood_cfg.get("limit"),
            download=ood_cfg.get("download", True),
            shuffle=False,
            model_image_size=backbone_size,
            benchmark_name=ood_cfg.get("benchmark_name", None),
        )

        ood_test = rebuild_loader_with_generator(clamp_loader(ood_test, ood_cfg.get("limit")), False, args.seed)

        t0 = time.time()
        sod = method.score_loader(adapter, ood_test, tag=f"OOD_{ood_name}")
        ood_score_sec = time.time() - t0

        metrics = compute_ood_metrics(sid, sod)

        pair = {
            "ood": ood_name,
            **metrics,
            "cost": cost_info,
            "timing": {
                "ood_score_sec": float(ood_score_sec),
                "ood_n": int(sod.size),
                "ood_ms_per_img": float(1000.0 * ood_score_sec / max(1, sod.size)),
            },
        }
        results["pairs"].append(pair)

        print(
            f"[{method.name}] ({adapter_type}) {id_name} vs {ood_name} | "
            f"AUROC={metrics['auroc']:.4f} | AUPR={metrics['aupr']:.4f} | "
            f"FPR95={metrics['fpr95']:.4f} | cost={cost_info['logical_cost']}"
        )

    with open(os.path.join(out_root, "main_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    flat = [
        {
            "method": results["method"],
            "method_arg": results["method_arg"],
            "adapter": results["adapter"],
            "id": id_name,
            "ood": p["ood"],
            "auroc": p["auroc"],
            "aupr": p["aupr"],
            "fpr95": p["fpr95"],
            "F_per_image": cost_info["F_per_image"],
            "J_per_image": cost_info["J_per_image"],
            "logical_cost": cost_info["logical_cost"],
            "seed": args.seed,
        }
        for p in results["pairs"]
    ]

    with open(os.path.join(out_root, "main_results_flat.json"), "w") as f:
        json.dump(flat, f, indent=2)


if __name__ == "__main__":
    main()