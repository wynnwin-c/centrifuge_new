from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pathlib
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from _2_models.fewshot_binary import FewShotBinaryNet, supervised_contrastive_loss  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_logger(log_dir: str, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class WindowDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, s_path: str, g_path: str, binary: bool = False):
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        self.s = np.load(s_path).astype(np.float32)
        self.g = np.load(g_path).astype(np.int64)
        mu = self.X.mean(axis=1, keepdims=True)
        std = self.X.std(axis=1, keepdims=True) + 1e-6
        self.X = (self.X - mu) / std
        if binary:
            self.y = (self.y > 0).astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.expand_dims(self.X[idx], 0)),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.s[idx], dtype=torch.float32),
            torch.tensor(self.g[idx], dtype=torch.long),
        )


def subset_array(dataset, attr: str):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        idx = np.asarray(dataset.indices)
        return getattr(base, attr)[idx]
    return getattr(dataset, attr)


def make_binary_datasets(src_name: str, tgt_name: str, processed_dir: pathlib.Path):
    def paths(name: str):
        return (
            processed_dir / f"{name}_X.npy",
            processed_dir / f"{name}_y.npy",
            processed_dir / f"{name}_s.npy",
            processed_dir / f"{name}_g.npy",
        )

    sx, sy, ss, sg = paths(src_name)
    tx, ty, ts, tg = paths(tgt_name)
    return (
        WindowDataset(str(sx), str(sy), str(ss), str(sg), binary=True),
        WindowDataset(str(tx), str(ty), str(ts), str(tg), binary=True),
    )


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, drop_last: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def choose_support_indices(dataset, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    y = subset_array(dataset, "y")
    local = np.arange(len(y))
    picked = []
    stats = {}
    for cls in [0, 1]:
        cls_idx = local[y == cls]
        if len(cls_idx) == 0:
            continue
        take = min(per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        picked.extend(chosen.tolist())
        stats[cls] = int(take)
    picked = sorted(set(picked))
    return Subset(dataset, picked), stats


def build_group_splits(dataset, mode: str, test_ratio: float, seed: int, repeats: int):
    y = dataset.y
    g = dataset.g
    class_groups = {cls: np.unique(g[y == cls]).tolist() for cls in [0, 1]}

    if mode == "logo":
        combos = list(itertools.product(class_groups[0], class_groups[1]))
        splits = []
        for gid0, gid1 in combos:
            test_groups = {0: [gid0], 1: [gid1]}
            train_idx, test_idx = group_split_indices(y, g, test_groups)
            splits.append({"name": f"logo_g0_{gid0}_g1_{gid1}", "train_idx": train_idx, "test_idx": test_idx, "test_groups": test_groups})
        return splits

    rng = np.random.default_rng(seed)
    splits = []
    for rep in range(repeats):
        test_groups = {}
        for cls in [0, 1]:
            groups = np.array(class_groups[cls], dtype=np.int64)
            groups = groups.copy()
            rng.shuffle(groups)
            n_test = max(1, int(round(len(groups) * test_ratio)))
            n_test = min(n_test, len(groups) - 1) if len(groups) > 1 else 1
            test_groups[cls] = groups[:n_test].tolist()
        train_idx, test_idx = group_split_indices(y, g, test_groups)
        splits.append({"name": f"repeat_{rep:02d}", "train_idx": train_idx, "test_idx": test_idx, "test_groups": test_groups})
    return splits


def group_split_indices(y: np.ndarray, g: np.ndarray, test_groups: dict[int, list[int]]):
    train_indices = []
    test_indices = []
    for cls in [0, 1]:
        is_test = np.isin(g, np.asarray(test_groups[cls])) & (y == cls)
        is_train = (~np.isin(g, np.asarray(test_groups[cls]))) & (y == cls)
        train_indices.extend(np.where(is_train)[0].tolist())
        test_indices.extend(np.where(is_test)[0].tolist())
    return sorted(train_indices), sorted(test_indices)


@dataclass
class EvalResult:
    window_acc: float
    window_f1: float
    group_acc: float
    group_f1: float
    fault_recall: float
    window_cm: list[list[int]]
    group_cm: list[list[int]]
    support_stats: dict
    split_name: str


def train_epoch(model, loader, optimizer, device, ce_weight: float, con_weight: float, temperature: float):
    model.train()
    loss_meter = 0.0
    cls_meter = 0.0
    con_meter = 0.0
    for xb, yb, _, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        cls_loss = F.cross_entropy(out["logits"], yb)
        con_loss = supervised_contrastive_loss(out["emb"], yb, temperature=temperature)
        loss = ce_weight * cls_loss + con_weight * con_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += float(loss.item())
        cls_meter += float(cls_loss.item())
        con_meter += float(con_loss.item())
    steps = max(1, len(loader))
    return {"loss": loss_meter / steps, "cls": cls_meter / steps, "con": con_meter / steps}


def freeze_backbone(model: FewShotBinaryNet, freeze: bool):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


@torch.no_grad()
def compute_prototypes(model, support_set, device):
    loader = DataLoader(support_set, batch_size=256, shuffle=False, num_workers=0, drop_last=False)
    emb_all = []
    y_all = []
    for xb, yb, _, _ in loader:
        xb = xb.to(device)
        emb = model(xb)["emb"].cpu()
        emb_all.append(emb)
        y_all.append(yb)
    emb = torch.cat(emb_all, dim=0)
    y = torch.cat(y_all, dim=0)
    protos = []
    for cls in [0, 1]:
        cls_emb = emb[y == cls]
        protos.append(F.normalize(cls_emb.mean(dim=0), dim=0))
    return torch.stack(protos, dim=0).to(device)


@torch.no_grad()
def evaluate_with_prototypes(model, dataset, support_set, device, alpha: float):
    prototypes = compute_prototypes(model, support_set, device)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False)
    prob_all = []
    y_all = []
    g_all = []
    for xb, yb, _, gb in loader:
        xb = xb.to(device)
        out = model(xb)
        cls_prob = F.softmax(out["logits"], dim=1)
        proto_logits = torch.matmul(out["emb"], prototypes.T)
        proto_prob = F.softmax(proto_logits, dim=1)
        fused = alpha * cls_prob + (1.0 - alpha) * proto_prob
        prob_all.append(fused.cpu().numpy())
        y_all.append(yb.numpy())
        g_all.append(gb.numpy())
    probs = np.concatenate(prob_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    groups = np.concatenate(g_all, axis=0)
    y_pred = probs.argmax(axis=1)
    window_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    window_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    fault_recall = float(report["1"]["recall"]) if "1" in report else 0.0

    group_probs = []
    group_y = []
    for gid in np.unique(groups):
        idx = groups == gid
        group_probs.append(probs[idx].mean(axis=0))
        group_y.append(int(np.bincount(y_true[idx]).argmax()))
    group_probs = np.asarray(group_probs)
    group_y = np.asarray(group_y)
    group_pred = group_probs.argmax(axis=1)
    group_cm = confusion_matrix(group_y, group_pred, labels=[0, 1])
    group_f1 = float(f1_score(group_y, group_pred, average="macro", zero_division=0))
    return {
        "window_acc": float(accuracy_score(y_true, y_pred)),
        "window_f1": window_f1,
        "group_acc": float(accuracy_score(group_y, group_pred)),
        "group_f1": group_f1,
        "fault_recall": fault_recall,
        "window_cm": window_cm.tolist(),
        "group_cm": group_cm.tolist(),
    }


def run_split(model_cfg, train_cfg, ds_src, ds_tgt, split, device, logger, split_seed):
    src_loader = build_loader(ds_src, train_cfg["batch_size"], True, train_cfg["num_workers"], False)
    tgt_train = Subset(ds_tgt, split["train_idx"])
    tgt_test = Subset(ds_tgt, split["test_idx"])
    support_set, support_stats = choose_support_indices(tgt_train, train_cfg["semi_target_per_class"], split_seed)
    support_loader = build_loader(support_set, min(train_cfg["batch_size"], max(2, len(support_set))), True, 0, False)

    model = FewShotBinaryNet(
        in_channels=model_cfg["in_channels"],
        feat_dim=model_cfg["feat_dim"],
        emb_dim=model_cfg["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["pretrain_lr"], weight_decay=train_cfg["weight_decay"])
    best_state = None
    best_src_acc = -1.0
    for epoch in range(1, train_cfg["pretrain_epochs"] + 1):
        stats = train_epoch(model, src_loader, optimizer, device, 1.0, train_cfg["supcon_weight"], train_cfg["temperature"])
        if epoch == train_cfg["pretrain_epochs"] or epoch % max(1, train_cfg["pretrain_epochs"] // 5) == 0:
            src_eval = evaluate_with_prototypes(model, ds_src, ds_src, device, alpha=1.0)
            logger.info(f"{split['name']} pretrain epoch={epoch} loss={stats['loss']:.4f} src_acc={src_eval['window_acc']:.4f}")
            if src_eval["window_acc"] > best_src_acc:
                best_src_acc = src_eval["window_acc"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    freeze_backbone(model, True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg["finetune_head_lr"], weight_decay=train_cfg["weight_decay"])
    for epoch in range(1, train_cfg["head_epochs"] + 1):
        stats = train_epoch(model, support_loader, optimizer, device, 1.0, train_cfg["support_supcon_weight"], train_cfg["temperature"])
        logger.info(f"{split['name']} head_ft epoch={epoch} loss={stats['loss']:.4f}")

    freeze_backbone(model, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["finetune_lr"], weight_decay=train_cfg["weight_decay"])
    for epoch in range(1, train_cfg["finetune_epochs"] + 1):
        stats = train_epoch(model, support_loader, optimizer, device, 1.0, train_cfg["support_supcon_weight"], train_cfg["temperature"])
        logger.info(f"{split['name']} full_ft epoch={epoch} loss={stats['loss']:.4f}")

    eval_metrics = evaluate_with_prototypes(model, tgt_test, support_set, device, alpha=train_cfg["prototype_alpha"])
    logger.info(
        f"{split['name']} window_acc={eval_metrics['window_acc']:.4f} group_acc={eval_metrics['group_acc']:.4f} "
        f"window_f1={eval_metrics['window_f1']:.4f} group_f1={eval_metrics['group_f1']:.4f} fault_recall={eval_metrics['fault_recall']:.4f}"
    )
    return EvalResult(
        window_acc=eval_metrics["window_acc"],
        window_f1=eval_metrics["window_f1"],
        group_acc=eval_metrics["group_acc"],
        group_f1=eval_metrics["group_f1"],
        fault_recall=eval_metrics["fault_recall"],
        window_cm=eval_metrics["window_cm"],
        group_cm=eval_metrics["group_cm"],
        support_stats=support_stats,
        split_name=split["name"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "3_train" / "config_train.yaml"))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--per_class", type=int, default=None)
    parser.add_argument("--split_mode", type=str, choices=["repeated", "logo"], default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--head_epochs", type=int, default=None)
    parser.add_argument("--finetune_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_cfg = next((t for t in cfg["tasks"] if t["name"] == args.task), None)
    if task_cfg is None:
        raise ValueError(f"task not found: {args.task}")

    few_cfg = cfg["fewshot_binary"]
    if args.per_class is not None:
        few_cfg["semi_target_per_class"] = args.per_class
    if args.split_mode is not None:
        few_cfg["split_mode"] = args.split_mode
    if args.repeats is not None:
        few_cfg["repeated_splits"] = args.repeats
    if args.pretrain_epochs is not None:
        few_cfg["pretrain_epochs"] = args.pretrain_epochs
    if args.head_epochs is not None:
        few_cfg["head_epochs"] = args.head_epochs
    if args.finetune_epochs is not None:
        few_cfg["finetune_epochs"] = args.finetune_epochs

    out_root = ROOT / "4_results" / "fewshot_binary" / args.task
    ensure_dirs([out_root])
    logger = build_logger(str(out_root), f"fewshot_binary_{args.task}")
    set_seed(int(few_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = ROOT / "1_data" / "processed"

    ds_src, ds_tgt = make_binary_datasets(task_cfg["source"], task_cfg["target"], processed_dir)
    splits = build_group_splits(
        ds_tgt,
        mode=few_cfg["split_mode"],
        test_ratio=float(few_cfg["target_group_test_ratio"]),
        seed=int(few_cfg["seed"]),
        repeats=int(few_cfg["repeated_splits"]),
    )

    logger.info(
        f"task={args.task} split_mode={few_cfg['split_mode']} splits={len(splits)} per_class={few_cfg['semi_target_per_class']}"
    )
    logger.info(
        f"source_binary_labels={dict(zip(*np.unique(ds_src.y, return_counts=True)))} "
        f"target_binary_labels={dict(zip(*np.unique(ds_tgt.y, return_counts=True)))}"
    )

    all_results = []
    model_cfg = {
        "in_channels": int(cfg["data"]["in_channels"]),
        "feat_dim": int(few_cfg["feat_dim"]),
        "embedding_dim": int(few_cfg["embedding_dim"]),
    }

    for idx, split in enumerate(splits):
        split_seed = int(few_cfg["seed"]) + idx
        result = run_split(model_cfg, few_cfg, ds_src, ds_tgt, split, device, logger, split_seed)
        payload = result.__dict__.copy()
        payload["test_groups"] = split["test_groups"]
        all_results.append(payload)

    summary = {
        "window_acc_mean": float(np.mean([r["window_acc"] for r in all_results])),
        "window_acc_std": float(np.std([r["window_acc"] for r in all_results])),
        "group_acc_mean": float(np.mean([r["group_acc"] for r in all_results])),
        "group_acc_std": float(np.std([r["group_acc"] for r in all_results])),
        "window_f1_mean": float(np.mean([r["window_f1"] for r in all_results])),
        "group_f1_mean": float(np.mean([r["group_f1"] for r in all_results])),
        "fault_recall_mean": float(np.mean([r["fault_recall"] for r in all_results])),
        "splits": all_results,
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(
        f"summary group_acc_mean={summary['group_acc_mean']:.4f} window_acc_mean={summary['window_acc_mean']:.4f} "
        f"group_acc_std={summary['group_acc_std']:.4f}"
    )


if __name__ == "__main__":
    main()
