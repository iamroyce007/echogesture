"""Train SignNet on cached landmarks (data/splits.json).

  python scripts/train.py --epochs 150
Outputs checkpoints/model.pt (weights + vocab + config) and checkpoints/calib.npy (quantization calibration set).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.features import augment, build_features  # noqa: E402
from echogesture.model import SignNet  # noqa: E402


class LandmarkDataset(Dataset):
    def __init__(self, data_dir: Path, items: list[dict], train: bool, seed: int = 0):
        self.seqs = [np.load(data_dir / it["npy"]).astype(np.float32) for it in items]
        self.labels = [it["label"] for it in items]
        self.train = train
        self.rng = np.random.default_rng(seed)
        if not train:  # deterministic features, cache once
            self.cache = [build_features(s) for s in self.seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        x = build_features(augment(self.seqs[i], self.rng)) if self.train else self.cache[i]
        return torch.from_numpy(x), self.labels[i]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = top1 = top5 = 0
    for x, y in loader:
        logits = model(x.to(device)).cpu()
        k = min(5, logits.shape[1])
        pred = logits.topk(k, dim=1).indices
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y[:, None]).any(1).sum().item()
        n += len(y)
    return top1 / max(n, 1), top5 / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-on-val", action="store_true",
                    help="final model: also train on the val split for a fixed number of epochs, keep last epoch")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    splits = json.loads((args.data_dir / "splits.json").read_text())
    vocab = splits["vocab"]
    by = {s: [i for i in splits["items"] if i["split"] == s] for s in ("train", "val", "test")}
    if args.train_on_val:
        by["train"] = by["train"] + by["val"]
    if not by["val"]:  # fall back: hold out 10% of train
        k = max(1, len(by["train"]) // 10)
        by["val"], by["train"] = by["train"][:k], by["train"][k:]
    print({s: len(v) for s, v in by.items()}, "classes:", len(vocab), "device:", args.device)

    train_ds = LandmarkDataset(args.data_dir, by["train"], train=True, seed=args.seed)
    val_ds = LandmarkDataset(args.data_dir, by["val"], train=False)
    test_ds = LandmarkDataset(args.data_dir, by["test"], train=False) if by["test"] else None
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, batch_size=256)

    model = SignNet(len(vocab), dim=args.dim, depth=args.depth).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_dl)
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best, t0 = -1.0, time.time()
    config = dict(dim=args.dim, depth=args.depth, num_classes=len(vocab))
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        v1, v5 = evaluate(model, val_dl, args.device)
        if v1 > best or (args.train_on_val and ep == args.epochs):
            best = max(best, v1)
            torch.save(dict(state_dict=model.state_dict(), vocab=vocab, config=config), args.out_dir / "model.pt")
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"ep {ep:3d} loss {tot / len(train_ds):.3f} val top1 {v1:.3f} top5 {v5:.3f} best {best:.3f} "
                  f"({time.time() - t0:.0f}s)")

    ckpt = torch.load(args.out_dir / "model.pt", map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    summary = dict(val_top1=best)
    if test_ds:
        t1, t5 = evaluate(model, DataLoader(test_ds, batch_size=256), args.device)
        summary.update(test_top1=t1, test_top5=t5)
    print("summary:", summary)
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))

    # calibration set for static INT8 quantization: un-augmented training features
    idx = np.random.default_rng(0).permutation(len(train_ds))[:300]
    calib = np.stack([build_features(train_ds.seqs[i]) for i in idx])
    np.save(args.out_dir / "calib.npy", calib)


if __name__ == "__main__":
    main()
