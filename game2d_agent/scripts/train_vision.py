"""
Train VisionEncoder to predict game state labels from raw RGB frames.

Usage:
    python scripts/train_vision.py
    python scripts/train_vision.py --data data/vision/dataset.npz --epochs 30
    python scripts/train_vision.py --epochs 50 --lr 5e-4 --batch-size 512
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.vision_encoder import VisionNet, SCENE_DIM


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    data = np.load(args.data)
    frames = data["frames"]   # (N, 84, 84, 3) uint8
    labels = data["labels"]   # (N, 18) float32
    N = len(frames)
    print(f"Dataset     : {N:,} samples")

    # (N, 84, 84, 3) uint8 → (N, 3, 84, 84) float32 [0,1]
    X = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    Y = torch.from_numpy(labels)

    n_val   = max(2000, int(0.1 * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(TensorDataset(X, Y), [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=device.type == "cuda")

    print(f"Train / val : {n_train:,} / {n_val:,}")
    print(f"Label dim   : {SCENE_DIM}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = VisionNet(in_channels=3, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder_state_dict": model.encoder.state_dict(),
                "embed_dim":          args.embed_dim,
            }, args.out)
            saved = "  [saved]"

        print(f"epoch {epoch:>3}/{args.epochs}"
              f"  train {train_loss:.5f}"
              f"  val {val_loss:.5f}"
              f"  lr {lr_now:.2e}"
              f"{saved}")

    print(f"\nBest val loss : {best_val_loss:.5f}")
    print(f"Encoder saved → {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train VisionEncoder from game frames")
    parser.add_argument("--data",       default="data/vision/dataset.npz")
    parser.add_argument("--out",        default="models/vision_encoder.pt")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--embed-dim",  type=int,   default=64,
                        help="CNN embedding size (default: 64)")
    parser.add_argument("--workers",    type=int,   default=4,
                        help="DataLoader worker processes for data loading (default: 4)")
    args = parser.parse_args()
    train(args)
