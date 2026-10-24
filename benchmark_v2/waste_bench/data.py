"""Manifest -> protocol splits -> transforms -> DataLoaders."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .constants import MATERIALS, N_CLASSES, IMAGENET_MEAN_LIST, IMAGENET_STD_LIST
from .utils import logger


def build_transform(input_size, train, aug_mode="strong"):
    """Augment training images only; validation/test remain deterministic."""
    assert aug_mode in ("off", "mild", "strong")
    norm = transforms.Normalize(IMAGENET_MEAN_LIST, IMAGENET_STD_LIST)
    if not train:
        return transforms.Compose([transforms.Resize((input_size, input_size)),
                                   transforms.ToTensor(), norm])
    aug = [transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
           transforms.RandomHorizontalFlip()]
    if aug_mode in ("mild", "strong"):
        aug += [transforms.RandomVerticalFlip(0.2),
                transforms.RandomRotation(30),
                transforms.RandomAffine(0, translate=(0.15, 0.15), shear=10)]
    if aug_mode == "strong":
        aug += [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(0.1),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))]
    return transforms.Compose([transforms.Resize((int(input_size * 1.15),) * 2),
                               *aug, transforms.ToTensor(), norm])


class ManifestDataset(Dataset):
    """Rows of manifest.csv -> (image tensor, material index, recyclable flag)."""

    def __init__(self, rows, input_size=224, train=True, aug_mode="strong"):
        self.rows = rows
        self.tf = build_transform(input_size, train, aug_mode)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["material"]), int(r["recyclable"])


def load_manifest(path):
    df = pd.read_csv(path)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    return df


def select_rows(df, protocol):
    """Returns (train_rows, val_rows, test_rows).

    in_domain: original TrashNet-only protocol (clean split).
    taco_only: TACO-only protocol, grouped by source photo (wild split).
    cross_domain: train on clean, validate/test on held-out wild.
    joint: train on clean+wild train; validate/test on wild val/test.
    """
    clean = df[df.domain == "clean"]
    wild = df[df.domain == "wild"]
    rec = lambda d: d.to_dict("records")
    if protocol == "in_domain":
        return (rec(clean[clean.split == "train"]), rec(clean[clean.split == "val"]),
                rec(clean[clean.split == "test"]))
    if protocol == "taco_only":
         return (rec(wild[wild.split == "train"]), rec(wild[wild.split == "val"]),
                        rec(wild[wild.split == "test"]))
        #if wild.empty:
         #   raise ValueError("taco_only requires a manifest containing domain='wild' rows")
        # If the v3 policy-aware columns exist, train/evaluate only rows explicitly
        # eligible for strict evaluation and use the v3 target, not the legacy label.
        #if "strict_eval_include" in wild.columns:
         #   wild = wild[wild["strict_eval_include"].fillna(0).astype(int) == 1].copy()
        #if "recyclable_v3" in wild.columns:
        #    wild = wild[wild["recyclable_v3"].notna()].copy()
         #   wild["recyclable"] = wild["recyclable_v3"].astype(int)
        #if wild.empty:
        #    raise ValueError("No eligible TACO rows remain. Build benchmark_v3/manifest_v3.csv first.")
        #splits = {s: wild[wild.split == s] for s in ("train", "val", "test")}
        
        #if any(part.empty for part in splits.values()):
         #   raise ValueError(
         #       "taco_only requires non-empty TACO train/val/test splits; "
         #       "check benchmark_v3/manifest_v3.csv and strict-evaluation filtering."
         #   )
        #logger.info("TACO-only strict rows: train=%d val=%d test=%d",
         #           len(splits["train"]), len(splits["val"]), len(splits["test"]))
        #if "photo" in wild.columns:
         #   overlap = (
          #      set(splits["train"]["photo"].dropna())
           #     & set(splits["test"]["photo"].dropna())
            #)
            #if overlap:
          #      raise ValueError(f"TACO photo leakage: {len(overlap)} source photos overlap train/test")
        #return rec(splits["train"]), rec(splits["val"]), rec(splits["test"])
    if protocol == "cross_domain":
        return rec(clean), rec(wild[wild.split == "val"]), rec(wild[wild.split == "test"])
    if protocol == "joint":
        tr = pd.concat([clean[clean.split == "train"], wild[wild.split == "train"]])
        return rec(tr), rec(wild[wild.split == "val"]), rec(wild[wild.split == "test"])
    raise ValueError(protocol)


def make_loader(rows, input_size, train, aug_mode, bs, num_workers=2):
    ds = ManifestDataset(rows, input_size, train, aug_mode)
    return DataLoader(ds, bs, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def class_weights_from(rows, device):
    """Inverse-frequency weights per material, computed from training rows."""
    raw_counts = np.bincount([r["material"] for r in rows], minlength=N_CLASSES).astype(float)
    counts = raw_counts.copy()
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    logger.info("  class weights (from %d training rows):", len(rows))
    for mat, c, wi in zip(MATERIALS, raw_counts, w):
        logger.info(f"    {mat:<10s} n={int(c):5d}  weight={wi:.4f}")
    return torch.tensor(w, dtype=torch.float32, device=device)
