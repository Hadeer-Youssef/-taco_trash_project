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
    """aug_mode controls on-the-fly variation for TRAIN samples only
    (val/test are always plain resize + normalize):

      "off"    : RandomResizedCrop + RandomHorizontalFlip only
      "mild"   : "off" + small rotation / vertical flip / shift+shear — geometric
                 variety that doesn't touch colour/texture
      "strong" : "mild" + colour jitter, grayscale, blur — the domain-randomizing
                 set meant to fight the clean/wild gap (cross_domain, joint); it
                 makes the task harder with no payoff when train and test share a
                 domain (in_domain)."""
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
    # manifest.csv was written on Windows; normalise so it also works on Linux/Kaggle
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    return df


def select_rows(df, protocol):
    """Returns (train_rows, val_rows, test_rows) as lists of dicts.

      P1 in_domain    : train clean/train -> val clean/val -> test clean/test
      P2 cross_domain : train clean/all   -> (no val)       -> test wild/all
      P3 joint        : train clean+wild/train -> val wild/val -> test wild/test

    val_rows is None when the protocol has no held-out labelled split."""
    clean = df[df.domain == "clean"]
    wild = df[df.domain == "wild"]
    rec = lambda d: d.to_dict("records")
    if protocol == "in_domain":
        return (rec(clean[clean.split == "train"]), rec(clean[clean.split == "val"]),
                rec(clean[clean.split == "test"]))
    if protocol == "cross_domain":
        return rec(clean), None, rec(wild)
    if protocol == "joint":
        tr = pd.concat([clean[clean.split == "train"], wild[wild.split == "train"]])
        return rec(tr), rec(wild[wild.split == "val"]), rec(wild[wild.split == "test"])
    raise ValueError(protocol)


def make_loader(rows, input_size, train, aug_mode, bs, num_workers=2):
    ds = ManifestDataset(rows, input_size, train, aug_mode)
    return DataLoader(ds, bs, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def class_weights_from(rows, device):
    """Inverse-frequency weights per material, computed from the training rows."""
    raw_counts = np.bincount([r["material"] for r in rows], minlength=N_CLASSES).astype(float)
    counts = raw_counts.copy()
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    logger.info("  class weights (from %d training rows):", len(rows))
    for mat, c, wi in zip(MATERIALS, raw_counts, w):
        logger.info(f"    {mat:<10s} n={int(c):5d}  weight={wi:.4f}")
    return torch.tensor(w, dtype=torch.float32, device=device)
