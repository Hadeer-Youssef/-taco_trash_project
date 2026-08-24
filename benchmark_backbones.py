"""
Cross-domain backbone benchmark  (Phase 2)
==========================================
Runs the five transfer-learning backbones from the TrashNet notebook
(ResNet50, DenseNet121, MobileNetV2, VGG13, Inception_v3) on the *harmonized*
manifest under the three protocols, and reports the material (6-class) and
recyclability (binary) metrics that the paper needs.

Why a clean re-implementation rather than loading the notebook's .pth files:
the notebook builds each head with ad-hoc, mutable cell state (several competing
DenseNet heads, in-place freezes). For a reproducible benchmark we define each
backbone once, with a standard 6-class head, and drive everything off manifest.csv.
The architecture *set* and the transfer-learning recipe are identical to the
notebook; only the bookkeeping is made deterministic.

Protocols (read from manifest's domain/split columns):
  P1 in_domain    : train clean(train) -> test clean(test)
  P2 cross_domain : train clean(all)   -> test wild(all)     <-- headline
  P3 joint        : train clean+wild(train) -> test wild(test)

Material accuracy is the headline number; recyclability is derived per-image from
the manifest's `recyclable` column. non_recyclable RECALL is reported explicitly
(a false "recyclable" contaminates the stream).

Usage (on a GPU box, e.g. Kaggle):
    python benchmark_backbones.py --manifest manifest.csv \
        --backbones resnet50 densenet121 mobilenet_v2 vgg13 inception_v3 \
        --protocol cross_domain --epochs 15 --out results_p2.json
"""

import os, json, argparse, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix)

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MATERIAL_RECYCLABLE = [1, 1, 1, 1, 1, 0]  # index aligned to MATERIALS (rule baseline)


# ----------------------------------------------------------------------------
# Dataset driven by the frozen manifest
# ----------------------------------------------------------------------------
class ManifestDataset(Dataset):
    def __init__(self, rows, input_size=224, train=True, domain_randomize=True):
        self.rows = rows
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            aug = [transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                   transforms.RandomHorizontalFlip()]
            if domain_randomize:
                aug += [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                        transforms.RandomGrayscale(0.1),
                        transforms.GaussianBlur(3, sigma=(0.1, 2.0))]
            self.tf = transforms.Compose([transforms.Resize((int(input_size * 1.15),) * 2),
                                          *aug, transforms.ToTensor(), norm])
        else:
            self.tf = transforms.Compose([transforms.Resize((input_size, input_size)),
                                          transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["material"]), int(r["recyclable"])


# ----------------------------------------------------------------------------
# Backbone factory — one standard 6-class head per architecture
# ----------------------------------------------------------------------------
def build_backbone(name, n_classes=6, freeze_features=True):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, 224
    if name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
        return m, 224
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1")
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        m.classifier = nn.Sequential(nn.Dropout(0.2),
                                     nn.Linear(1280, n_classes))
        return m, 224
    if name == "vgg13":
        m = models.vgg13(weights="IMAGENET1K_V1")
        if freeze_features:
            for p in m.features.parameters():
                p.requires_grad = False
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, n_classes)
        return m, 224
    if name == "inception_v3":
        m = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        if m.AuxLogits is not None:
            m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, n_classes)
        return m, 299
    raise ValueError(f"unknown backbone {name}")


def forward_logits(model, x, is_inception, training):
    """Inception in train mode returns (logits, aux_logits)."""
    out = model(x)
    if is_inception and training and isinstance(out, tuple):
        return out[0], out[1]
    if isinstance(out, tuple):
        return out[0], None
    return out, None


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def train_model(model, loader, device, epochs, is_inception, class_weight=None):
    model.to(device).train()
    ce = nn.CrossEntropyLoss(weight=class_weight)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    for e in range(epochs):
        tot = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits, aux = forward_logits(model, x, is_inception, True)
            loss = ce(logits, y)
            if aux is not None:
                loss = loss + 0.4 * ce(aux, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        print(f"  epoch {e+1}/{epochs} loss={tot/len(loader.dataset):.4f}")
    return model


@torch.no_grad()
def eval_model(model, loader, device, is_inception):
    model.to(device).eval()
    ym, pm, yr = [], [], []
    for x, y, rec in loader:
        x = x.to(device)
        logits, _ = forward_logits(model, x, is_inception, False)
        pred = logits.argmax(1).cpu()
        pm += pred.tolist(); ym += y.tolist(); yr += rec.tolist()
    # recyclability via material rule (deterministic) and reported alongside material
    pr = [MATERIAL_RECYCLABLE[i] for i in pm]
    return {
        "material_acc": accuracy_score(ym, pm),
        "material_macro_f1": f1_score(ym, pm, average="macro", zero_division=0),
        "binary_acc": accuracy_score(yr, pr),
        "binary_macro_f1": f1_score(yr, pr, average="macro", zero_division=0),
        "non_recyclable_recall": recall_score(yr, pr, pos_label=0, zero_division=0),
        "material_cm": confusion_matrix(ym, pm, labels=list(range(6))).tolist(),
        "material_report": classification_report(ym, pm, target_names=MATERIALS,
                                                  zero_division=0, output_dict=True),
    }


# ----------------------------------------------------------------------------
def select_rows(df, protocol):
    clean = df[df.domain == "clean"]
    wild = df[df.domain == "wild"]
    if protocol == "in_domain":
        return clean[clean.split == "train"].to_dict("records"), \
               clean[clean.split == "test"].to_dict("records")
    if protocol == "cross_domain":
        return clean.to_dict("records"), wild.to_dict("records")
    if protocol == "joint":
        tr = pd.concat([clean[clean.split == "train"], wild[wild.split == "train"]])
        return tr.to_dict("records"), wild[wild.split == "test"].to_dict("records")
    raise ValueError(protocol)


def class_weights_from(rows, device):
    counts = np.bincount([r["material"] for r in rows], minlength=6).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (6 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "densenet121", "mobilenet_v2", "vgg13", "inception_v3"])
    ap.add_argument("--protocol", choices=["in_domain", "cross_domain", "joint"],
                    default="cross_domain")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--no_domain_randomize", action="store_true")
    ap.add_argument("--class_weighted", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.manifest)
    train_rows, test_rows = select_rows(df, args.protocol)
    print(f"protocol={args.protocol}  train={len(train_rows)}  test={len(test_rows)}")

    results = {}
    for name in args.backbones:
        print(f"\n=== {name} ===")
        model, input_size = build_backbone(name)
        is_inc = name.lower() == "inception_v3"
        tr = DataLoader(ManifestDataset(train_rows, input_size, True,
                                        not args.no_domain_randomize),
                        args.bs, shuffle=True, num_workers=2)
        te = DataLoader(ManifestDataset(test_rows, input_size, False),
                        args.bs, num_workers=2)
        cw = class_weights_from(train_rows, device) if args.class_weighted else None
        train_model(model, tr, device, args.epochs, is_inc, class_weight=cw)
        m = eval_model(model, te, device, is_inc)
        results[name] = m
        print(f"  material_acc={m['material_acc']:.3f} macroF1={m['material_macro_f1']:.3f} "
              f"binary_acc={m['binary_acc']:.3f} non_rec_recall={m['non_recyclable_recall']:.3f}")

    out = args.out or f"results_{args.protocol}.json"
    json.dump({"protocol": args.protocol, "epochs": args.epochs,
               "domain_randomize": not args.no_domain_randomize,
               "class_weighted": args.class_weighted, "results": results},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
