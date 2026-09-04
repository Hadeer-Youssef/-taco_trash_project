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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

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
@torch.no_grad()
def quick_eval_acc_loss(model, loader, device, is_inception, ce):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = forward_logits(model, x, is_inception, False)
        loss = ce(logits, y)
        tot_loss += loss.item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    model.train()
    return tot_loss / n, tot_correct / n


def train_model(model, loader, device, epochs, is_inception, class_weight=None, val_loader=None):
    model.to(device).train()
    ce = nn.CrossEntropyLoss(weight=class_weight)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for e in range(epochs):
        tot_loss, tot_correct, n = 0.0, 0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits, aux = forward_logits(model, x, is_inception, True)
            loss = ce(logits, y)
            if aux is not None:
                loss = loss + 0.4 * ce(aux, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        train_loss, train_acc = tot_loss / n, tot_correct / n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        msg = f"  epoch {e+1}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f}"
        if val_loader is not None:
            val_loss, val_acc = quick_eval_acc_loss(model, val_loader, device, is_inception, ce)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        print(msg)
    return model, history


@torch.no_grad()
def eval_model(model, loader, device, is_inception, n_samples=12):
    model.to(device).eval()
    ym, pm, yr, probs_all, sample_imgs = [], [], [], [], []
    for x, y, rec in loader:
        x = x.to(device)
        logits, _ = forward_logits(model, x, is_inception, False)
        probs = F.softmax(logits, dim=1).cpu()
        pred = probs.argmax(1)
        pm += pred.tolist(); ym += y.tolist(); yr += rec.tolist()
        probs_all.append(probs)
        if len(sample_imgs) < n_samples:
            take = min(n_samples - len(sample_imgs), x.size(0))
            for i in range(take):
                sample_imgs.append((x[i].cpu(), int(y[i]), int(pred[i])))
    probs_all = torch.cat(probs_all).numpy()
    # recyclability via material rule (deterministic) and reported alongside material
    pr = [MATERIAL_RECYCLABLE[i] for i in pm]
    return {
        "material_acc": accuracy_score(ym, pm),
        "material_macro_f1": f1_score(ym, pm, average="macro", zero_division=0),
        "binary_acc": accuracy_score(yr, pr),
        "binary_macro_f1": f1_score(yr, pr, average="macro", zero_division=0),
        "non_recyclable_recall": recall_score(yr, pr, pos_label=0, zero_division=0),
        "material_cm": confusion_matrix(ym, pm, labels=list(range(6))).tolist(),
        "material_report": classification_report(ym, pm, labels=list(range(6)),
                                                  target_names=MATERIALS,
                                                  zero_division=0, output_dict=True),
        "y_true": ym,
        "y_probs": probs_all,
        "samples": sample_imgs,
    }


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(img_t):
    img = img_t * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_model_report(name, history, eval_out, out_dir="reports"):
    """Reproduces the notebook-style 5-panel report for one trained backbone:
    loss curve, accuracy curve, classification report, confusion matrix
    heatmap, per-class ROC, and a grid of sample predictions."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(eval_out["y_true"])
    y_probs = eval_out["y_probs"]
    cm = np.array(eval_out["material_cm"])
    report_df = pd.DataFrame(eval_out["material_report"]).T

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(f"Result for {name}", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"], label=f"{name}_train_loss")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label=f"{name}_val_loss")
    ax.set_xlabel("Number of Epochs"); ax.set_ylabel("Loss"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot([a * 100 for a in history["train_acc"]], label=f"{name}_train_acc")
    if history["val_acc"]:
        ax.plot([a * 100 for a in history["val_acc"]], label=f"{name}_val_acc")
    ax.set_xlabel("Number of Epochs"); ax.set_ylabel("Accuracy %"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.text(0, 1, report_df.round(2).to_string(), family="monospace", fontsize=9,
            va="top")

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(MATERIALS))); ax.set_xticklabels(MATERIALS, rotation=45, ha="right")
    ax.set_yticks(range(len(MATERIALS))); ax.set_yticklabels(MATERIALS)
    ax.set_title(f"Confusion Matrix Heatmap - {name}")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 0])
    y_true_bin = label_binarize(y_true, classes=list(range(len(MATERIALS))))
    for i, mat in enumerate(MATERIALS):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{mat} (area = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Multi-class ROC Curve - {name}"); ax.legend(fontsize=7, loc="lower right")

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    n_show = min(8, len(eval_out["samples"]))
    if n_show:
        inner = gs[2, 1].subgridspec(2, 4)
        for k in range(n_show):
            img_t, y_i, p_i = eval_out["samples"][k]
            a = fig.add_subplot(inner[k // 4, k % 4])
            a.imshow(_denorm(img_t))
            correct = y_i == p_i
            a.set_title(f"true:{MATERIALS[y_i]}\npred:{MATERIALS[p_i]}",
                        fontsize=7, color="black" if correct else "red")
            a.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"report_{name}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  saved {path}")


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
    ap.add_argument("--report_dir", default="reports",
                    help="where per-model PNG reports (loss/acc curves, "
                         "classification report, confusion matrix, ROC, samples) are saved")
    ap.add_argument("--no_report", action="store_true",
                    help="skip generating the per-model PNG report")
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
        model, history = train_model(model, tr, device, args.epochs, is_inc,
                                      class_weight=cw, val_loader=te)
        m = eval_model(model, te, device, is_inc)
        print(f"  material_acc={m['material_acc']:.3f} macroF1={m['material_macro_f1']:.3f} "
              f"binary_acc={m['binary_acc']:.3f} non_rec_recall={m['non_recyclable_recall']:.3f}")

        if not args.no_report:
            plot_model_report(name, history, m, out_dir=args.report_dir)

        # drop the heavy per-sample arrays before serializing to JSON
        m_json = {k: v for k, v in m.items() if k not in ("y_true", "y_probs", "samples")}
        m_json["history"] = history
        results[name] = m_json

    out = args.out or f"results_{args.protocol}.json"
    json.dump({"protocol": args.protocol, "epochs": args.epochs,
               "domain_randomize": not args.no_domain_randomize,
               "class_weighted": args.class_weighted, "results": results},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
