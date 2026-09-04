"""
Cross-domain backbone benchmark  (Phase 2)
==========================================
Runs four transfer-learning backbones from the TrashNet notebook
(ResNet50, DenseNet121, MobileNetV2, Inception_v3) on the *harmonized*
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
        --backbones resnet50 densenet121 mobilenet_v2 inception_v3 \
        --protocol cross_domain --epochs 15 --out results_p2.json
"""

import os, json, argparse, random, logging, time, threading
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

logger = logging.getLogger("benchmark_backbones")


def setup_logging(log_file=None):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------------------
# GPU utilization sampling (background thread, nvidia-smi via pynvml if present)
# ----------------------------------------------------------------------------
class GpuUtilSampler:
    """Polls GPU utilization % on a background thread while a block runs.
    Silently produces no samples (avg/max report as None) if no NVML/GPU is
    available, so this is always safe to use even on CPU-only machines."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._backend = None
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._pynvml = pynvml
                self._backend = "pynvml"
            except Exception:
                self._backend = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                if self._backend == "pynvml":
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self.samples.append(util.gpu)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._backend is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False

    def summary(self):
        if not self.samples:
            return {"gpu_util_avg_pct": None, "gpu_util_max_pct": None}
        return {"gpu_util_avg_pct": float(np.mean(self.samples)),
                "gpu_util_max_pct": float(np.max(self.samples))}


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
def _mlp_head(in_features, hidden, n_classes, dropout):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def build_backbone(name, n_classes=6, freeze_features=True, finetune_head=False):
    """finetune_head: for each backbone, swaps the plain linear head for a
    2-layer MLP head (in_feats -> hidden -> n_classes, ReLU + dropout) and
    unfreezes roughly the last stage/block of the backbone, instead of a
    fully-frozen backbone + linear probe. The same partial-unfreeze philosophy
    is applied consistently across architectures so results are comparable:
      resnet50      : unfreeze layer4[2] (conv1+bn1),      head 2048->96->6
      mobilenet_v2  : unfreeze features[16:19],            head 1280->256->6
      densenet121   : unfreeze denseblock4 + norm5,        head 1024->256->6
      inception_v3  : unfreeze Mixed_7c,                   head 2048->256->6
    """
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        if finetune_head:
            m.fc = _mlp_head(m.fc.in_features, 96, n_classes, 0.5)
            for p in m.fc.parameters():
                p.requires_grad = True
            blk = m.layer4[2]
            for sub in ["conv1", "bn1"]:
                for p in getattr(blk, sub).parameters():
                    p.requires_grad = True
        else:
            m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m, 224
    if name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        if finetune_head:
            m.classifier = _mlp_head(m.classifier.in_features, 256, n_classes, 0.3)
            for p in m.classifier.parameters():
                p.requires_grad = True
            for sub in ["denseblock4", "norm5"]:
                for p in getattr(m.features, sub).parameters():
                    p.requires_grad = True
        else:
            m.classifier = nn.Linear(m.classifier.in_features, n_classes)
        return m, 224
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1")
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        if finetune_head:
            # unfreeze the last 3 blocks of the feature extractor (features is a
            # nn.Sequential of length 19, indices 0-18)
            for idx in [16, 17, 18]:
                for p in m.features[idx].parameters():
                    p.requires_grad = True
            in_features = m.classifier[1].in_features  # should be 1280
            m.classifier = _mlp_head(in_features, 256, n_classes, 0.3)
            for p in m.classifier.parameters():
                p.requires_grad = True
        else:
            m.classifier = nn.Sequential(nn.Dropout(0.2),
                                         nn.Linear(1280, n_classes))
        return m, 224
    if name == "inception_v3":
        m = models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        if freeze_features:
            for p in m.parameters():
                p.requires_grad = False
        if finetune_head:
            m.fc = _mlp_head(m.fc.in_features, 256, n_classes, 0.3)
            for p in m.fc.parameters():
                p.requires_grad = True
            for p in m.Mixed_7c.parameters():
                p.requires_grad = True
        else:
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
    tot_loss, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = forward_logits(model, x, is_inception, False)
        loss = ce(logits, y)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        y_true += y.tolist()
        y_pred += logits.argmax(1).tolist()
    model.train()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return tot_loss / n, acc, f1


def train_model(model, loader, device, epochs, is_inception, class_weight=None,
                 val_loader=None, patience=None):
    """Trains up to `epochs`. If val_loader and patience are both given, tracks
    the best epoch by validation loss, restores those weights at the end, and
    stops early once `patience` epochs pass with no improvement.
    Also times the whole run and samples GPU utilization throughout, both
    reported in the returned history under train_time_sec / gpu_util_avg_pct."""
    model.to(device).train()
    ce = nn.CrossEntropyLoss(weight=class_weight)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0

    train_start = time.perf_counter()
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for e in range(epochs):
            tot_loss, n = 0.0, 0
            y_true, y_pred = [], []
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                logits, aux = forward_logits(model, x, is_inception, True)
                loss = ce(logits, y)
                if aux is not None:
                    loss = loss + 0.4 * ce(aux, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item() * x.size(0)
                n += x.size(0)
                y_true += y.tolist()
                y_pred += logits.argmax(1).tolist()
            train_loss = tot_loss / n
            train_acc = accuracy_score(y_true, y_pred)
            train_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            msg = (f"Epoch [{e+1}/{epochs}]\n"
                   f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

            if val_loader is not None:
                val_loss, val_acc, val_f1 = quick_eval_acc_loss(model, val_loader, device, is_inception, ce)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)
                msg += f"\n  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"

                if patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss, best_epoch = val_loss, e
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                        msg += "\n  Model improved and saved."
                    else:
                        epochs_no_improve += 1
            logger.info(msg)

            if patience is not None and val_loader is not None and epochs_no_improve >= patience:
                logger.info(f"  early stopping at epoch {e+1} (no improvement for {patience} epochs, "
                            f"best was epoch {best_epoch+1} val_loss={best_val_loss:.4f})")
                break

    train_time_sec = time.perf_counter() - train_start
    gpu_stats = gpu_sampler.summary()
    history["train_time_sec"] = train_time_sec
    history.update(gpu_stats)
    logger.info(f"  training time: {train_time_sec:.1f}s"
                + (f"  gpu_util avg/max: {gpu_stats['gpu_util_avg_pct']:.0f}%/"
                   f"{gpu_stats['gpu_util_max_pct']:.0f}%"
                   if gpu_stats["gpu_util_avg_pct"] is not None else "  gpu_util: n/a"))

    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch + 1
        history["best_val_loss"] = best_val_loss
        logger.info(f"  restored weights from best epoch {best_epoch+1}")

    return model, history


@torch.no_grad()
def eval_model(model, loader, device, is_inception, n_samples=12, warmup_batches=2):
    """Also measures inference latency/throughput on `device`: the first
    `warmup_batches` are excluded from timing (cudnn autotune / lazy init),
    each remaining batch is timed with a cuda synchronize so GPU latency is
    measured accurately rather than just kernel-launch time."""
    model.to(device).eval()
    ym, pm, yr, probs_all, sample_imgs = [], [], [], [], []
    batch_times, n_timed_images = [], 0
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for bi, (x, y, rec) in enumerate(loader):
            x = x.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, _ = forward_logits(model, x, is_inception, False)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if bi >= warmup_batches:
                batch_times.append(elapsed)
                n_timed_images += x.size(0)

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

    total_time = sum(batch_times)
    latency_ms_per_image = (total_time / n_timed_images * 1000) if n_timed_images else None
    throughput_ips = (n_timed_images / total_time) if total_time > 0 else None
    gpu_stats = gpu_sampler.summary()

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
        "inference_latency_ms_per_image": latency_ms_per_image,
        "inference_throughput_images_per_sec": throughput_ips,
        "inference_gpu_util_avg_pct": gpu_stats["gpu_util_avg_pct"],
        "inference_gpu_util_max_pct": gpu_stats["gpu_util_max_pct"],
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
    logger.info(f"  saved {path}")


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
                    default=["resnet50", "densenet121", "mobilenet_v2", "inception_v3"])
    ap.add_argument("--protocol", choices=["in_domain", "cross_domain", "joint"],
                    default="cross_domain")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop if val_loss doesn't improve for this many epochs; "
                         "also enables restoring the best-val-loss epoch's weights. "
                         "Pair with a large --epochs to let it find the best stopping point.")
    ap.add_argument("--finetune_head", action="store_true",
                    help="all backbones: richer 2-layer MLP head (ReLU+dropout) plus "
                         "unfreezing the last backbone stage, instead of a fully-frozen "
                         "backbone + single linear head (see build_backbone docstring "
                         "for the per-architecture unfreeze target)")
    ap.add_argument("--no_domain_randomize", action="store_true")
    ap.add_argument("--class_weighted", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report_dir", default="reports",
                    help="where per-model PNG reports (loss/acc curves, "
                         "classification report, confusion matrix, ROC, samples) are saved")
    ap.add_argument("--no_report", action="store_true",
                    help="skip generating the per-model PNG report")
    ap.add_argument("--log_file", default=None,
                    help="also write the run log to this file (default: "
                         "<out-without-.json>.log next to the results JSON)")
    args = ap.parse_args()

    out = args.out or f"results_{args.protocol}.json"
    log_file = args.log_file or (os.path.splitext(out)[0] + ".log")
    setup_logging(log_file)

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.manifest)
    train_rows, test_rows = select_rows(df, args.protocol)
    logger.info(f"protocol={args.protocol}  train={len(train_rows)}  test={len(test_rows)}  device={device}")

    results = {}
    for name in args.backbones:
        logger.info(f"\n=== {name} ===")
        model, input_size = build_backbone(name, finetune_head=args.finetune_head)
        is_inc = name.lower() == "inception_v3"
        tr = DataLoader(ManifestDataset(train_rows, input_size, True,
                                        not args.no_domain_randomize),
                        args.bs, shuffle=True, num_workers=2)
        te = DataLoader(ManifestDataset(test_rows, input_size, False),
                        args.bs, num_workers=2)
        cw = class_weights_from(train_rows, device) if args.class_weighted else None
        model, history = train_model(model, tr, device, args.epochs, is_inc,
                                      class_weight=cw, val_loader=te, patience=args.patience)
        m = eval_model(model, te, device, is_inc)
        logger.info(f"  material_acc={m['material_acc']:.3f} macroF1={m['material_macro_f1']:.3f} "
                    f"binary_acc={m['binary_acc']:.3f} non_rec_recall={m['non_recyclable_recall']:.3f}")
        lat = m["inference_latency_ms_per_image"]
        thr = m["inference_throughput_images_per_sec"]
        gpu_i = m["inference_gpu_util_avg_pct"]
        logger.info(f"  train_time={history['train_time_sec']:.1f}s"
                    + (f"  train_gpu_util_avg={history['gpu_util_avg_pct']:.0f}%"
                       if history.get("gpu_util_avg_pct") is not None else "")
                    + (f"  inference_latency={lat:.2f}ms/img  throughput={thr:.1f} img/s"
                       if lat is not None else "")
                    + (f"  inference_gpu_util_avg={gpu_i:.0f}%" if gpu_i is not None else ""))

        if not args.no_report:
            plot_model_report(name, history, m, out_dir=args.report_dir)

        # drop the heavy per-sample arrays before serializing to JSON
        m_json = {k: v for k, v in m.items() if k not in ("y_true", "y_probs", "samples")}
        m_json["history"] = history
        results[name] = m_json

    json.dump({"protocol": args.protocol, "epochs": args.epochs,
               "domain_randomize": not args.no_domain_randomize,
               "class_weighted": args.class_weighted, "results": results},
              open(out, "w"), indent=2)
    logger.info(f"\nwrote {out}")
    logger.info(f"wrote {log_file}")


if __name__ == "__main__":
    main()
