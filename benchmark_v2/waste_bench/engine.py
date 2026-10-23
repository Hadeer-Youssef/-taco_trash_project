"""Training loop, early stopping and evaluation (metrics, latency, TTA)."""

import contextlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix)

from .constants import MATERIALS, MATERIAL_RECYCLABLE, N_CLASSES
from .models import param_groups, set_train_mode
from .utils import GpuUtilSampler, format_duration, logger


def forward_logits(model, x, is_inception, training):
    """Inception in train mode returns (logits, aux_logits)."""
    out = model(x)
    if is_inception and training and isinstance(out, tuple):
        return out[0], out[1]
    if isinstance(out, tuple):
        return out[0], None
    return out, None


def predict_from_logits(raw_out, ecoc_head=None):
    """(pred class index, per-class probabilities) for either head type."""
    if ecoc_head is None:
        probs = F.softmax(raw_out.float(), dim=1)
        return probs.argmax(1), probs
    return ecoc_head.decode(raw_out)


def loss_and_pred(raw_out, y, ecoc_head=None, class_weight=None, label_smoothing=0.0):
    """Single branch point between the two head types:
      softmax: CrossEntropy (optionally class-weighted / label-smoothed)
      ECOC   : BCE of each bit against the true class's codebook row, with the
               class weight (if any) applied per sample.
    Returns (loss, pred, probs)."""
    if ecoc_head is None:
        loss = nn.functional.cross_entropy(raw_out, y, weight=class_weight,
                                           label_smoothing=label_smoothing)
    else:
        targets = ecoc_head.codebook.to(raw_out.device)[y]
        per_bit = nn.functional.binary_cross_entropy_with_logits(raw_out, targets, reduction="none")
        if class_weight is not None:
            per_bit = per_bit * class_weight.to(raw_out.device)[y].unsqueeze(1)
        loss = per_bit.mean()
    pred, probs = predict_from_logits(raw_out, ecoc_head)
    return loss, pred, probs


def _autocast(device, amp):
    if amp and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def quick_eval(model, loader, device, is_inception, class_weight=None, ecoc_head=None, amp=False):
    """(loss, accuracy, macro-F1) on a loader — used every epoch for early stopping."""
    model.eval()
    tot_loss, n, y_true, y_pred = 0.0, 0, [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(device, amp):
            logits, _ = forward_logits(model, x, is_inception, False)
            loss, pred, _ = loss_and_pred(logits, y, ecoc_head, class_weight)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        y_true += y.tolist()
        y_pred += pred.tolist()
    return (tot_loss / n, accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


# kept under its old name for any external caller
quick_eval_acc_loss = quick_eval


def train_model(model, loader, device, epochs, is_inception, class_weight=None,
                val_loader=None, patience=None, lr_schedule="cosine", lr=3e-4,
                ecoc_head=None, name=None, backbone_lr_mult=1.0, label_smoothing=0.0,
                monitor="val_loss", freeze_bn=False, amp=False):
    """Trains up to `epochs`, evaluating on `val_loader` after every epoch.

    With `patience`, keeps the weights of the best epoch by `monitor`
    (val_loss: lower is better; val_f1: higher is better), stops after
    `patience` epochs without improvement, and restores the best weights.
    Prefer val_f1 with --ecoc_head: its BCE loss is on a different scale from
    cross-entropy and does not track decoded-class accuracy closely.

    lr_schedule="cosine" decays every param group's lr to ~0 over `epochs`;
    backbone_lr_mult scales the lr of unfrozen backbone layers relative to the
    head (needs `name` to tell them apart)."""
    model.to(device)
    set_train_mode(model, freeze_bn)
    groups = (param_groups(model, name, lr, backbone_lr_mult) if name
              else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if lr_schedule == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda") if (amp and device == "cuda") else None
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    higher_better = monitor == "val_f1"
    best_score = float("-inf") if higher_better else float("inf")
    best_epoch, best_state, epochs_no_improve = -1, None, 0

    train_start = time.perf_counter()
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for e in range(epochs):
            set_train_mode(model, freeze_bn)
            tot_loss, n, y_true, y_pred = 0.0, 0, [], []
            for x, y, _ in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with _autocast(device, amp):
                    logits, aux = forward_logits(model, x, is_inception, True)
                    loss, pred, _ = loss_and_pred(logits, y, ecoc_head, class_weight, label_smoothing)
                    if aux is not None and ecoc_head is None:
                        aux_loss, _, _ = loss_and_pred(aux, y, None, class_weight, label_smoothing)
                        loss = loss + 0.4 * aux_loss
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                tot_loss += loss.item() * x.size(0)
                n += x.size(0)
                y_true += y.tolist()
                y_pred += pred.tolist()

            cur_lr = opt.param_groups[0]["lr"]
            history["train_loss"].append(tot_loss / n)
            history["train_acc"].append(accuracy_score(y_true, y_pred))
            history["train_f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
            history["lr"].append(cur_lr)
            if scheduler is not None:
                scheduler.step()
            msg = (f"Epoch [{e+1}/{epochs}]  lr={cur_lr:.2e}\n"
                   f"  Train Loss: {history['train_loss'][-1]:.4f}, "
                   f"Acc: {history['train_acc'][-1]:.4f}, F1: {history['train_f1'][-1]:.4f}")

            if val_loader is not None:
                val_loss, val_acc, val_f1 = quick_eval(model, val_loader, device, is_inception,
                                                       class_weight, ecoc_head, amp)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_f1"].append(val_f1)
                msg += f"\n  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"

                if patience is not None:
                    score = val_f1 if higher_better else val_loss
                    improved = score > best_score if higher_better else score < best_score
                    if improved:
                        best_score, best_epoch = score, e
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                        msg += "\n  Model improved and saved."
                    else:
                        epochs_no_improve += 1
            logger.info(msg)

            if patience is not None and val_loader is not None and epochs_no_improve >= patience:
                logger.info(f"  early stopping at epoch {e+1} (no improvement for {patience} epochs, "
                            f"best was epoch {best_epoch+1} {monitor}={best_score:.4f})")
                break

    history["train_time_sec"] = time.perf_counter() - train_start
    gpu_stats = gpu_sampler.summary()
    history.update(gpu_stats)
    logger.info(f"  training time: {format_duration(history['train_time_sec'])}"
                + (f"  gpu_util avg/max: {gpu_stats['gpu_util_avg_pct']:.0f}%/"
                   f"{gpu_stats['gpu_util_max_pct']:.0f}%"
                   if gpu_stats["gpu_util_avg_pct"] is not None else "  gpu_util: n/a"))

    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch + 1
        history["best_monitor"] = monitor
        history["best_score"] = best_score
        if monitor == "val_loss":
            history["best_val_loss"] = best_score
        logger.info(f"  restored weights from best epoch {best_epoch+1}")
    return model, history


@torch.no_grad()
def eval_model(model, loader, device, is_inception, n_samples=12, warmup_batches=2,
               samples_per_class=2, ecoc_head=None, tta=False, amp=False):
    """Final evaluation: material metrics, recyclability derived by rule,
    inference latency/throughput, and a class-balanced grid of sample images.

    tta=True averages the class probabilities of each image and its
    horizontal flip. Timing excludes `warmup_batches` and synchronises CUDA
    so it measures real GPU time, not kernel-launch time."""
    model.to(device).eval()
    material_true, material_pred, recyclable_true, probs_all = [], [], [], []
    per_class_samples = {c: [] for c in range(N_CLASSES)}
    batch_times, n_timed_images = [], 0
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for bi, (x, y, rec) in enumerate(loader):
            x = x.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with _autocast(device, amp):
                logits, _ = forward_logits(model, x, is_inception, False)
                pred, probs = predict_from_logits(logits, ecoc_head)
                if tta:
                    logits_f, _ = forward_logits(model, torch.flip(x, dims=[3]), is_inception, False)
                    _, probs_f = predict_from_logits(logits_f, ecoc_head)
                    probs = (probs + probs_f) / 2
                    pred = probs.argmax(1)
            if device == "cuda":
                torch.cuda.synchronize()
            if bi >= warmup_batches:
                batch_times.append(time.perf_counter() - t0)
                n_timed_images += x.size(0)

            pred, probs = pred.cpu(), probs.float().cpu()
            material_pred += pred.tolist()
            material_true += y.tolist()
            recyclable_true += rec.tolist()
            probs_all.append(probs)
            # manifest.csv is ordered by class, so collect a few samples per
            # class instead of the first N rows (which would be one class)
            for i in range(x.size(0)):
                y_i = int(y[i])
                if len(per_class_samples[y_i]) < samples_per_class:
                    per_class_samples[y_i].append((x[i].cpu(), y_i, int(pred[i]), int(rec[i])))

    probs_all = torch.cat(probs_all).numpy()
    recyclable_from_rule = [MATERIAL_RECYCLABLE[i] for i in material_pred]
    sample_imgs = [s for cls in per_class_samples.values() for s in cls][:n_samples]
    total_time = sum(batch_times)
    gpu_stats = gpu_sampler.summary()
    # accuracy of always guessing the majority recyclability class — the floor
    # recyclability_acc must clear before it says anything about learning
    majority = 1 if sum(recyclable_true) >= len(recyclable_true) / 2 else 0
    majority_acc = sum(v == majority for v in recyclable_true) / len(recyclable_true)

    return {
        "material_acc": accuracy_score(material_true, material_pred),
        "material_macro_f1": f1_score(material_true, material_pred, average="macro", zero_division=0),
        "recyclability_acc": accuracy_score(recyclable_true, recyclable_from_rule),
        "recyclability_acc_majority_baseline": majority_acc,
        "recyclability_macro_f1": f1_score(recyclable_true, recyclable_from_rule,
                                           average="macro", zero_division=0),
        "non_recyclable_recall": recall_score(recyclable_true, recyclable_from_rule,
                                              pos_label=0, zero_division=0),
        "n_test": len(material_true),
        "n_test_non_recyclable": int(sum(v == 0 for v in recyclable_true)),
        "material_cm": confusion_matrix(material_true, material_pred,
                                        labels=list(range(N_CLASSES))).tolist(),
        "material_report": classification_report(material_true, material_pred,
                                                 labels=list(range(N_CLASSES)),
                                                 target_names=MATERIALS, zero_division=0,
                                                 output_dict=True),
        "inference_latency_ms_per_image": (total_time / n_timed_images * 1000) if n_timed_images else None,
        "inference_throughput_images_per_sec": (n_timed_images / total_time) if total_time > 0 else None,
        "inference_gpu_util_avg_pct": gpu_stats["gpu_util_avg_pct"],
        "inference_gpu_util_max_pct": gpu_stats["gpu_util_max_pct"],
        "y_true": material_true,
        "y_probs": probs_all,
        "samples": sample_imgs,
    }
