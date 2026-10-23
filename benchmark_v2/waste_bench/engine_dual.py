"""Dual-head training and evaluation: one backbone, 6 material logits + 2
recyclability logits, trained with CE(material) + rec_weight * CE(recyclability).

The model is an ordinary build_backbone(n_classes=N_OUTPUTS); this module only
splits its output vector. engine.py is untouched and still serves the
single-head runs.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, classification_report, confusion_matrix)

from .constants import MATERIALS, MATERIAL_RECYCLABLE, N_CLASSES
from .engine import forward_logits, _autocast
from .models import param_groups, set_train_mode
from .utils import GpuUtilSampler, format_duration, logger

N_REC = 2                       # index 0 = non-recyclable, 1 = recyclable (as in the manifest)
N_OUTPUTS = N_CLASSES + N_REC
REC_NAMES = ["non-recyclable", "recyclable"]


def split_logits(out):
    return out[:, :N_CLASSES], out[:, N_CLASSES:]


def class_weights_by(rows, key, n_classes, device, names=None):
    """Inverse-frequency weights for the label stored under `key` in each row."""
    raw = np.bincount([int(r[key]) for r in rows], minlength=n_classes).astype(float)
    counts = raw.copy()
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    if names:
        logger.info("  class weights for %s (from %d training rows): %s", key, len(rows),
                    "  ".join(f"{nm} n={int(c)} w={wi:.3f}" for nm, c, wi in zip(names, raw, w)))
    return torch.tensor(w, dtype=torch.float32, device=device)


def dual_loss(out, y_mat, y_rec, w_mat=None, w_rec=None, label_smoothing=0.0, rec_weight=1.0):
    mat_logits, rec_logits = split_logits(out)
    l_mat = F.cross_entropy(mat_logits, y_mat, weight=w_mat, label_smoothing=label_smoothing)
    l_rec = F.cross_entropy(rec_logits, y_rec, weight=w_rec, label_smoothing=label_smoothing)
    return l_mat + rec_weight * l_rec


@torch.no_grad()
def quick_eval_dual(model, loader, device, is_inception, w_mat=None, w_rec=None,
                    rec_weight=1.0, amp=False):
    """Per-epoch validation: combined loss, accuracy and macro-F1 of both heads."""
    model.eval()
    tot_loss, n = 0.0, 0
    ym, pm, yr, pr = [], [], [], []
    for x, y, rec in loader:
        x, y, rec = x.to(device), y.to(device), rec.to(device)
        with _autocast(device, amp):
            out, _ = forward_logits(model, x, is_inception, False)
            loss = dual_loss(out, y, rec, w_mat, w_rec, 0.0, rec_weight)
        mat_logits, rec_logits = split_logits(out)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)
        ym += y.tolist(); pm += mat_logits.argmax(1).tolist()
        yr += rec.tolist(); pr += rec_logits.argmax(1).tolist()
    return {"loss": tot_loss / n,
            "mat_acc": accuracy_score(ym, pm),
            "mat_f1": f1_score(ym, pm, average="macro", zero_division=0),
            "rec_acc": accuracy_score(yr, pr),
            "rec_f1": f1_score(yr, pr, average="macro", zero_division=0)}


def train_dual(model, loader, device, epochs, is_inception, w_mat=None, w_rec=None,
               rec_weight=1.0, val_loader=None, patience=None, lr_schedule="cosine", lr=3e-4,
               name=None, backbone_lr_mult=1.0, label_smoothing=0.0, monitor="val_loss",
               freeze_bn=False, amp=False):
    """Same recipe as engine.train_model (AdamW, cosine, differential lr, early
    stopping with best-epoch restore) with the dual loss.

    monitor: val_loss = combined dual loss (lower is better);
             val_f1   = mean of material and recyclability macro-F1 (higher is better)."""
    model.to(device)
    set_train_mode(model, freeze_bn)
    groups = (param_groups(model, name, lr, backbone_lr_mult) if name
              else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if lr_schedule == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda") if (amp and device == "cuda") else None
    history = {"train_loss": [], "train_mat_acc": [], "train_rec_acc": [],
               "val_loss": [], "val_mat_acc": [], "val_rec_acc": [], "val_f1": [], "lr": []}

    higher_better = monitor == "val_f1"
    best_score = float("-inf") if higher_better else float("inf")
    best_epoch, best_state, epochs_no_improve = -1, None, 0

    train_start = time.perf_counter()
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for e in range(epochs):
            set_train_mode(model, freeze_bn)
            tot_loss, n, ok_mat, ok_rec = 0.0, 0, 0, 0
            for x, y, rec in loader:
                x = x.to(device, non_blocking=True)
                y, rec = y.to(device, non_blocking=True), rec.to(device, non_blocking=True)
                with _autocast(device, amp):
                    out, aux = forward_logits(model, x, is_inception, True)
                    loss = dual_loss(out, y, rec, w_mat, w_rec, label_smoothing, rec_weight)
                    if aux is not None:
                        loss = loss + 0.4 * dual_loss(aux, y, rec, w_mat, w_rec,
                                                      label_smoothing, rec_weight)
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                mat_logits, rec_logits = split_logits(out)
                tot_loss += loss.item() * x.size(0)
                n += x.size(0)
                ok_mat += (mat_logits.argmax(1) == y).sum().item()
                ok_rec += (rec_logits.argmax(1) == rec).sum().item()

            cur_lr = opt.param_groups[0]["lr"]
            history["train_loss"].append(tot_loss / n)
            history["train_mat_acc"].append(ok_mat / n)
            history["train_rec_acc"].append(ok_rec / n)
            history["lr"].append(cur_lr)
            if scheduler is not None:
                scheduler.step()
            msg = (f"Epoch [{e+1}/{epochs}]  lr={cur_lr:.2e}\n"
                   f"  Train Loss: {history['train_loss'][-1]:.4f}, "
                   f"material acc: {history['train_mat_acc'][-1]:.4f}, "
                   f"recyclability acc: {history['train_rec_acc'][-1]:.4f}")

            if val_loader is not None:
                v = quick_eval_dual(model, val_loader, device, is_inception, w_mat, w_rec,
                                    rec_weight, amp)
                val_f1 = (v["mat_f1"] + v["rec_f1"]) / 2
                history["val_loss"].append(v["loss"])
                history["val_mat_acc"].append(v["mat_acc"])
                history["val_rec_acc"].append(v["rec_acc"])
                history["val_f1"].append(val_f1)
                msg += (f"\n  Val   Loss: {v['loss']:.4f}, material acc: {v['mat_acc']:.4f}, "
                        f"recyclability acc: {v['rec_acc']:.4f}, mean F1: {val_f1:.4f}")

                if patience is not None:
                    score = val_f1 if higher_better else v["loss"]
                    improved = score > best_score if higher_better else score < best_score
                    if improved:
                        best_score, best_epoch = score, e
                        best_state = {k: t.detach().cpu().clone()
                                      for k, t in model.state_dict().items()}
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
        logger.info(f"  restored weights from best epoch {best_epoch+1}")
    return model, history


@torch.no_grad()
def eval_dual(model, loader, device, is_inception, n_samples=8, warmup_batches=2,
              samples_per_class=2, tta=False, amp=False):
    """Final test metrics. recyclability_* come from the recyclability head;
    rule_* are what the old material->recyclable rule would have scored on the
    same predictions, so one run shows both side by side."""
    model.to(device).eval()
    ym, pm_all, yr, pr_all = [], [], [], []
    mat_probs, nonrec_probs = [], []
    per_class = {c: [] for c in range(N_CLASSES)}
    batch_times, n_timed = [], 0
    gpu_sampler = GpuUtilSampler()
    with gpu_sampler:
        for bi, (x, y, rec) in enumerate(loader):
            x = x.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with _autocast(device, amp):
                out, _ = forward_logits(model, x, is_inception, False)
                m_logits, r_logits = split_logits(out)
                p_mat, p_rec = F.softmax(m_logits.float(), 1), F.softmax(r_logits.float(), 1)
                if tta:
                    out_f, _ = forward_logits(model, torch.flip(x, dims=[3]), is_inception, False)
                    mf, rf = split_logits(out_f)
                    p_mat = (p_mat + F.softmax(mf.float(), 1)) / 2
                    p_rec = (p_rec + F.softmax(rf.float(), 1)) / 2
            if device == "cuda":
                torch.cuda.synchronize()
            if bi >= warmup_batches:
                batch_times.append(time.perf_counter() - t0)
                n_timed += x.size(0)

            pm, pr = p_mat.argmax(1).cpu(), p_rec.argmax(1).cpu()
            ym += y.tolist(); pm_all += pm.tolist()
            yr += rec.tolist(); pr_all += pr.tolist()
            mat_probs.append(p_mat.cpu())
            nonrec_probs.append(p_rec[:, 0].cpu())
            for i in range(x.size(0)):
                y_i = int(y[i])
                if len(per_class[y_i]) < samples_per_class:
                    per_class[y_i].append((x[i].cpu(), y_i, int(pm[i]), int(rec[i]), int(pr[i])))

    mat_probs = torch.cat(mat_probs).numpy()
    nonrec_probs = torch.cat(nonrec_probs).numpy()
    rule = [MATERIAL_RECYCLABLE[i] for i in pm_all]
    majority = 1 if sum(yr) >= len(yr) / 2 else 0
    total_time = sum(batch_times)
    gpu_stats = gpu_sampler.summary()
    try:
        auc_nonrec = float(roc_auc_score([int(v == 0) for v in yr], nonrec_probs))
    except ValueError:      # only one recyclability class in the test split
        auc_nonrec = None

    return {
        "dual_head": True,
        "material_acc": accuracy_score(ym, pm_all),
        "material_macro_f1": f1_score(ym, pm_all, average="macro", zero_division=0),
        "recyclability_acc": accuracy_score(yr, pr_all),
        "recyclability_macro_f1": f1_score(yr, pr_all, average="macro", zero_division=0),
        "non_recyclable_recall": recall_score(yr, pr_all, pos_label=0, zero_division=0),
        "non_recyclable_precision": precision_score(yr, pr_all, pos_label=0, zero_division=0),
        "recyclability_auc_nonrec": auc_nonrec,
        "recyclability_acc_majority_baseline": sum(v == majority for v in yr) / len(yr),
        "rule_recyclability_acc": accuracy_score(yr, rule),
        "rule_non_recyclable_recall": recall_score(yr, rule, pos_label=0, zero_division=0),
        "n_test": len(ym),
        "n_test_non_recyclable": int(sum(v == 0 for v in yr)),
        "material_cm": confusion_matrix(ym, pm_all, labels=list(range(N_CLASSES))).tolist(),
        "recyclability_cm": confusion_matrix(yr, pr_all, labels=[0, 1]).tolist(),
        "material_report": classification_report(ym, pm_all, labels=list(range(N_CLASSES)),
                                                 target_names=MATERIALS, zero_division=0,
                                                 output_dict=True),
        "inference_latency_ms_per_image": (total_time / n_timed * 1000) if n_timed else None,
        "inference_throughput_images_per_sec": (n_timed / total_time) if total_time > 0 else None,
        "inference_gpu_util_avg_pct": gpu_stats["gpu_util_avg_pct"],
        "inference_gpu_util_max_pct": gpu_stats["gpu_util_max_pct"],
        "y_true": ym, "y_rec_true": yr,
        "y_probs": mat_probs, "nonrec_probs": nonrec_probs,
        "samples": [s for cls in per_class.values() for s in cls][:n_samples],
    }
