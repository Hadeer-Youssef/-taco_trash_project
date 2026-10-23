"""Console summary and PNG report for dual-head (material + recyclability) runs."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .constants import MATERIALS
from .engine_dual import REC_NAMES
from .reporting import _denorm
from .utils import format_duration, logger


def log_dual_summary(m, history):
    logger.info(f"  material_acc={m['material_acc']:.3f} macroF1={m['material_macro_f1']:.3f}")
    logger.info(f"  recyclability (direct head): acc={m['recyclability_acc']:.3f} "
                f"non_recyclable_recall={m['non_recyclable_recall']:.3f} "
                f"non_recyclable_precision={m['non_recyclable_precision']:.3f} "
                f"auc={m['recyclability_auc_nonrec'] if m['recyclability_auc_nonrec'] is None else round(m['recyclability_auc_nonrec'], 3)}")
    logger.info(f"  recyclability (old rule on the predicted material): "
                f"acc={m['rule_recyclability_acc']:.3f} "
                f"non_recyclable_recall={m['rule_non_recyclable_recall']:.3f}")
    base = m["recyclability_acc_majority_baseline"]
    gain = m["recyclability_acc"] - base
    logger.info(f"  direct recyclability_acc vs. 'always guess majority' baseline ({base:.3f}): "
                f"{'+' if gain >= 0 else ''}{gain:.3f}")
    n, n_nr = m["n_test"], m["n_test_non_recyclable"]
    if n and n_nr:
        logger.info(f"  resolution: 1 test image = {100/n:.2f}% acc; 1 non-recyclable image = "
                    f"{100/n_nr:.1f}% recall ({n} test images, {n_nr} non-recyclable)")
    lat, thr = m["inference_latency_ms_per_image"], m["inference_throughput_images_per_sec"]
    logger.info(f"  train_time={format_duration(history['train_time_sec'])}"
                + (f"  inference_latency={lat:.2f}ms/img  throughput={thr:.1f} img/s"
                   if lat is not None else ""))


def _confusion(ax, cm, labels, title):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    return im


def plot_dual_report(name, history, m, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 17))
    fig.suptitle(f"Result for {name} (material + recyclability heads)", fontsize=16,
                 fontweight="bold")
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(history["train_loss"], label="train_loss")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Number of Epochs"); ax.set_ylabel("combined loss"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for key, lab in (("train_mat_acc", "train material"), ("train_rec_acc", "train recyclability"),
                     ("val_mat_acc", "val material"), ("val_rec_acc", "val recyclability")):
        if history.get(key):
            ax.plot([a * 100 for a in history[key]], label=lab,
                    linestyle="--" if key.startswith("train") else "-")
    ax.set_xlabel("Number of Epochs"); ax.set_ylabel("Accuracy %"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    fig.colorbar(_confusion(ax, m["material_cm"], MATERIALS, f"Material confusion - {name}"),
                 ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    fig.colorbar(_confusion(ax, m["recyclability_cm"], REC_NAMES,
                            f"Recyclability confusion (direct head) - {name}"),
                 ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 0])
    y_nonrec = np.array([int(v == 0) for v in m["y_rec_true"]])
    if 0 < y_nonrec.sum() < len(y_nonrec):
        fpr, tpr, _ = roc_curve(y_nonrec, m["nonrec_probs"])
        ax.plot(fpr, tpr, label=f"non-recyclable detection (AUC = {m['recyclability_auc_nonrec']:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: detecting non-recyclable - {name}"); ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.5, 1.0, "Sample predictions (material / recyclability, true vs predicted)",
            fontsize=9, ha="center", va="bottom", transform=ax.transAxes)
    n_show = min(8, len(m["samples"]))
    if n_show:
        inner = gs[2, 1].subgridspec(3, 4, height_ratios=[0.25, 1, 1], hspace=0.9)
        for k in range(n_show):
            img_t, y_i, p_i, rec_i, rec_p = m["samples"][k]
            a = fig.add_subplot(inner[1 + k // 4, k % 4])
            a.imshow(_denorm(img_t))
            ok = y_i == p_i and rec_i == rec_p
            a.set_title(f"true:{MATERIALS[y_i]} ({REC_NAMES[rec_i]})\n"
                        f"pred:{MATERIALS[p_i]} ({REC_NAMES[rec_p]})",
                        fontsize=6, color="black" if ok else "red", pad=3)
            a.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"report_{name}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    logger.info(f"  saved {path}")
