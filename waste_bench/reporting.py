"""Per-model PNG report and console summary of an evaluation."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from .constants import MATERIALS, MATERIAL_RECYCLABLE, IMAGENET_MEAN, IMAGENET_STD
from .utils import format_duration, logger


def _denorm(img_t):
    img = img_t * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def log_eval_summary(m, history):
    """Headline test metrics, how to read them, and speed."""
    logger.info(f"  material_acc={m['material_acc']:.3f} macroF1={m['material_macro_f1']:.3f} "
                f"recyclability_acc={m['recyclability_acc']:.3f} "
                f"non_recyclable_recall={m['non_recyclable_recall']:.3f}")
    base = m["recyclability_acc_majority_baseline"]
    gain = m["recyclability_acc"] - base
    logger.info(f"  recyclability_acc vs. trivial 'always guess majority class' baseline "
                f"({base:.3f}): {'+' if gain >= 0 else ''}{gain:.3f} "
                + ("-- barely above the trivial baseline, treat recyclability_acc with caution"
                   if gain < 0.03 else "-- clears the trivial baseline"))
    n, n_nr = m["n_test"], m["n_test_non_recyclable"]
    logger.info(f"  read non_recyclable_recall as the trustworthy number: of all truly "
                f"non-recyclable items, the model correctly caught {m['non_recyclable_recall']*100:.1f}% "
                f"of them (higher is always better here, unlike recyclability_acc above)")
    if n and n_nr:
        logger.info(f"  resolution: 1 test image = {100/n:.2f}% material_acc; "
                    f"1 non-recyclable image = {100/n_nr:.1f}% non_recyclable_recall "
                    f"({n} test images, {n_nr} non-recyclable)")
    lat, thr, gpu_i = (m["inference_latency_ms_per_image"], m["inference_throughput_images_per_sec"],
                       m["inference_gpu_util_avg_pct"])
    logger.info(f"  train_time={format_duration(history['train_time_sec'])}"
                + (f"  train_gpu_util_avg={history['gpu_util_avg_pct']:.0f}%"
                   if history.get("gpu_util_avg_pct") is not None else "")
                + (f"  inference_latency={lat:.2f}ms/img  throughput={thr:.1f} img/s"
                   if lat is not None else "")
                + (f"  inference_gpu_util_avg={gpu_i:.0f}%" if gpu_i is not None else ""))


def plot_model_report(name, history, eval_out, out_dir="reports"):
    """5-panel report: loss curve, accuracy curve, classification report,
    confusion matrix, per-class ROC, and a class-balanced sample grid."""
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
    ax.text(0, 1, report_df.round(2).to_string(), family="monospace", fontsize=9, va="top")

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
    ax.text(0.5, 1.0, "Sample predictions (spread across all classes)",
            fontsize=9, ha="center", va="bottom", transform=ax.transAxes)
    n_show = min(12, len(eval_out["samples"]))
    if n_show:
        # height_ratios leaves room for the title; hspace keeps each tile's
        # two-line caption from colliding with the row above
        inner = gs[2, 1].subgridspec(4, 4, height_ratios=[0.25, 1, 1, 1], hspace=0.9)
        rec_label = {1: "recyclable", 0: "non-recyclable"}
        # "true" recyclability is the manual ground-truth label; "pred" is the
        # rule MATERIAL_RECYCLABLE applied to the predicted material
        for k in range(n_show):
            img_t, y_i, p_i, rec_i = eval_out["samples"][k]
            a = fig.add_subplot(inner[1 + k // 4, k % 4])
            a.imshow(_denorm(img_t))
            a.set_title(f"true:{MATERIALS[y_i]} ({rec_label[rec_i]})\n"
                        f"pred:{MATERIALS[p_i]} ({rec_label[MATERIAL_RECYCLABLE[p_i]]})",
                        fontsize=6, color="black" if y_i == p_i else "red", pad=3)
            a.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"report_{name}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    logger.info(f"  saved {path}")
