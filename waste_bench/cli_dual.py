"""Command line for dual-head runs: material (6) + recyclability (2) from one backbone.

Same flags as cli.py (it reuses its parser) plus --rec_weight. Outputs go to
runs/<backbone>/<tag>_dual/ so they never mix with single-head runs.
"""

import json
import os

import torch

from .cli import build_parser, _log_run_header
from .config import RunConfig, run_tag
from .constants import MATERIALS
from .data import load_manifest, select_rows, make_loader
from .engine_dual import (N_OUTPUTS, REC_NAMES, class_weights_by, train_dual, eval_dual,
                          quick_eval_dual)
from .constants import N_CLASSES
from .models import build_backbone, count_trainable
from .reporting_dual import plot_dual_report, log_dual_summary
from .utils import logger, setup_logging, set_seed


def parse_args(argv=None):
    ap = build_parser()
    ap.add_argument("--rec_weight", type=float, default=1.0,
                    help="weight of the recyclability loss relative to the material loss "
                         "(0 = train material only, the recyclability head stays untrained)")
    ap.add_argument("--no_rec_class_weights", action="store_true",
                    help="with --class_weighted, weight only the material loss; the "
                         "recyclability loss stays unweighted (in cross_domain the clean "
                         "training set has 5%% non-recyclable, so its inverse-frequency "
                         "weight is ~17:1)")
    kw = vars(ap.parse_args(argv))
    rec_weight = kw.pop("rec_weight")
    no_rec_cw = kw.pop("no_rec_class_weights")
    if kw.pop("no_domain_randomize"):
        kw["aug_mode"] = "off"
    cfg = RunConfig(**kw)
    if cfg.ecoc_head:
        raise SystemExit("--ecoc_head is not supported with the dual head")
    return cfg, rec_weight, no_rec_cw


def dual_tag(cfg, rec_weight, no_rec_cw=False):
    return (run_tag(cfg) + "_dual" + (f"-rw{rec_weight:g}" if rec_weight != 1.0 else "")
            + ("-norcw" if no_rec_cw else ""))


def run_backbone_dual(name, cfg, rec_weight, no_rec_cw, tag, device, rows):
    train_rows, val_rows, test_rows = rows
    logger.info(f"\n=== {name} (dual head: {N_CLASSES} materials + 2 recyclability) ===")
    model, input_size = build_backbone(name, n_classes=N_OUTPUTS, finetune_head=cfg.finetune_head,
                                       unfreeze=cfg.resolved_unfreeze(),
                                       head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout)
    logger.info(f"  unfreeze={cfg.resolved_unfreeze()}  trainable params: {count_trainable(model):,}"
                + (f"  backbone lr = {cfg.lr * cfg.backbone_lr_mult:.1e}"
                   if cfg.backbone_lr_mult != 1.0 else "")
                + f"  rec_weight={rec_weight:g}")

    aug = cfg.resolved_aug_mode()
    train_loader = make_loader(train_rows, input_size, True, aug, cfg.bs, cfg.workers)
    test_loader = make_loader(test_rows, input_size, False, aug, cfg.bs, cfg.workers)
    val_loader = (make_loader(val_rows, input_size, False, aug, cfg.bs, cfg.workers)
                  if val_rows else None)
    select_loader = val_loader if cfg.resolved_select_on() == "val" else test_loader
    is_inc = name == "inception_v3"
    w_mat = w_rec = None
    if cfg.class_weighted:
        w_mat = class_weights_by(train_rows, "material", N_CLASSES, device, MATERIALS)
        if not no_rec_cw:
            w_rec = class_weights_by(train_rows, "recyclable", 2, device, REC_NAMES)

    model, history = train_dual(
        model, train_loader, device, cfg.epochs, is_inc, w_mat, w_rec, rec_weight,
        val_loader=select_loader, patience=cfg.patience, lr_schedule=cfg.lr_schedule,
        lr=cfg.lr, name=name, backbone_lr_mult=cfg.backbone_lr_mult,
        label_smoothing=cfg.label_smoothing, monitor=cfg.monitor,
        freeze_bn=cfg.freeze_bn, amp=cfg.amp)

    m = eval_dual(model, test_loader, device, is_inc, tta=cfg.tta, amp=cfg.amp)
    if val_loader is not None:
        v = quick_eval_dual(model, val_loader, device, is_inc, amp=cfg.amp)
        m["val_acc"], m["val_macro_f1"] = v["mat_acc"], v["mat_f1"]
        m["val_recyclability_acc"] = v["rec_acc"]
        logger.info(f"  val (selection split) material acc={v['mat_acc']:.3f} "
                    f"recyclability acc={v['rec_acc']:.3f}")
    log_dual_summary(m, history)

    run_dir = os.path.join(cfg.runs_dir, name, tag)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"  run folder: {run_dir}")
    if not cfg.no_report:
        plot_dual_report(name, history, m, out_dir=cfg.report_dir or run_dir)

    if not cfg.no_ckpt:
        ckpt_dir = cfg.ckpt_dir or run_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        torch.save({
            "backbone": name, "dual_head": True, "n_outputs": N_OUTPUTS, "rec_weight": rec_weight,
            "finetune_head": cfg.finetune_head, "unfreeze": cfg.resolved_unfreeze(),
            "head_hidden": cfg.head_hidden, "head_dropout": cfg.head_dropout,
            "input_size": input_size, "materials": MATERIALS, "rec_names": REC_NAMES,
            "protocol": cfg.protocol, "aug_mode": aug, "class_weighted": cfg.class_weighted,
            "lr": cfg.lr, "lr_schedule": cfg.lr_schedule, "tag": tag,
            "material_acc": m["material_acc"], "recyclability_acc": m["recyclability_acc"],
            "non_recyclable_recall": m["non_recyclable_recall"],
            "best_epoch": history.get("best_epoch"),
            "model_state_dict": model.state_dict(),
        }, ckpt_path)
        logger.info(f"  saved checkpoint {ckpt_path}  (best epoch: "
                    f"{history.get('best_epoch', cfg.epochs)})")

    skip = ("y_true", "y_rec_true", "y_probs", "nonrec_probs", "samples")
    m_json = {k: v for k, v in m.items() if k not in skip}
    m_json["history"] = history
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"backbone": name, "tag": tag, "config": cfg.to_dict(), "rec_weight": rec_weight,
                   "protocol": cfg.protocol, "epochs": cfg.epochs, "aug_mode": aug,
                   "finetune_head": cfg.finetune_head, "class_weighted": cfg.class_weighted,
                   "lr": cfg.lr, "lr_schedule": cfg.lr_schedule,
                   "results": m_json}, f, indent=2)
    return m_json


def run(cfg, rec_weight, no_rec_cw=False):
    tag = dual_tag(cfg, rec_weight, no_rec_cw)
    out = cfg.out or os.path.join(cfg.runs_dir, "_summaries", f"{tag}.json")
    log_file = cfg.log_file or os.path.join(cfg.runs_dir, "_summaries", f"{tag}.log")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    setup_logging(log_file)
    logger.info(f"run tag: {tag}")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = select_rows(load_manifest(cfg.manifest), cfg.protocol)
    if cfg.resolved_select_on() == "val" and not rows[1]:
        raise SystemExit(f"--select_on val: protocol {cfg.protocol} has no val split")
    _log_run_header(cfg, device, len(rows[0]), len(rows[1] or []), len(rows[2]))

    results = {}
    for name in cfg.backbones:
        set_seed(cfg.seed)
        results[name] = run_backbone_dual(name, cfg, rec_weight, no_rec_cw, tag, device, rows)

    with open(out, "w") as f:
        json.dump({"tag": tag, "config": cfg.to_dict(), "rec_weight": rec_weight,
                   "protocol": cfg.protocol, "epochs": cfg.epochs,
                   "aug_mode": cfg.resolved_aug_mode(), "class_weighted": cfg.class_weighted,
                   "results": results}, f, indent=2)
    logger.info(f"\nwrote {out}")
    logger.info(f"wrote {log_file}")


def main(argv=None):
    run(*parse_args(argv))


if __name__ == "__main__":
    main()
