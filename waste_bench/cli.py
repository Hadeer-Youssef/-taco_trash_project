"""Command line: flags -> RunConfig -> train/evaluate every backbone -> save."""

import argparse
import json
import os

import torch

from .config import RunConfig, run_tag, UNFREEZE_LEVELS
from .constants import BACKBONES, MATERIALS, MATERIAL_RECYCLABLE
from .data import load_manifest, select_rows, make_loader, class_weights_from
from .engine import train_model, eval_model, quick_eval
from .models import build_backbone, attach_ecoc_head, get_head, count_trainable
from .reporting import plot_model_report, log_eval_summary
from .utils import logger, setup_logging, set_seed


def build_parser():
    ap = argparse.ArgumentParser(description="Harmonized TACO+TrashNet backbone benchmark")
    d = RunConfig()
    g = ap.add_argument_group("data / protocol")
    g.add_argument("--manifest", default=d.manifest)
    g.add_argument("--protocol", choices=["in_domain", "cross_domain", "joint"], default=d.protocol)
    g.add_argument("--backbones", nargs="+", choices=BACKBONES, default=d.backbones)
    g.add_argument("--select_on", choices=["auto", "val", "test"], default=d.select_on,
                   help="split used to pick the best epoch / early stop. 'auto' = the "
                        "held-out val split for in_domain and joint (test is then only "
                        "used for the final number), test for cross_domain (no labelled "
                        "target val split). 'test' reproduces runs made before this option.")

    g = ap.add_argument_group("model")
    g.add_argument("--finetune_head", action="store_true",
                   help="MLP head (in->hidden->6, ReLU+dropout) and, unless --unfreeze says "
                        "otherwise, unfreeze the last backbone block")
    g.add_argument("--head_hidden", type=int, nargs="+", default=d.head_hidden,
                   help="custom head: hidden layer widths on top of the pretrained features, "
                        "each Linear->BatchNorm->ReLU->Dropout, e.g. --head_hidden 512 256 "
                        "(resnet50: 2048->512->256->6). Overrides --finetune_head's fixed MLP")
    g.add_argument("--head_dropout", type=float, default=d.head_dropout,
                   help="dropout inside the custom head (default 0.3)")
    g.add_argument("--unfreeze", choices=UNFREEZE_LEVELS, default=d.unfreeze,
                   help="how much of the pretrained backbone to train: none (head only), "
                        "last (last block), stage (last stage), two_stages, all. 'auto' = "
                        "last with --finetune_head, none without. Pair deeper levels with "
                        "--backbone_lr_mult 0.1")
    g.add_argument("--freeze_bn", action="store_true",
                   help="keep BatchNorm running stats of frozen layers fixed (by default "
                        "they keep updating even when the layer's weights are frozen)")
    g.add_argument("--ecoc_head", action="store_true",
                   help="ECOC head from Wang, Heliyon 2024 (see paper_household_garbage_ecoc_capsa.md)")
    g.add_argument("--ecoc_codebook", choices=["ordinal", "onevsrest"], default=d.ecoc_codebook)

    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=d.epochs)
    g.add_argument("--bs", type=int, default=d.bs)
    g.add_argument("--lr", type=float, default=d.lr, help="lr of the head (peak lr with cosine)")
    g.add_argument("--lr_schedule", choices=["cosine", "none"], default=d.lr_schedule)
    g.add_argument("--backbone_lr_mult", type=float, default=d.backbone_lr_mult,
                   help="lr of unfrozen backbone layers = --lr x this (e.g. 0.1)")
    g.add_argument("--label_smoothing", type=float, default=d.label_smoothing,
                   help="cross-entropy label smoothing, e.g. 0.1 (softmax head only)")
    g.add_argument("--class_weighted", action="store_true")
    g.add_argument("--patience", type=int, default=d.patience,
                   help="early stop after this many epochs without improvement; also "
                        "restores the best epoch's weights")
    g.add_argument("--monitor", choices=["val_loss", "val_f1"], default=d.monitor,
                   help="early-stopping criterion (val_f1 recommended with --ecoc_head)")
    g.add_argument("--aug_mode", choices=["auto", "off", "mild", "strong"], default=d.aug_mode,
                   help="train augmentation; auto = mild for in_domain, strong otherwise")
    g.add_argument("--no_domain_randomize", action="store_true",
                   help="deprecated alias for --aug_mode off")
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--amp", action="store_true", help="mixed precision on CUDA (faster)")

    g = ap.add_argument_group("evaluation")
    g.add_argument("--tta", action="store_true",
                   help="average predictions over each test image and its horizontal flip")

    g = ap.add_argument_group("outputs")
    g.add_argument("--runs_dir", default=d.runs_dir,
                   help="outputs go to <runs_dir>/<backbone>/<tag>/ (report, checkpoint, "
                        "results.json) and <runs_dir>/_summaries/<tag>.{json,log}")
    g.add_argument("--out", default=None, help="override path of the summary JSON")
    g.add_argument("--report_dir", default=None, help="override folder of the report PNG")
    g.add_argument("--log_file", default=None, help="override path of the run log")
    g.add_argument("--ckpt_dir", default=None, help="override folder of checkpoint.pt")
    g.add_argument("--no_report", action="store_true")
    g.add_argument("--no_ckpt", action="store_true")
    return ap


def config_from_args(argv=None):
    args = build_parser().parse_args(argv)
    kw = vars(args)
    if kw.pop("no_domain_randomize"):
        kw["aug_mode"] = "off"
    return RunConfig(**kw)


def _log_run_header(cfg, device, n_train, n_val, n_test):
    aug = cfg.resolved_aug_mode()
    logger.info(f"protocol={cfg.protocol}  device={device}  aug_mode={aug}"
                + ("  (auto-selected)" if cfg.aug_mode == "auto" else "") + f"  seed={cfg.seed}")
    logger.info(f"  train samples: {n_train}  (augmentation: {aug}, applied on-the-fly each "
                f"epoch -- sample COUNT does not change)")
    logger.info(f"  val samples: {n_val}  test samples: {n_test}  (never augmented)")
    sel = cfg.resolved_select_on()
    if sel == "test":
        logger.info("  model selection on: TEST split -- best epoch is chosen on the same data "
                    "that is reported, so test numbers are slightly optimistic"
                    + (" (cross_domain has no labelled target-domain val split)"
                       if cfg.protocol == "cross_domain" else ""))
    else:
        logger.info("  model selection on: VAL split -- test is used only for the final numbers")
    logger.info(f"  effective training views over {cfg.epochs} epochs: "
                f"{n_train} x {cfg.epochs} = {n_train * cfg.epochs}")
    if cfg.ecoc_head and cfg.label_smoothing:
        logger.info("  note: --label_smoothing only applies to the softmax head; ignored with ECOC")
    if cfg.ecoc_head and cfg.monitor == "val_loss" and cfg.patience:
        logger.info("  note: with --ecoc_head, --monitor val_f1 is a better early-stopping "
                    "signal (BCE loss does not track decoded-class accuracy closely)")


def run_backbone(name, cfg, tag, device, rows):
    train_rows, val_rows, test_rows = rows
    logger.info(f"\n=== {name} ===")
    model, input_size = build_backbone(name, finetune_head=cfg.finetune_head,
                                       unfreeze=cfg.resolved_unfreeze(),
                                       head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout)
    ecoc_head = None
    if cfg.ecoc_head:
        model = attach_ecoc_head(model, name, codebook_type=cfg.ecoc_codebook)
        ecoc_head = get_head(model, name)
        logger.info(f"  ecoc_head: {ecoc_head.codebook.shape[1]} binary classifiers "
                    f"({cfg.ecoc_codebook} codebook), replacing the softmax head")
    logger.info(f"  unfreeze={cfg.resolved_unfreeze()}  trainable params: {count_trainable(model):,}"
                + (f"  backbone lr = {cfg.lr * cfg.backbone_lr_mult:.1e}"
                   if cfg.backbone_lr_mult != 1.0 else ""))

    aug = cfg.resolved_aug_mode()
    train_loader = make_loader(train_rows, input_size, True, aug, cfg.bs)
    test_loader = make_loader(test_rows, input_size, False, aug, cfg.bs)
    val_loader = make_loader(val_rows, input_size, False, aug, cfg.bs) if val_rows else None
    select_loader = val_loader if cfg.resolved_select_on() == "val" else test_loader
    is_inc = name == "inception_v3"
    cw = class_weights_from(train_rows, device) if cfg.class_weighted else None

    model, history = train_model(
        model, train_loader, device, cfg.epochs, is_inc, class_weight=cw,
        val_loader=select_loader, patience=cfg.patience, lr_schedule=cfg.lr_schedule,
        lr=cfg.lr, ecoc_head=ecoc_head, name=name, backbone_lr_mult=cfg.backbone_lr_mult,
        label_smoothing=0.0 if cfg.ecoc_head else cfg.label_smoothing,
        monitor=cfg.monitor, freeze_bn=cfg.freeze_bn, amp=cfg.amp)

    m = eval_model(model, test_loader, device, is_inc, ecoc_head=ecoc_head, tta=cfg.tta, amp=cfg.amp)
    if val_loader is not None:
        _, m["val_acc"], m["val_macro_f1"] = quick_eval(model, val_loader, device, is_inc,
                                                        ecoc_head=ecoc_head, amp=cfg.amp)
        logger.info(f"  val (selection split) acc={m['val_acc']:.3f} macroF1={m['val_macro_f1']:.3f}")
    log_eval_summary(m, history)

    run_dir = os.path.join(cfg.runs_dir, name, tag)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"  run folder: {run_dir}")
    if not cfg.no_report:
        plot_model_report(name, history, m, out_dir=cfg.report_dir or run_dir)

    if not cfg.no_ckpt:
        ckpt_dir = cfg.ckpt_dir or run_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        torch.save({
            "backbone": name, "finetune_head": cfg.finetune_head,
            "unfreeze": cfg.resolved_unfreeze(), "ecoc_head": cfg.ecoc_head,
            "head_hidden": cfg.head_hidden, "head_dropout": cfg.head_dropout,
            "ecoc_codebook": cfg.ecoc_codebook, "input_size": input_size,
            "materials": MATERIALS, "material_recyclable": MATERIAL_RECYCLABLE,
            "protocol": cfg.protocol, "aug_mode": aug, "class_weighted": cfg.class_weighted,
            "lr": cfg.lr, "lr_schedule": cfg.lr_schedule, "tag": tag,
            "material_acc": m["material_acc"], "recyclability_acc": m["recyclability_acc"],
            "non_recyclable_recall": m["non_recyclable_recall"],
            "best_epoch": history.get("best_epoch"),
            "model_state_dict": model.state_dict(),
        }, ckpt_path)
        logger.info(f"  saved checkpoint {ckpt_path}  (best epoch: "
                    f"{history.get('best_epoch', cfg.epochs)}, material_acc={m['material_acc']:.3f})")

    m_json = {k: v for k, v in m.items() if k not in ("y_true", "y_probs", "samples")}
    m_json["history"] = history
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"backbone": name, "tag": tag, "config": cfg.to_dict(),
                   # flat keys kept for older readers
                   "protocol": cfg.protocol, "epochs": cfg.epochs, "aug_mode": aug,
                   "finetune_head": cfg.finetune_head, "class_weighted": cfg.class_weighted,
                   "lr": cfg.lr, "lr_schedule": cfg.lr_schedule,
                   "results": m_json}, f, indent=2)
    return m_json


def run(cfg: RunConfig):
    tag = run_tag(cfg)
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
        set_seed(cfg.seed)  # same seed per backbone, independent of run order
        results[name] = run_backbone(name, cfg, tag, device, rows)

    with open(out, "w") as f:
        json.dump({"tag": tag, "config": cfg.to_dict(), "protocol": cfg.protocol,
                   "epochs": cfg.epochs, "aug_mode": cfg.resolved_aug_mode(),
                   "class_weighted": cfg.class_weighted, "results": results}, f, indent=2)
    logger.info(f"\nwrote {out}")
    logger.info(f"wrote {log_file}")


def main(argv=None):
    run(config_from_args(argv))


if __name__ == "__main__":
    main()
