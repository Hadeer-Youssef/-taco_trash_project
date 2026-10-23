"""Cross-domain matrix runs: train on clean, wild or both; test on BOTH test sets.

One run fills one row of the generalization matrix

                   clean test    wild test
    train clean        x             x
    train wild         x             x
    train both         x             x

with the dual head (6 materials + 2 recyclability, see engine_dual.py). Model
selection uses the validation split of the TRAINING domain(s) by default, so no
run ever selects on the domain it is tested out-of-domain on.

Meant for manifest_grouped.csv (TACO split by photo, see make_grouped_manifest.py).
Same flags as cli.py plus the ones below; --protocol is ignored.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from .cli import build_parser
from .config import RunConfig, run_tag
from .constants import MATERIALS, N_CLASSES
from .data import load_manifest, make_loader
from .engine_dual import (N_OUTPUTS, REC_NAMES, class_weights_by, train_dual, eval_dual,
                          quick_eval_dual)
from .models import build_backbone, count_trainable
from .reporting_dual import plot_dual_report, log_dual_summary
from .utils import logger, setup_logging, set_seed

TEST_SETS = ("clean_test", "wild_test")


def parse_args(argv=None):
    ap = build_parser()
    ap.set_defaults(protocol="matrix", manifest="manifest_grouped.csv")
    g = ap.add_argument_group("matrix")
    g.add_argument("--train_on", choices=["clean", "wild", "both"], required=True)
    g.add_argument("--val_on", choices=["train", "clean", "wild"], default="train",
                   help="selection split: 'train' = val split(s) of the training domain(s) "
                        "(both -> clean val + wild val); clean / wild force one of them")
    g.add_argument("--wild_fraction", type=float, default=1.0,
                   help="keep this fraction of the wild TRAINING photos (all crops of a kept "
                        "photo); val/test are never subsampled. The subset depends on --seed")
    g.add_argument("--rec_weight", type=float, default=1.0)
    g.add_argument("--no_rec_class_weights", action="store_true")
    kw = vars(ap.parse_args(argv))
    extra = {k: kw.pop(k) for k in ("train_on", "val_on", "wild_fraction", "rec_weight",
                                    "no_rec_class_weights")}
    kw.pop("no_domain_randomize")
    cfg = RunConfig(**kw)
    if cfg.ecoc_head:
        raise SystemExit("--ecoc_head is not supported with the dual head")
    if not 0 < extra["wild_fraction"] <= 1:
        raise SystemExit("--wild_fraction must be in (0, 1]")
    if extra["wild_fraction"] < 1 and extra["train_on"] == "clean":
        raise SystemExit("--wild_fraction needs --train_on wild or both")
    return cfg, extra


def matrix_tag(cfg, ex):
    rest = run_tag(cfg)[len(cfg.protocol) + 1:]          # drop the "matrix_" prefix
    tag = f"train-{ex['train_on']}"
    if ex["wild_fraction"] < 1:
        tag += f"_wf{ex['wild_fraction']:g}"
    if ex["val_on"] != "train":
        tag += f"_val-{ex['val_on']}"
    tag += "_" + rest + "_dual"
    if ex["rec_weight"] != 1.0:
        tag += f"-rw{ex['rec_weight']:g}"
    if ex["no_rec_class_weights"]:
        tag += "-norcw"
    return tag


def _subsample_photos(rows, fraction, seed):
    photos = sorted({r["photo"] for r in rows})
    rng = np.random.RandomState(seed)
    keep = set(rng.choice(photos, size=max(1, round(fraction * len(photos))), replace=False))
    return [r for r in rows if r["photo"] in keep], len(keep), len(photos)


def select_matrix_rows(df, ex, seed):
    rec = lambda d: d.to_dict("records")
    clean, wild = df[df.domain == "clean"], df[df.domain == "wild"]
    split = lambda d, s: rec(d[d.split == s])
    wild_train, info = split(wild, "train"), {}
    if ex["wild_fraction"] < 1:
        wild_train, kept, total = _subsample_photos(wild_train, ex["wild_fraction"], seed)
        info = {"wild_train_photos": kept, "wild_train_photos_total": total}
    train = {"clean": split(clean, "train"), "wild": wild_train,
             "both": split(clean, "train") + wild_train}[ex["train_on"]]
    val_on = ex["train_on"] if ex["val_on"] == "train" else ex["val_on"]
    val = {"clean": split(clean, "val"), "wild": split(wild, "val"),
           "both": split(clean, "val") + split(wild, "val")}[val_on]
    tests = {"clean_test": split(clean, "test"), "wild_test": split(wild, "test")}
    info.update({"n_train": len(train), "n_train_wild": len(wild_train) if ex["train_on"] != "clean" else 0,
                 "n_val": len(val), "val_on": val_on})
    return train, val, tests, info


def run(cfg, ex):
    tag = matrix_tag(cfg, ex)
    log_file = cfg.log_file or os.path.join(cfg.runs_dir, "_summaries", f"{tag}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    setup_logging(log_file)
    logger.info(f"run tag: {tag}")
    df = load_manifest(cfg.manifest)
    if "photo" not in df.columns:
        logger.info(f"  WARNING: {cfg.manifest} has no 'photo' column -- it is not the photo-grouped "
                    f"manifest; TACO crops of one photo may sit in different splits")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows, tests, info = select_matrix_rows(df, ex, cfg.seed)
    aug = cfg.resolved_aug_mode()
    logger.info(f"train_on={ex['train_on']}  wild_fraction={ex['wild_fraction']:g}  "
                f"selection on {info['val_on']} val  device={device}  aug_mode={aug}  seed={cfg.seed}")
    logger.info(f"  train {info['n_train']} (wild {info['n_train_wild']})"
                + (f"  wild photos kept {info['wild_train_photos']}/{info['wild_train_photos_total']}"
                   if "wild_train_photos" in info else "")
                + f"  val {info['n_val']}  clean_test {len(tests['clean_test'])}  "
                  f"wild_test {len(tests['wild_test'])}")

    summary = {}
    for name in cfg.backbones:
        set_seed(cfg.seed)
        logger.info(f"\n=== {name} (dual head) ===")
        model, input_size = build_backbone(name, n_classes=N_OUTPUTS, finetune_head=cfg.finetune_head,
                                           unfreeze=cfg.resolved_unfreeze(),
                                           head_hidden=cfg.head_hidden, head_dropout=cfg.head_dropout)
        logger.info(f"  unfreeze={cfg.resolved_unfreeze()}  trainable params: {count_trainable(model):,}")
        train_loader = make_loader(train_rows, input_size, True, aug, cfg.bs, cfg.workers)
        val_loader = make_loader(val_rows, input_size, False, aug, cfg.bs, cfg.workers)
        w_mat = w_rec = None
        if cfg.class_weighted:
            w_mat = class_weights_by(train_rows, "material", N_CLASSES, device, MATERIALS)
            if not ex["no_rec_class_weights"]:
                w_rec = class_weights_by(train_rows, "recyclable", 2, device, REC_NAMES)
        is_inc = name == "inception_v3"

        model, history = train_dual(
            model, train_loader, device, cfg.epochs, is_inc, w_mat, w_rec, ex["rec_weight"],
            val_loader=val_loader, patience=cfg.patience, lr_schedule=cfg.lr_schedule, lr=cfg.lr,
            name=name, backbone_lr_mult=cfg.backbone_lr_mult, label_smoothing=cfg.label_smoothing,
            monitor=cfg.monitor, freeze_bn=cfg.freeze_bn, amp=cfg.amp)

        run_dir = os.path.join(cfg.runs_dir, name, tag)
        os.makedirs(run_dir, exist_ok=True)
        v = quick_eval_dual(model, val_loader, device, is_inc, amp=cfg.amp)
        logger.info(f"  val ({info['val_on']}) material acc={v['mat_acc']:.3f} "
                    f"recyclability acc={v['rec_acc']:.3f}")
        results = {"val": v, "history": history}
        for test_name in TEST_SETS:
            loader = make_loader(tests[test_name], input_size, False, aug, cfg.bs, cfg.workers)
            m = eval_dual(model, loader, device, is_inc, tta=cfg.tta, amp=cfg.amp)
            logger.info(f"\n  --- {test_name} ({len(tests[test_name])} images) ---")
            log_dual_summary(m, history)
            if not cfg.no_report:
                plot_dual_report(f"{name}_{test_name}", history, m, out_dir=cfg.report_dir or run_dir)
            results[test_name] = {k: val for k, val in m.items()
                                  if k not in ("y_true", "y_rec_true", "y_probs", "nonrec_probs",
                                               "samples")}

        if not cfg.no_ckpt:
            ckpt_path = os.path.join(cfg.ckpt_dir or run_dir, "checkpoint.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"backbone": name, "dual_head": True, "n_outputs": N_OUTPUTS,
                        "head_hidden": cfg.head_hidden, "head_dropout": cfg.head_dropout,
                        "unfreeze": cfg.resolved_unfreeze(), "finetune_head": cfg.finetune_head,
                        "input_size": input_size, "materials": MATERIALS, "rec_names": REC_NAMES,
                        "tag": tag, "matrix": {**ex, **info}, "best_epoch": history.get("best_epoch"),
                        "model_state_dict": model.state_dict()}, ckpt_path)
            logger.info(f"  saved checkpoint {ckpt_path}  (best epoch: "
                        f"{history.get('best_epoch', cfg.epochs)})")

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump({"backbone": name, "tag": tag, "config": cfg.to_dict(),
                       "matrix": {**ex, **info}, "manifest": cfg.manifest,
                       "results": results}, f, indent=2)
        summary[name] = {t: {k: results[t][k] for k in ("material_acc", "material_macro_f1",
                                                        "recyclability_acc", "recyclability_auc_nonrec")}
                         for t in TEST_SETS}

    out = cfg.out or os.path.join(cfg.runs_dir, "_summaries", f"{tag}.json")
    with open(out, "w") as f:
        json.dump({"tag": tag, "matrix": {**ex, **info}, "config": cfg.to_dict(),
                   "results": summary}, f, indent=2)
    logger.info(f"\nwrote {out}\nwrote {log_file}")


def main(argv=None):
    run(*parse_args(argv))


if __name__ == "__main__":
    main()
