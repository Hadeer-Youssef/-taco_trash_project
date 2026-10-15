"""
Ensemble + error analysis over saved checkpoints
================================================
Loads every checkpoint under runs/ whose run tag contains --match, computes
each model's class probabilities once on the val and test splits of the
protocol, and reports

  1. every single model,
  2. one seed-ensemble per backbone (mean of its checkpoints' probabilities),
  3. the ensemble of ALL selected checkpoints,

on val and test, so ensembles are chosen/compared on val and test is only the
final number. It also writes the test images that are misclassified, and
which of them EVERY single model gets wrong (the "hard core": if it is large
the ceiling comes from ambiguous labels, not from the model).

    python ensemble_eval.py --protocol in_domain --match ls0.1 --match tta
    python ensemble_eval.py --protocol in_domain --match two_stages --backbones resnet50 densenet121

Needs the checkpoints on this machine (run it on Kaggle where they were
trained) and must be started from the repo root (manifest paths are relative).
"""

import argparse
import glob
import itertools
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from run_predict import load_checkpoint
from waste_bench.constants import MATERIALS, MATERIAL_RECYCLABLE
from waste_bench.data import load_manifest, select_rows, make_loader
from waste_bench.engine import forward_logits, predict_from_logits


@torch.no_grad()
def collect_probs(model, ecoc_head, loader, device, is_inception, tta):
    """(N, 6) class probabilities of one model, in the loader's (fixed) order."""
    model.to(device).eval()
    out = []
    for x, _, _ in loader:
        x = x.to(device)
        logits, _ = forward_logits(model, x, is_inception, False)
        _, probs = predict_from_logits(logits, ecoc_head)
        if tta:
            logits_f, _ = forward_logits(model, torch.flip(x, dims=[3]), is_inception, False)
            _, probs_f = predict_from_logits(logits_f, ecoc_head)
            probs = (probs + probs_f) / 2
        out.append(probs.float().cpu())
    return torch.cat(out).numpy()


def metrics(probs, y, rec):
    pred = probs.argmax(1)
    rec_pred = np.array(MATERIAL_RECYCLABLE)[pred]
    return {"acc": accuracy_score(y, pred),
            "f1": f1_score(y, pred, average="macro", zero_division=0),
            "nr_rec": recall_score(rec, rec_pred, pos_label=0, zero_division=0)}


def find_checkpoints(runs_dir, protocol, matches, backbones):
    found = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "*", "checkpoint.pt"))):
        tag = os.path.basename(os.path.dirname(path))
        backbone = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if not tag.startswith(protocol + "_"):
            continue
        if any(m not in tag for m in matches):
            continue
        if backbones and backbone not in backbones:
            continue
        found.append((backbone, tag, path))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--protocol", default="in_domain", choices=["in_domain", "joint"],
                    help="needs a protocol with a val split")
    ap.add_argument("--match", action="append", default=[],
                    help="substring the run tag must contain (repeatable, all must match)")
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--tta", action="store_true", help="flip-average each model's predictions")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_dir", default=".", help="where ensemble_errors.csv / ensemble_table.csv go")
    args = ap.parse_args()

    ckpts = find_checkpoints(args.runs_dir, args.protocol, args.match, args.backbones)
    if not ckpts:
        raise SystemExit(f"no checkpoint under {args.runs_dir}/ for protocol={args.protocol} "
                         f"matching {args.match}")
    print(f"{len(ckpts)} checkpoints:")
    for b, t, _ in ckpts:
        print(f"  {b:13s} {t}")

    _, val_rows, test_rows = select_rows(load_manifest(args.manifest), args.protocol)
    if not val_rows:
        raise SystemExit(f"protocol {args.protocol} has no val split")
    splits = {"val": val_rows, "test": test_rows}
    y = {s: np.array([r["material"] for r in rows]) for s, rows in splits.items()}
    rec = {s: np.array([r["recyclable"] for r in rows]) for s, rows in splits.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs = {}  # (backbone, tag) -> {"val": array, "test": array}
    for backbone, tag, path in ckpts:
        model, ckpt, ecoc_head = load_checkpoint(path, device)
        per = {}
        for s, rows in splits.items():
            loader = make_loader(rows, ckpt["input_size"], False, "off", args.bs, args.workers)
            per[s] = collect_probs(model, ecoc_head, loader, device,
                                   backbone == "inception_v3", args.tta)
        probs[(backbone, tag)] = per
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  evaluated {backbone}/{tag}")

    def row(name, members):
        res = {"ensemble": name, "n_models": len(members)}
        for s in ("val", "test"):
            avg = np.mean([probs[m][s] for m in members], axis=0)
            m = metrics(avg, y[s], rec[s])
            res.update({f"{s}_acc": m["acc"], f"{s}_f1": m["f1"]})
            if s == "test":
                res["test_nr_rec"] = m["nr_rec"]
        return res

    rows = []
    by_backbone = defaultdict(list)
    for key in probs:
        by_backbone[key[0]].append(key)
        rows.append(row(f"single: {key[0]} | {key[1]}", [key]))
    for b, members in by_backbone.items():
        if len(members) > 1:
            rows.append(row(f"seeds of {b}", members))
    if len(by_backbone) > 1:
        # plain mean over every selected checkpoint, so a backbone with more
        # seeds gets proportionally more weight; select the same number of
        # seeds per backbone (--match / --backbones) if that matters
        rows.append(row("ALL checkpoints", list(probs)))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print("\n" + table.round(3).to_string(index=False))
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "ensemble_table.csv"), index=False)

    # ---- error analysis on test ------------------------------------------
    ens = np.mean([probs[m]["test"] for m in probs], axis=0)
    ens_pred = ens.argmax(1)
    single_wrong = np.stack([probs[m]["test"].argmax(1) != y["test"] for m in probs])
    every_wrong = single_wrong.all(axis=0)
    ens_wrong = ens_pred != y["test"]
    print(f"\ntest images: {len(y['test'])}")
    print(f"  wrong in the ALL ensemble             : {int(ens_wrong.sum())}")
    print(f"  wrong in EVERY single model           : {int(every_wrong.sum())}  <- the hard core")
    print(f"  wrong in at least one single model    : {int(single_wrong.any(axis=0).sum())}")

    err = pd.DataFrame({
        "path": [r["path"] for r in splits["test"]],
        "true": [MATERIALS[i] for i in y["test"]],
        "ensemble_pred": [MATERIALS[i] for i in ens_pred],
        "n_models_wrong": single_wrong.sum(axis=0),
        "n_models": len(probs),
        "true_recyclable": rec["test"],
    })
    err = err[(err.n_models_wrong > 0) | ens_wrong]
    err = err.sort_values(["n_models_wrong", "true"], ascending=[False, True])
    err.to_csv(os.path.join(args.out_dir, "ensemble_errors.csv"), index=False)
    print(f"wrote {os.path.join(args.out_dir, 'ensemble_errors.csv')} and ensemble_table.csv")
    conf = pd.crosstab(err[err.n_models_wrong == len(probs)].true,
                       err[err.n_models_wrong == len(probs)].ensemble_pred)
    if not conf.empty:
        print("\nhard core (all models wrong): true -> ensemble prediction")
        print(conf.to_string())


if __name__ == "__main__":
    main()
