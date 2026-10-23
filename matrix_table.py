"""
Generalization-matrix table: mean ± std over seeds of every run_matrix.py run.

    python matrix_table.py                 # all backbones under runs/
    python matrix_table.py --csv matrix.csv
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = [("material_acc", "mat_acc"), ("material_macro_f1", "mat_F1"),
           ("recyclability_acc", "rec_acc"), ("recyclability_auc_nonrec", "rec_AUC")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    for f in glob.glob(os.path.join(args.runs_dir, "*", "train-*", "results.json")):
        d = json.load(open(f))
        group = re.sub(r"_s\d+(?=_dual)", "", d["tag"])
        groups[(d["backbone"], group)].append(d)

    rows = []
    for (backbone, group), runs in sorted(groups.items()):
        mx = runs[0]["matrix"]
        for test in ("clean_test", "wild_test"):
            vals = {short: np.array([r["results"][test][key] for r in runs
                                     if r["results"][test][key] is not None], dtype=float)
                    for key, short in METRICS}
            rows.append({"backbone": backbone, "train_on": mx["train_on"],
                         "wild_fraction": mx["wild_fraction"], "test": test, "n_seeds": len(runs),
                         "group": group,
                         **{k: (v.mean(), v.std()) if len(v) else (float("nan"), float("nan"))
                            for k, v in vals.items()}})

    hdr = f"{'backbone':14s} {'train_on':8s} {'wf':>5s} {'test':10s} {'n':>2s} " + \
          " ".join(f"{s:>13s}" for _, s in METRICS)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['backbone']:14s} {r['train_on']:8s} {r['wild_fraction']:5g} {r['test']:10s} "
              f"{r['n_seeds']:2d} " + " ".join(f"{r[s][0]:.3f}±{r[s][1]:.3f}" for _, s in METRICS))
    print("\n(groups whose tags differ in other options, e.g. -rw or _val-, are listed separately; "
          "full tag in --csv)")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["backbone", "train_on", "wild_fraction", "test", "n_seeds", "group"]
                       + [f"{s}_{x}" for _, s in METRICS for x in ("mean", "std")])
            for r in rows:
                w.writerow([r["backbone"], r["train_on"], r["wild_fraction"], r["test"],
                            r["n_seeds"], r["group"]]
                           + [f"{v:.4f}" for _, s in METRICS for v in r[s]])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
