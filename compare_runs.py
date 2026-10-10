"""
Compare every run under runs/ in one table
==========================================
Reads each runs/<backbone>/<tag>/results.json, groups runs that differ only
by --seed, and prints mean ± std per configuration, sorted best first.

    python compare_runs.py --protocol in_domain
    python compare_runs.py --protocol in_domain --backbone resnet50 --baseline in_domain_aug-mild
    python compare_runs.py --csv comparison.csv

Columns: n = seeds, acc/f1 = test material accuracy / macro-F1, nr_rec =
non-recyclable recall, val_acc = accuracy on the selection split, ep = best
epoch, time = training minutes. With --baseline, d_acc is the accuracy
change vs. that tag (same backbone).
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

SEED_SUFFIX = re.compile(r"_s\d+$")


def load_runs(runs_dir):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "results.json")):
        with open(path) as f:
            d = json.load(f)
        r = d.get("results", {})
        backbone = d.get("backbone") or os.path.basename(os.path.dirname(os.path.dirname(path)))
        tag = d.get("tag") or os.path.basename(os.path.dirname(path))
        cfg = d.get("config", {})
        hist = r.get("history", {})
        rows.append({
            "backbone": backbone,
            "protocol": d.get("protocol") or tag.split("_aug")[0],
            "config": SEED_SUFFIX.sub("", tag),
            "seed": cfg.get("seed", 42),
            "selected_on": cfg.get("select_on_resolved", "test (old run)"),
            "acc": r.get("material_acc"),
            "f1": r.get("material_macro_f1"),
            "nr_rec": r.get("non_recyclable_recall"),
            "val_acc": r.get("val_acc"),
            "ep": hist.get("best_epoch"),
            "time": (hist.get("train_time_sec") or 0) / 60,
        })
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["backbone", "config", "selected_on"])
    out = g.agg(n=("acc", "size"), acc=("acc", "mean"), acc_std=("acc", "std"),
                f1=("f1", "mean"), nr_rec=("nr_rec", "mean"), val_acc=("val_acc", "mean"),
                ep=("ep", "mean"), time=("time", "mean")).reset_index()
    return out.sort_values(["backbone", "acc"], ascending=[True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--baseline", default=None, help="config tag to diff against (without _s<seed>)")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV")
    args = ap.parse_args()

    df = load_runs(args.runs_dir)
    if df.empty:
        raise SystemExit(f"no results.json under {args.runs_dir}/")
    if args.protocol:
        df = df[df.protocol == args.protocol]
    if args.backbone:
        df = df[df.backbone == args.backbone]
    table = summarize(df)

    if args.baseline:
        base = table[table.config == args.baseline].set_index("backbone")["acc"]
        table["d_acc"] = table.apply(lambda r: r.acc - base.get(r.backbone, float("nan")), axis=1)

    fmt = {c: "{:.3f}".format for c in ["acc", "acc_std", "f1", "nr_rec", "val_acc", "d_acc"]
           if c in table}
    fmt.update({"ep": "{:.0f}".format, "time": "{:.1f}".format})
    with pd.option_context("display.width", 200, "display.max_colwidth", 70,
                           "display.max_rows", 500):
        print(table.to_string(index=False, formatters=fmt, na_rep="-"))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
