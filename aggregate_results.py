"""
Results aggregation (Phase 4)
=============================
Consumes the results_*.json files written by benchmark_backbones.py and produces
the paper's tables and the domain-gap figure.

    python aggregate_results.py \
        --in_domain results_in_domain.json \
        --cross results_cross_domain.json \
        --joint results_joint.json

Outputs: table_t3_main.md, table_t4_backbones.md, fig_domain_gap.png
Ablation deltas (T5) are produced by passing extra --ablation name=file.json pairs.
"""
import json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.load(open(path)) if path else None


def best_backbone(res):
    return max(res["results"].items(), key=lambda kv: kv[1]["material_acc"])


def t3(in_d, cross, joint, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Table T3 — Main results (best backbone per protocol)\n\n")
        f.write("| protocol | material acc | material macroF1 | binary acc | "
                "binary macroF1 | non_rec recall |\n|---|---|---|---|---|---|\n")
        gap = {}
        for name, r in [("P1 in-domain", in_d), ("P2 cross-domain", cross),
                        ("P3 joint", joint)]:
            if not r:
                continue
            bk, m = best_backbone(r)
            gap[name] = m["material_acc"]
            f.write(f"| {name} ({bk}) | {m['material_acc']:.3f} | "
                    f"{m['material_macro_f1']:.3f} | {m['binary_acc']:.3f} | "
                    f"{m['binary_macro_f1']:.3f} | {m['non_recyclable_recall']:.3f} |\n")
        if "P1 in-domain" in gap and "P2 cross-domain" in gap:
            f.write(f"\n**Domain gap** = acc(P1) - acc(P2) = "
                    f"{gap['P1 in-domain'] - gap['P2 cross-domain']:.3f}\n")
        if "P3 joint" in gap and "P2 cross-domain" in gap:
            f.write(f"**Recovery** = acc(P3) - acc(P2) = "
                    f"{gap['P3 joint'] - gap['P2 cross-domain']:.3f}\n")
    print("wrote", out)


def t4(cross, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Table T4 — Five-backbone benchmark on P2 (cross-domain)\n\n")
        f.write("| backbone | material acc | material macroF1 | binary acc | non_rec recall |\n")
        f.write("|---|---|---|---|---|\n")
        for bk, m in sorted(cross["results"].items(),
                            key=lambda kv: -kv[1]["material_acc"]):
            f.write(f"| {bk} | {m['material_acc']:.3f} | {m['material_macro_f1']:.3f} | "
                    f"{m['binary_acc']:.3f} | {m['non_recyclable_recall']:.3f} |\n")
    print("wrote", out)


def fig_gap(in_d, cross, joint, out):
    labels, accs = [], []
    for name, r in [("in-domain", in_d), ("cross-domain", cross), ("joint", joint)]:
        if r:
            labels.append(name); accs.append(best_backbone(r)[1]["material_acc"])
    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, accs, color=["#4c72b0", "#c44e52", "#55a868"])
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}", ha="center")
    plt.ylabel("material accuracy"); plt.ylim(0, 1)
    plt.title("Domain gap and recovery (best backbone)")
    plt.tight_layout(); plt.savefig(out, dpi=130)
    print("wrote", out)


def t5(cross, ablations, out):
    base = best_backbone(cross)[1]["material_acc"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Table T5 — Ablations (delta material acc vs P2 cross-domain)\n\n")
        f.write(f"Baseline P2 best-backbone material acc = {base:.3f}\n\n")
        f.write("| ablation | material acc | delta vs P2 | non_rec recall |\n|---|---|---|---|\n")
        for name, path in ablations:
            r = load(path)
            m = best_backbone(r)[1] if "results" in r else r
            f.write(f"| {name} | {m['material_acc']:.3f} | "
                    f"{m['material_acc'] - base:+.3f} | {m['non_recyclable_recall']:.3f} |\n")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_domain")
    ap.add_argument("--cross", required=True)
    ap.add_argument("--joint")
    ap.add_argument("--ablation", action="append", default=[],
                    help="name=path.json (repeatable)")
    args = ap.parse_args()
    in_d, cross, joint = load(args.in_domain), load(args.cross), load(args.joint)
    t3(in_d, cross, joint, "table_t3_main.md")
    t4(cross, "table_t4_backbones.md")
    fig_gap(in_d, cross, joint, "fig_domain_gap.png")
    if args.ablation:
        abls = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args.ablation]
        t5(cross, abls, "table_t5_ablations.md")
