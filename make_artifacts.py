"""
Dataset artifacts (Phase 1 / Phase 4 helpers)
=============================================
- sanity crop grid (one row per material, clean vs wild) -> fig_crops_grid.png
- Table T1 dataset composition (material x domain, recyclable split) -> table_t1.md
Driven entirely off the frozen manifest.csv.
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def sanity_grid(df, out_png, per=4, seed=42):
    rng = np.random.RandomState(seed)
    domains = ["clean", "wild"]
    ncols = per * len(domains)
    fig, axes = plt.subplots(len(MATERIALS), ncols, figsize=(ncols * 1.4, len(MATERIALS) * 1.4))
    for ri, mi in enumerate(range(len(MATERIALS))):
        col = 0
        for dom in domains:
            sub = df[(df.material == mi) & (df.domain == dom)]
            picks = sub.sample(min(per, len(sub)), random_state=int(rng.randint(1e6))) if len(sub) else sub
            paths = list(picks["path"])
            for j in range(per):
                ax = axes[ri, col]
                ax.set_xticks([]); ax.set_yticks([])
                if j < len(paths) and os.path.exists(paths[j]):
                    ax.imshow(Image.open(paths[j]).convert("RGB"))
                if ri == 0:
                    ax.set_title(f"{dom}" if j == 0 else "", fontsize=8)
                if col == 0:
                    ax.set_ylabel(MATERIALS[mi], fontsize=9)
                col += 1
    fig.suptitle("Sanity crop grid: material rows x (clean | wild) domains", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


def table_t1(df, out_md):
    df = df.copy()
    df["mat"] = df["material"].map(lambda i: MATERIALS[int(i)])
    rows = []
    for dom in ["clean", "wild"]:
        for mi, m in enumerate(MATERIALS):
            sub = df[(df.domain == dom) & (df.material == mi)]
            rec = int((sub.recyclable == 1).sum()); non = int((sub.recyclable == 0).sum())
            rows.append((dom, m, len(sub), rec, non))
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table T1 — Dataset composition (material x domain)\n\n")
        f.write("| domain | material | total | recyclable | non_recyclable |\n")
        f.write("|---|---|---|---|---|\n")
        for dom, m, t, rec, non in rows:
            f.write(f"| {dom} | {m} | {t} | {rec} | {non} |\n")
        f.write("\n**Totals**\n\n")
        for dom in ["clean", "wild"]:
            s = df[df.domain == dom]
            f.write(f"- {dom}: {len(s)} images "
                    f"({int((s.recyclable==1).sum())} recyclable / "
                    f"{int((s.recyclable==0).sum())} non_recyclable)\n")
    print("wrote", out_md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    sanity_grid(df, "fig_crops_grid.png")
    table_t1(df, "table_t1.md")
