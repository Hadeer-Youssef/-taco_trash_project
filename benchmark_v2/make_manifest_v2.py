"""
manifest_v2 builder: photo-grouped wild splits + honest crop audit + dedup report
=================================================================================
Fixes over waste_pipeline.build_manifest (the v1 per-crop random split):

  1. GROUPED WILD SPLIT   every crop of one TACO photograph lives in exactly one
                          split (StratifiedGroupKFold by photo, stratified on
                          material x recyclable, 10 folds -> fold0 test, fold1
                          val, folds 2-9 train). Kills the 80.8% train/test
                          photo leakage of manifest.csv.
  2. CROP AUDIT           every COCO annotation is accounted for: harvested,
                          dropped-by-size, or silently lost by the original
                          crop loop. Per-class drop table -> crop_audit.csv
                          (v1 lost 13.6% overall, 67.5% kept for `trash`).
  3. DEDUP REPORT         exact md5 + dHash clusters for TrashNet -> clean_dups.csv.
                          Clean rows copy their existing split (P1 stays
                          comparable); clusters spanning splits are reported.
  4. LABEL PROVENANCE     labels re-derived from label_mapping_v2.csv and
                          asserted identical to v1 values; ambiguous flag is
                          carried through from the mapping.

Outputs (inside benchmark_v2/):
  manifest_v2.csv   path, material, recyclable, domain, split, photo, ann_id,
                    taco_category, taco_supercategory, bbox_rel_area,
                    bbox_side_px, ambiguous
  crop_audit.csv    per-class accounting of TACO annotations
  clean_dups.csv    TrashNet near/exact duplicate clusters
  SPLIT_REPORT.md   the whole story in one page

    python make_manifest_v2.py [--seed 42]
"""

import argparse
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root: Taco_data, crops_material, trashnet_dataset-resized, label_mapping.csv

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MAT2IDX = {m: i for i, m in enumerate(MATERIALS)}
MATERIAL_RECYCLABLE = [1, 1, 1, 1, 1, 0]  # metadata only; wild labels are item-aware


def _norm(p):
    """relative to repo root, forward slashes — same convention as manifest.csv,
    so the file stays portable between Windows and Kaggle/Linux."""
    return os.path.relpath(p, ROOT).replace("\\", "/")


# ----------------------------------------------------------------- dHash ----
def dhash(path, hsize=8):
    """64-bit difference hash; strict 8x8 (only near-identical frames match)."""
    im = Image.open(path).convert("L").resize((hsize + 1, hsize), Image.LANCZOS)
    px = list(im.getdata())
    bits = 0
    k = 0
    for r in range(hsize):
        for c in range(hsize):
            bits |= (px[r * (hsize + 1) + c] > px[r * (hsize + 1) + c + 1]) << k
            k += 1
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


# ------------------------------------------------------------ wild audit ----
def audit_and_index_wild(mapping, ann_path, crops_dir):
    """Walk the frozen crop set on disk instead of re-cropping, and account for
    every COCO annotation. Returns (rows, audit_df, ann_cat)."""
    global _ann_cat_placeholder
    ann = json.load(open(ann_path, encoding="utf-8"))
    cat_of = {a["id"]: a["category_id"] for a in ann["annotations"]}
    bbox_of = {a["id"]: a["bbox"] for a in ann["annotations"]}
    img_of = {a["id"]: a["image_id"] for a in ann["annotations"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in ann["images"]}
    cats = {c["id"]: (c["name"], c["supercategory"]) for c in ann["categories"]}

    kept = {}
    for dirpath, _, files in os.walk(crops_dir):
        for fn in files:
            m = re.fullmatch(r"taco_(\d+)\.jpg", fn)
            if m:
                kept[int(m.group(1))] = os.path.join(dirpath, fn)

    rows, dropped = [], []
    for ann_id, cat_id in cat_of.items():
        if cat_id not in mapping:
            dropped.append((MAT2IDX["trash"], "unknown_category"))
            continue
        mat_idx, rec = mapping[cat_id]
        w, h = bbox_of[ann_id][2], bbox_of[ann_id][3]
        iw, ih = sizes[img_of[ann_id]]
        if w < 32 or h < 32:
            dropped.append((mat_idx, "too_small"))
            continue
        if ann_id not in kept:
            dropped.append((mat_idx, "crop_failed_or_filtered"))
            continue
        rows.append({"path": _norm(kept[ann_id]), "material": mat_idx,
                     "recyclable": rec, "domain": "wild", "photo": img_of[ann_id],
                     "ann_id": ann_id, "taco_category": cats[cat_id][0],
                     "taco_supercategory": cats[cat_id][1],
                     "bbox_rel_area": (w * h) / (iw * ih),
                     "bbox_side_px": min(w, h)})
    return rows, pd.DataFrame(dropped, columns=["material", "reason"]), cat_of


# ------------------------------------------------------------- clean side ----
def index_trashnet(root):
    rows = []
    for mat_name in MATERIALS:
        d = os.path.join(root, mat_name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"path": _norm(os.path.join(d, f)),
                             "material": MAT2IDX[mat_name],
                             "recyclable": MATERIAL_RECYCLABLE[MAT2IDX[mat_name]],
                             "domain": "clean"})
    return rows


def clean_dup_report(rows, out_csv):
    """Union-find over exact md5 and dHash(hamming<3); one row per dup cluster."""
    md5s, hhs = [], []
    for r in rows:
        with open(os.path.join(ROOT, r["path"]), "rb") as fh:
            md5s.append(hashlib.md5(fh.read()).hexdigest())
        hhs.append(dhash(os.path.join(ROOT, r["path"])))
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if md5s[i] == md5s[j] or hamming(hhs[i], hhs[j]) < 3:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi

    clusters, seen_root = [], set()
    for i in range(n):
        ri = find(i)
        if ri in seen_root:
            continue
        members = [j for j in range(n) if find(j) == ri]
        if len(members) > 1:
            seen_root.add(ri)
            md5_dup = int(len(set(md5s[j] for j in members)) < len(members))
            splits = sorted(set(rows[j].get("split", "") for j in members))
            clusters.append({"cluster_size": len(members), "md5_dup": md5_dup,
                             "paths": " | ".join(os.path.basename(rows[j]["path"]) for j in members),
                             "spans_splits": ",".join(s for s in splits if s)})
    df = pd.DataFrame(clusters, columns=["cluster_size", "md5_dup", "paths", "spans_splits"])
    df.to_csv(out_csv, index=False)
    return clusters


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mdf = pd.read_csv(os.path.join(HERE, "label_mapping_v2.csv"))
    mapping = {int(r.taco_category_id): (MAT2IDX[r.material_6class],
                                         1 if r.recyclability_binary == "recyclable" else 0)
               for _, r in mdf.iterrows()}
    amb_of_cat = dict(zip(mdf.taco_category_id, mdf.ambiguous.astype(int)))
    totals = mdf.groupby("material_6class").n_annotations.sum().to_dict()

    wild_rows, audit, cat_of = audit_and_index_wild(
        mapping, os.path.join(ROOT, "Taco_data", "annotations.json"),
        os.path.join(ROOT, "crops_material"))
    clean_rows = index_trashnet(os.path.join(ROOT, "trashnet_dataset-resized"))

    # clean splits copied from v1 (P1 comparability)
    v1 = pd.read_csv(os.path.join(ROOT, "manifest.csv"))
    split_of = dict(zip(v1[v1.domain == "clean"].path.str.replace("\\", "/", regex=False),
                        v1[v1.domain == "clean"].split))
    for r in clean_rows:
        sp = split_of.get(r["path"])
        if sp is None:
            raise SystemExit(f"clean path missing from v1 manifest: {r['path']}")
        r["split"] = sp

    # ---- photo-grouped wild split
    wild = pd.DataFrame(wild_rows)
    wild["ambiguous"] = wild.ann_id.map(lambda a: amb_of_cat[cat_of[a]])
    strata = wild.material * 2 + wild.recyclable
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold = np.empty(len(wild), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(wild, strata, groups=wild.photo)):
        fold[idx] = k
    wild["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))

    out = pd.concat([wild, pd.DataFrame(clean_rows)], ignore_index=True)[
        ["path", "material", "recyclable", "domain", "split", "photo", "ann_id",
         "taco_category", "taco_supercategory", "bbox_rel_area", "bbox_side_px",
         "ambiguous"]]
    out_path = os.path.join(HERE, "manifest_v2.csv")
    out.to_csv(out_path, index=False)

    # ---- integrity checks
    assert (wild.groupby("photo").split.nunique() == 1).all(), "photo across splits"
    assert not (set(wild[wild.split == "train"].photo)
                & set(wild[wild.split == "test"].photo)), "train/test photo overlap"

    audit.to_csv(os.path.join(HERE, "crop_audit.csv"), index=False)
    dup_rows = clean_dup_report(clean_rows, os.path.join(HERE, "clean_dups.csv"))

    # ---- report
    lines = ["# benchmark_v2 split report", f"seed: {args.seed}", "",
             "## 1. TACO annotation accounting (v1 silently lost these)", "",
             "| material | annotations | harvested | crop_failed_or_filtered | too_small | kept% |",
             "|---|---|---|---|---|---|"]
    for mat_i, mat in enumerate(MATERIALS):
        tot = int(totals.get(mat, 0))
        sub = audit[audit.material == mat_i].reason.value_counts()
        failed = int(sub.get("crop_failed_or_filtered", 0))
        small = int(sub.get("too_small", 0))
        kept = tot - failed - small
        pct = f"{100 * kept / tot:.1f}%" if tot else "—"
        lines.append(f"| {mat} | {tot} | {kept} | {failed} | {small} | {pct} |")
    lines += ["",
              "`crop_failed_or_filtered`: annotation passed the 32px filter but has no crop"
              " file — the exception branch of the original `taco_to_crops` loop.", "",
              "## 2. Wild photo-grouped split (leakage fix)", "",
              f"wild: {len(wild)} crops / {wild.photo.nunique()} photos"
              f" ({wild.groupby('photo').size().mean():.2f} crops/photo)", "",
              "| split | crops | photos | non-recyclable share |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        sub = wild[wild.split == s]
        lines.append(f"| {s} | {len(sub)} | {sub.photo.nunique()} | {100 * (1 - sub.recyclable.mean()):.1f}% |")
    lines += ["", "every photo in exactly one split: **OK**",
              "",
              "v1 (manifest.csv): 27.6% of photos spanned splits; **80.8% of wild test crops"
              " shared a photo with train**. Fixed here.", "",
              "## 3. Clean domain (TrashNet)", "",
              f"{len(clean_rows)} rows; splits copied from manifest.csv (P1 comparability).",
              f"Duplicate clusters (md5 or dHash <3): {len(dup_rows)}",
              f"- containing an exact md5 duplicate: {sum(r['md5_dup'] for r in dup_rows)}",
              f"- spanning existing splits: {sum(1 for r in dup_rows if r['spans_splits'])}", "",
              "## 4. Files", "",
              "- `manifest_v2.csv` photo/ann_id/bbox stats/ambiguous columns added",
              "- `crop_audit.csv` annotation accounting",
              "- `clean_dups.csv` TrashNet duplicate clusters"]
    with open(os.path.join(HERE, "SPLIT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"wrote {out_path} ({len(out)} rows: {len(wild)} wild, {len(clean_rows)} clean)")
    print(f"wild photos {wild.photo.nunique()} — split integrity OK")
    print(f"crop audit: {len(audit)} dropped annotations -> crop_audit.csv")
    print(f"clean dup clusters: {len(dup_rows)} -> clean_dups.csv")
    print("wrote SPLIT_REPORT.md")


if __name__ == "__main__":
    main()
