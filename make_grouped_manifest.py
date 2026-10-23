"""
Photo-grouped manifest: manifest.csv -> manifest_grouped.csv
=============================================================
manifest.csv splits every TACO crop independently, so crops cut from the same
photograph land in different splits (about 80 % of wild val/test crops share a
photo with a training crop: same scene, background and lighting). This script
re-splits the wild domain BY PHOTO, so every crop of one photograph is in exactly
one split. The clean (TrashNet) rows and their splits are copied unchanged, so P1
numbers stay comparable.

Wild split: StratifiedGroupKFold(10 folds) stratified by (material, recyclable) and
grouped by photo; fold 0 -> test, fold 1 -> val, folds 2-9 -> train (~80/10/10).

Extra columns (empty for clean rows):
  photo               TACO image id of the source photograph (group key)
  taco_category       TACO category name (60 categories)
  taco_supercategory  TACO supercategory name

manifest.csv is left untouched.

    python make_grouped_manifest.py
    python make_grouped_manifest.py --seed 0 --out manifest_grouped_s0.csv
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--annotations", default="Taco_data/annotations.json")
    ap.add_argument("--out", default="manifest_grouped.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    ann = json.load(open(args.annotations, encoding="utf-8"))
    photo_of = {a["id"]: a["image_id"] for a in ann["annotations"]}
    cat_of = {a["id"]: a["category_id"] for a in ann["annotations"]}
    cats = {c["id"]: (c["name"], c["supercategory"]) for c in ann["categories"]}

    clean = df[df.domain == "clean"].copy()
    wild = df[df.domain == "wild"].copy()
    ann_id = wild.path.str.extract(r"taco_(\d+)\.jpg")[0].astype(int)
    missing = (~ann_id.isin(photo_of.keys())).sum()
    if missing:
        raise SystemExit(f"{missing} wild crops have no annotation id in {args.annotations}")
    wild["photo"] = ann_id.map(photo_of).astype(int)
    wild["taco_category"] = ann_id.map(lambda a: cats[cat_of[a]][0])
    wild["taco_supercategory"] = ann_id.map(lambda a: cats[cat_of[a]][1])

    strata = wild.material * 2 + wild.recyclable
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold = np.empty(len(wild), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(wild, strata, groups=wild.photo)):
        fold[idx] = k
    wild["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))

    for col in ("photo", "taco_category", "taco_supercategory"):
        clean[col] = ""
    out = pd.concat([clean, wild])[["path", "material", "recyclable", "domain", "split",
                                    "photo", "taco_category", "taco_supercategory"]]
    out.to_csv(args.out, index=False)

    # ---- checks and summary
    photo_splits = wild.groupby("photo").split.nunique()
    assert (photo_splits == 1).all(), "a photo was split across splits"
    print(f"wrote {args.out}: {len(out)} rows ({len(clean)} clean unchanged, {len(wild)} wild "
          f"from {wild.photo.nunique()} photos)")
    print("every photo is in exactly one split: OK")
    old = df[df.domain == "wild"].split.value_counts()
    new = wild.split.value_counts()
    print("\nwild split sizes (old per-crop -> new per-photo):")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}  {old.get(s, 0):5d} -> {new.get(s, 0):5d}   "
              f"photos={wild[wild.split == s].photo.nunique():4d}   "
              f"non-recyclable={(wild[wild.split == s].recyclable == 0).mean():.3f}")
    tab = pd.crosstab(wild.material.map(lambda i: MATERIALS[i]), wild.split)[["train", "val", "test"]]
    tab["test_share"] = (tab["test"] / tab.sum(axis=1)).round(3)
    print("\nwild crops per material and split:")
    print(tab.to_string())


if __name__ == "__main__":
    main()
