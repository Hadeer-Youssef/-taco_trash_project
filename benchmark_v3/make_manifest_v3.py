"""
Build manifest_v3.csv from the existing, photo-grouped manifest_v2.csv.

This is intentionally a non-destructive label audit layer:
- legacy 'recyclable' stays untouched for reproducing prior experiments;
- recyclable_v3 is blank/unknown for rows requiring policy or condition review;
- strict_eval_include filters those rows out of strict metrics;
- split assignments and image paths are preserved exactly.

Run from repository root:
  python benchmark_v3/make_manifest_v3.py
  python benchmark_v3/make_manifest_v3.py --strict-only
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(repo / "benchmark_v2" / "manifest_v2.csv"))
    parser.add_argument("--mapping", default=str(here / "label_mapping_v3.csv"))
    parser.add_argument("--out", default=str(here / "manifest_v3.csv"))
    parser.add_argument(
        "--strict-only", action="store_true",
        help="write only rows eligible for strict evaluation; default keeps all rows and flags them",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    mapping = pd.read_csv(args.mapping, keep_default_na=False)

    required_manifest = {"path", "domain", "split", "taco_category", "recyclable"}
    required_mapping = {"taco_category", "recyclability_label_v3", "strict_eval_include"}
    if required_manifest - set(manifest.columns):
        raise SystemExit(f"Manifest missing columns: {sorted(required_manifest - set(manifest.columns))}")
    if required_mapping - set(mapping.columns):
        raise SystemExit(f"Mapping missing columns: {sorted(required_mapping - set(mapping.columns))}")

    if mapping["taco_category"].duplicated().any():
        raise SystemExit("Mapping contains duplicate taco_category names")

    labels = mapping.set_index("taco_category")["recyclability_label_v3"].to_dict()
    include = mapping.set_index("taco_category")["strict_eval_include"].to_dict()

    wild_mask = manifest["domain"].eq("wild")
    missing = sorted(set(manifest.loc[wild_mask, "taco_category"].dropna()) - set(labels))
    if missing:
        raise SystemExit(f"Wild categories absent from v3 mapping: {missing}")

    # Clean/TrashNet rows do not have item-level TACO category labels.
    # Keep their legacy coarse label, but explicitly mark their source limitation.
    manifest["recyclable_v3"] = manifest["recyclable"].astype("Int64")
    manifest["strict_eval_include"] = 1
    manifest["label_source_note"] = "TrashNet coarse material-level legacy label"
    for idx in manifest.index[wild_mask]:
        category = manifest.at[idx, "taco_category"]
        label = labels[category]
        manifest.at[idx, "recyclable_v3"] = (
            1 if label == "recyclable" else 0 if label == "non_recyclable" else pd.NA
        )
        manifest.at[idx, "strict_eval_include"] = int(include[category])
        manifest.at[idx, "label_source_note"] = (
            "TACO item-level label; policy/condition review required"
            if label == "unknown"
            else "TACO item-level provisional legacy policy label"
        )

    manifest["strict_eval_include"] = manifest["strict_eval_include"].astype(int)
    if args.strict_only:
        manifest = manifest.loc[manifest["strict_eval_include"].eq(1)].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)

    print(f"Wrote: {out}")
    print(f"Rows: {len(manifest)}")
    print(f"Rows eligible for strict evaluation: {int(manifest['strict_eval_include'].sum())}")
    print(f"Rows with unknown v3 recyclability: {int(manifest['recyclable_v3'].isna().sum())}")
    print("Legacy 'recyclable' column and split assignments are preserved.")
    print("Warning: TrashNet labels remain coarse material-level labels, not item-level policy truth.")


if __name__ == "__main__":
    main()
