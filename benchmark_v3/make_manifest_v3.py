"""
Build manifest_v3.csv from the existing, photo-grouped manifest_v2.csv.

This is a non-destructive label audit layer: legacy labels and split assignments
remain unchanged, while unresolved TACO labels become unknown for strict use.
TrashNet has only coarse material labels, so it is excluded from strict binary
recyclability evaluation but remains available for legacy comparisons/material
classification.

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
        help="write only rows eligible for strict binary recyclability evaluation",
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

    # Keep legacy values for reproducibility; new target uses nullable integer.
    manifest["recyclable_v3"] = manifest["recyclable"].astype("Int64")
    # TrashNet has material-level labels, not item-level recycling policy labels.
    # Keep it for legacy/material experiments, but not strict binary recyclability.
    manifest["strict_eval_include"] = 0
    manifest["label_source_note"] = "TrashNet coarse material-level legacy label; excluded from strict binary recyclability evaluation"

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
    print(f"Rows eligible for strict binary recyclability evaluation: {int(manifest['strict_eval_include'].sum())}")
    print(f"Rows with unknown v3 recyclability: {int(manifest['recyclable_v3'].isna().sum())}")
    print("Legacy 'recyclable' column and split assignments are preserved.")
    print("TrashNet remains available but is excluded from strict binary recyclability evaluation.")


if __name__ == "__main__":
    main()
