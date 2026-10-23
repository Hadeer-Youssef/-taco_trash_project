"""Validate generated benchmark_v3 mapping and manifest without training a model."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", default=str(here / "label_mapping_v3.csv"))
    parser.add_argument("--manifest", default=str(here / "manifest_v3.csv"))
    args = parser.parse_args()

    mapping_path, manifest_path = Path(args.mapping), Path(args.manifest)
    if not mapping_path.exists():
        raise SystemExit(f"Missing {mapping_path}; run make_label_mapping_v3.py first")
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}; run make_manifest_v3.py first")

    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    manifest = pd.read_csv(manifest_path)

    expected_labels = {"recyclable", "non_recyclable", "unknown"}
    bad = set(mapping["recyclability_label_v3"]) - expected_labels
    if bad:
        raise SystemExit(f"Unexpected v3 labels: {sorted(bad)}")
    if mapping["taco_category_id"].duplicated().any():
        raise SystemExit("Duplicate category IDs in mapping")
    if not set(mapping["strict_eval_include"].astype(int)).issubset({0, 1}):
        raise SystemExit("strict_eval_include must be binary")

    unknown_should_exclude = mapping.loc[
        mapping["recyclability_label_v3"].eq("unknown"), "strict_eval_include"
    ].astype(int)
    if not unknown_should_exclude.eq(0).all():
        raise SystemExit("Unknown labels must have strict_eval_include=0")

    needed = {"recyclable", "recyclable_v3", "strict_eval_include", "domain", "split"}
    if needed - set(manifest.columns):
        raise SystemExit(f"Manifest missing columns: {sorted(needed - set(manifest.columns))}")
    if not set(manifest["strict_eval_include"].dropna().astype(int)).issubset({0, 1}):
        raise SystemExit("Manifest strict_eval_include must be binary")

    # The legacy labels must remain present and valid for comparability.
    if not set(manifest["recyclable"].dropna().astype(int)).issubset({0, 1}):
        raise SystemExit("Legacy recyclable column contains values outside {0,1}")
    known_v3 = manifest["recyclable_v3"].dropna().astype(int)
    if not set(known_v3).issubset({0, 1}):
        raise SystemExit("Known recyclable_v3 values must be binary")
    if manifest.loc[manifest["recyclable_v3"].isna(), "strict_eval_include"].eq(1).any():
        raise SystemExit("Rows with unknown recyclable_v3 cannot be strict-eval included")

    # Ensure strict manifest hasn't lost its split labels or paths.
    if manifest["path"].isna().any() or manifest["split"].isna().any():
        raise SystemExit("Manifest contains missing path/split")
    print("v3 validation passed")
    print(f"Mapping categories: {len(mapping)}")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Unknown v3 labels: {int(manifest['recyclable_v3'].isna().sum())}")
    print(f"Strict-eval rows: {int(manifest['strict_eval_include'].sum())}")


if __name__ == "__main__":
    main()
