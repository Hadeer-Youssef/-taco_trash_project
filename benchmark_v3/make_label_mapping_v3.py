"""
Create an auditable v3 view of the existing TACO label mapping.

Important: this script does NOT pretend to infer missing physical facts or silently
change legacy labels. It preserves the v2 binary target for reproducibility and adds:
- label_status: provisional / conditional_review
- recyclability_label_v3: legacy value for unflagged rows, "unknown" for rows
  already flagged ambiguous in v2
- strict_eval_include: 0 for ambiguous rows
- train_include_legacy: 1 for all rows with a valid legacy binary label
- train_include_strict: 1 only for unambiguous rows
- policy_id: explicit marker that the legacy policy still needs localization/review

Run from repo root or any directory:
  python benchmark_v3/make_label_mapping_v3.py
  python benchmark_v3/make_label_mapping_v3.py --in benchmark_v2/label_mapping_v2.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="inp",
        default=str(repo / "benchmark_v2" / "label_mapping_v2.csv"),
        help="Input v2 mapping CSV",
    )
    parser.add_argument(
        "--out", default=str(here / "label_mapping_v3.csv"),
        help="Output v3 mapping CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.inp, keep_default_na=False)
    required = {
        "taco_category_id", "taco_category", "material_6class",
        "recyclability_binary", "n_annotations", "ambiguous",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input mapping is missing required columns: {sorted(missing)}")

    if df["taco_category_id"].duplicated().any():
        dup = df.loc[df["taco_category_id"].duplicated(False), "taco_category"].tolist()
        raise SystemExit(f"Duplicate taco_category_id values found: {dup}")

    valid_materials = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    bad_material = sorted(set(df["material_6class"]) - valid_materials)
    if bad_material:
        raise SystemExit(f"Unknown material_6class values: {bad_material}")

    labels = set(df["recyclability_binary"])
    bad_labels = labels - {"recyclable", "non_recyclable"}
    if bad_labels:
        raise SystemExit(f"Invalid legacy recyclability labels: {sorted(bad_labels)}")

    ambiguous = pd.to_numeric(df["ambiguous"], errors="raise").astype(int).astype(bool)
    df["label_status_v3"] = ambiguous.map(
        {True: "conditional_review", False: "provisional"}
    )
    df["recyclability_label_v3"] = df["recyclability_binary"].where(
        ~ambiguous, "unknown"
    )
    df["strict_eval_include"] = (~ambiguous).astype(int)
    df["train_include_legacy"] = 1
    df["train_include_strict"] = (~ambiguous).astype(int)
    df["policy_id_v3"] = "legacy_curbside_policy_unlocalized"
    df["review_required"] = ambiguous.astype(int)
    df["review_action"] = ambiguous.map({
        True: "review policy/condition and assign a justified label before strict use",
        False: "provisional only; confirm policy is appropriate for study location",
    })

    # Emit a compact, reproducible audit summary.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Input categories              : {len(df)}")
    print(f"Categories requiring review   : {int(ambiguous.sum())}")
    print(f"Strict-evaluation categories  : {int((~ambiguous).sum())}")
    print(f"Annotations in mapping        : {int(pd.to_numeric(df['n_annotations']).sum())}")
    print(f"Output                        : {out}")
    print("\nMaterial x legacy binary-label audit:")
    print(pd.crosstab(df["material_6class"], df["recyclability_binary"]).to_string())
    print("\nNOTE: annotation counts are source totals, not counts of surviving crops.")
    print("NOTE: 'provisional' does not mean independently validated.")


if __name__ == "__main__":
    main()
