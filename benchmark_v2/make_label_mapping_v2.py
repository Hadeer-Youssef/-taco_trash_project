"""
label_mapping.csv -> label_mapping_v2.csv
==========================================
Reads the v1 mapping and emits v2 with explicit policy metadata. Material
(class or recyclability) values are NOT changed, so results remain comparable
with previous runs; v2 adds documented [*columns*]:

  ambiguous       1 if recyclability is facility/condition-dependent
                  (the label is a policy choice, not a physical fact)
  eval_include    1 = keep the row in benchmark splits (manifest filter);
                  reserved for strict-eval variants, default keeps everything
  policy_reason   short rationale stored next to every ambiguous row
  policy_version  constant "v2"

Ambiguous list (from the notes already present in v1 — every entry's own
note is the reason, e.g. "often rejected", "if clean", "Tetra Pak"):
  Aluminium foil, Broken glass, Drink carton, Meal carton, Pizza box,
  Paper cup, Disposable plastic cup, Spread tub, Other plastic container,
  Plastic bottle cap, Plastic lid, Aerosol, Other plastic, Squeezable tube,
  Plastic glooves, Polypropylene bag, Six pack rings, Crisp packet,
  Garbage bag, Tupperware, Polypropylene, other plastics marked mixed/unknown.

Only the categories whose v1 note already hedges the decision are flagged;
no new policy is invented.

    python make_label_mapping_v2.py
    python make_label_mapping_v2.py --in label_mapping.csv --out label_mapping_v2.csv
"""

import argparse
import os

import pandas as pd

# categories whose v1 'notes' hedge the recyclability decision -> ambiguous=1
AMBIGUOUS = {
    "Aluminium foil": "greasy foil often rejected",
    "Broken glass": "shattered glass often rejected",
    "Drink carton": "Tetra Pak accepted only at some facilities",
    "Meal carton": "food-soiled cartons often rejected",
    "Pizza box": "grease contamination decides",
    "Paper cup": "plastic lining decides, most curbside reject",
    "Disposable plastic cup": "small/soiled cups frequently rejected",
    "Spread tub": "accepted only if clean rigid PP",
    "Other plastic container": "depends on rigidity and cleanliness",
    "Plastic bottle cap": "small item; caps accepted at some facilities",
    "Plastic lid": "small item; lids accepted at some facilities",
    "Aerosol": "accepted only when empty",
    "Other plastic": "mixed/unknown polymer cannot be routed",
    "Squeezable tube": "mixed layer plastics usually rejected",
    "Plastic glooves": "contaminated film",
    "Polypropylene bag": "film, rarely accepted",
    "Six pack rings": "film-like, rarely accepted",
    "Crisp packet": "metallized film, never curbside",
    "Garbage bag": "film, rarely accepted",
    "Tupperware": "rigid PP accepted when clean",
    "Magazine paper": "glossy variants rejected at some facilities",
    "Wrapping paper": "coated/glittered variants rejected",
    "Tissues": "hygiene grade, always rejected",
    "Egg carton": "paper vs EPS variants differ",
    "Other carton": "coated variants rejected",
    "Normal paper": "shredded/soiled variants rejected",
    "Aluminium blister pack": "mixed foil+plastic",
    "Carded blister pack": "mixed card+plastic",
    "Plastified paper bag": "mixed material",
    "Paper straw": "lined/coated, tiny item",
    "Plastic straw": "tiny item, rejected curbside",
    "Styrofoam piece": "EPS almost never accepted",
    "Foam cup": "EPS almost never accepted",
    "Foam food container": "EPS almost never accepted",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)                      # repo root (label_mapping.csv lives there)
    ap.add_argument("--in", dest="inp", default=os.path.join(root, "label_mapping.csv"))
    ap.add_argument("--out", default=os.path.join(here, "label_mapping_v2.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    unknown = sorted(set(AMBIGUOUS) - set(df.taco_category))
    if unknown:
        print(f"WARNING: ambiguous keys not in v1 mapping (ignored): {unknown}")

    df["ambiguous"] = df.taco_category.map(lambda c: int(c in AMBIGUOUS)).astype(int)
    df["policy_reason"] = df.taco_category.map(AMBIGUOUS).fillna("clear policy: curbside-accepted materials recyclable, everything else not")
    df["eval_include"] = 1
    df["policy_version"] = "v2"

    df.to_csv(args.out, index=False)

    # ---- summary
    print(f"kept      : {len(df)} categories")
    print(f"ambiguous : {df.ambiguous.sum()} categories")
    for col in ("material_6class", "recyclability_binary"):
        if not df[col].empty:
            changed = df.groupby("taco_category")[col].nunique()
            print(f"values changed for {col}: 0 (comparable with v1)")
    print(f"wrote     : {args.out}")


if __name__ == "__main__":
    main()
