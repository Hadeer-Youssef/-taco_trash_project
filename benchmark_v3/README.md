# benchmark_v3 — auditable recyclability labels

This layer adds a policy-aware view without overwriting benchmark_v2 or changing
its historical labels. It is a **reviewable research baseline**, not a claim that
the old curbside labels are universally correct.

## Why v3 exists

The existing v2 mapping has a binary legacy label even when its own notes say
that the answer depends on contamination, product construction, or local facility
acceptance. A single `ambiguous` boolean also combines several different causes.
v3 therefore exposes unresolved cases as `unknown` for strict evaluation while
preserving the old label for reproducibility.

## Files

- `make_label_mapping_v3.py`: validates the source mapping and creates
  `label_mapping_v3.csv`.
- `make_manifest_v3.py`: adds `recyclable_v3`, `strict_eval_include`, and
  label provenance to the existing photo-grouped manifest.
- `validate_v3.py`: checks mapping and manifest invariants.

## Run

From repository root:

```bash
python benchmark_v3/make_label_mapping_v3.py
python benchmark_v3/make_manifest_v3.py
python benchmark_v3/validate_v3.py
```

To create a CSV containing only rows eligible for strict evaluation:

```bash
python benchmark_v3/make_manifest_v3.py --strict-only --out benchmark_v3/manifest_v3_strict.csv
```

Default `manifest_v3.csv` retains all rows and marks strict eligibility. The
original `recyclable` column remains unchanged. New `recyclable_v3` is 1, 0,
or blank/unknown. Do not feed `recyclable_v3` into a binary loss without first
filtering unknown rows.

## Important interpretation

- `label_status_v3=conditional_review` means the current mapping's own metadata
  indicates a conditional or policy-dependent decision. It is not a model
  prediction and does not mean the category is physically unrecyclable.
- `label_status_v3=provisional` means it was not flagged by the v2 ambiguity
  list. It does **not** mean independent ground-truth validation.
- `policy_id_v3=legacy_curbside_policy_unlocalized` explicitly records that a
  specific municipality/facility policy has not yet been selected.
- TrashNet labels are coarse material-level legacy labels. In particular, a
  TrashNet image labelled plastic does not reveal whether it is a bottle, film,
  straw, or another item. Its binary labels are not directly equivalent to
  item-level TACO recyclability labels. Keep this limitation in reports.
- The v3 builder intentionally preserves existing split assignments, including
  known TrashNet duplicate-cluster leakage. Treat duplicate-cluster regrouping as
  a separately reported experiment rather than silently changing the split.
- This layer does not repair the existing model/training code automatically.
  Strict experiments must use `strict_eval_include == 1` and `recyclable_v3`.
  Legacy runs should continue using the legacy `recyclable` column.

## Before publication

Choose and document the actual recycling policy (country/municipality/facility),
review conditional categories against an authoritative source, version the policy,
and manually audit cross-label exact duplicates in TrashNet. Then update the
mapping decisions with evidence and keep an audit trail for every changed label.
