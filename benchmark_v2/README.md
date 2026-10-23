# benchmark_v2 — photo-grouped benchmark (TACO-first)

This folder is a self-contained, corrected branch of the benchmark. The legacy
pipeline in the repo root stays untouched; compare freely.

## What was wrong in v1 (measured, not assumed)

| Problem | Evidence | Status here |
|---|---|---|
| Wild test crops shared a photo with train crops | 80.8% of wild test crops (manifest.csv); 27.6% of photos spanned splits | **fixed** — `manifest_v2.csv` splits by photo (`StratifiedGroupKFold`) |
| `cross_domain` selected best epoch on the **test set** (`config.py` resolved_select_on) | optimistic eval by construction | **fixed** — P2 now has a real wild/val selection split |
| Crop pipeline silently dropped 650/4784 (13.6%) annotations, class-skewed | trash kept only 67.7% (443 dropped), glass 80.3% | **accounted** — `crop_audit.csv`; all drops were the absolute 32px "too_small" filter |
| TrashNet duplicate clusters spanning splits | 14 clusters (dHash/md5), incl. 3 exact copies labelled as two different classes (e.g. glass115 ↔ metal91) | **reported** — `clean_dups.csv`; clean splits copied unchanged for P1 comparability |
| Recyclability policy undecidable items unflagged | Bottle cap vs lid, broken glass, Tetra Pak… | **flagged** — `label_mapping_v2.csv` (`ambiguous=1` on 34 categories + policy reason per row; **values unchanged** so runs stay comparable) |

## Files

| File | Purpose |
|---|---|
| `make_label_mapping_v2.py` | v1 mapping → `label_mapping_v2.csv` (adds `ambiguous`, `policy_reason`, `eval_include`, `policy_version`; changes no label values) |
| `make_manifest_v2.py` | builds `manifest_v2.csv`: crop audit + photo-grouped wild split + clean dedup report; writes `crop_audit.csv`, `clean_dups.csv`, `SPLIT_REPORT.md` |
| `label_mapping_v2.csv` | harmonized ontology with policy metadata |
| `manifest_v2.csv` | 6,659 rows (4,132 wild / 1,494 photos + 2,527 clean); columns: path, material, recyclable, domain, split, photo, ann_id, taco_category, taco_supercategory, bbox_rel_area, bbox_side_px, ambiguous |
| `SPLIT_REPORT.md` | one-page split facts |
| `waste_bench/` | patched copy of the training package: `manifest_v2.csv` default, `cross_domain` selects on wild val, run outputs under `benchmark_v2/runs/` |
| `smoke_test.py` | verifies manifest → splits → loaders with real image batches |

## Split layout (v2, seed 42)

- wild: train 3285 crops / 1192 photos · val 454 / 151 · test 393 / 151
- clean (unchanged): train 2019 · val 251 · test 257
- every photo in exactly one split — enforced by assertion at build time

## Running the benchmark (Kaggle GPU recommended)

From the **repo root** (paths in the manifest are repo-relative):

```bash
# P3 joint (dual head, phase-2 recipe)
python -m benchmark_v2.waste_bench.cli_dual --protocol joint --finetune_head \
    --unfreeze two_stages --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --epochs 30 --patience 8

# P2 cross-domain (now selects on wild val, tests on held-out wild test)
python -m benchmark_v2.waste_bench.cli_dual --protocol cross_domain --finetune_head \
    --unfreeze two_stages --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --epochs 15

# P1 in-domain (unchanged protocol, number for comparability)
python -m benchmark_v2.waste_bench.cli_dual --protocol in_domain --finetune_head \
    --unfreeze two_stages --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --epochs 30 --patience 8
```

Results land in `benchmark_v2/runs/<backbone>/<tag>_dual/` (`results.json`,
`checkpoint.pt`, report figure, console log).

## Known open items (next phases, by design)

1. TACO-only supervised baseline is intentionally **P3-side data**; a dedicated
   TACO-only protocol (train/val/test all wild) is the first experiment to add —
   it isolates "is TACO learnable at all" from the domain-gap narrative.
2. `ambiguous` rows stay in training/eval (37.1% of wild crops); a strict-eval
   table that excludes them is a one-line filter on the manifest and belongs in
   the analysis notebook, not the split decision.
3. The 14 TrashNet dup clusters still span splits; regrouping them changes P1
   numbers, so it must be a labelled ablation, not a silent fix.
