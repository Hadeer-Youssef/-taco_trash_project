# Runbook — Cross-Domain Waste Classification

Pipeline that turns the harmonized TACO+TrashNet benchmark into the paper's tables
and figures. Phases 0–1 are **already done locally** (artifacts committed below).
Phases 2–4 need a GPU (run on Kaggle/Colab).

## Already produced (done, CPU)

| artifact | what it is |
|---|---|
| `label_mapping.csv` (v2) | 60 TACO categories → 6 materials → binary recyclability, keyed on `taco_category_id`, full coverage, `mapped_via`/`n_annotations` columns |
| `coverage_report.md` | Table T2 — ontology coverage (all 60 cats, 0 fallbacks) |
| `crops_material/` | 4132 TACO material crops (32px min-size filter applied) |
| `manifest.csv` | 6659 rows: path, material, recyclable, domain (clean/wild), split — the single source of truth |
| `table_t1.md` | Table T1 — dataset composition |
| `fig_crops_grid.png` | F6 sanity grid (material × clean|wild) |

Key numbers: wild(TACO)=4132, clean(TrashNet)=2527. TACO is plastic-heavy
(2005) with a glass/paper long tail (~200 each). **Note (paper limitation):** in
TrashNet, non-recyclable ≈ the "trash" class only, so recyclability is
domain-confounded there; TACO's item-aware policy splits within materials
(plastic: 524 rec / 1481 non-rec). Report binary results with this caveat and lean
on the 6-class material metric as the primary headline.

## Phase 2 — core experiments (GPU)

```bash
# P2 cross-domain (headline) — all 5 backbones
python benchmark_backbones.py --protocol cross_domain --epochs 15 --out results_cross_domain.json

# P1 in-domain and P3 joint — best backbone + resnet50 is enough to save compute
python benchmark_backbones.py --protocol in_domain --backbones resnet50 densenet121 --epochs 15 --out results_in_domain.json
python benchmark_backbones.py --protocol joint --backbones resnet50 densenet121 --epochs 15 --out results_joint.json
```

Expect P2 material acc to drop clearly below P1 (the domain gap). If it does **not**
drop, the harmonization is wrong — stop and recheck before running the rest.

## Phase 3 — ablations (GPU)

```bash
# (a) augmentation off — quantifies how much domain-randomizing aug closes the gap
python benchmark_backbones.py --protocol cross_domain --no_domain_randomize --backbones resnet50 --out results_p2_noaug.json

# (b) class-weighted (long-tail) joint
python benchmark_backbones.py --protocol joint --class_weighted --backbones resnet50 --out results_joint_weighted.json

# (c) background-removal ablation: generate masked crops, then evaluate
python gradcam_ablation.py generate              # writes crops_material_masked/ + manifest_masked.csv (CPU ok)
#   train resnet50 on clean, eval on manifest_masked (point benchmark at it) -> results_p2_masked.json

# (d) DANN (cuttable) and material-rule-vs-binary-head: see waste_pipeline.py flags
python waste_pipeline.py --taco_images Taco_data --taco_ann Taco_data/annotations.json \
    --trashnet trashnet_dataset-resized --protocol cross_domain --dann
python waste_pipeline.py ... --protocol cross_domain --material_rule

# Grad-CAM money figure (needs a clean-trained resnet50 checkpoint)
python gradcam_ablation.py gradcam --ckpt resnet50_clean.pt --n 8   # -> fig_gradcam.png
```

## Phase 4 — assemble tables & figures

```bash
python aggregate_results.py \
  --in_domain results_in_domain.json --cross results_cross_domain.json --joint results_joint.json \
  --ablation no_aug=results_p2_noaug.json --ablation bg_removed=results_p2_masked.json
# -> table_t3_main.md, table_t4_backbones.md, table_t5_ablations.md, fig_domain_gap.png
```

`methodology.md` is the paper skeleton (sections map 1:1 to these results).
Release `label_mapping.csv` + `manifest.csv` + the crop scripts + `coverage_report.md`
as the benchmark.

## Files

- `waste_pipeline.py` — harmonization, crops, WasteNet (+DANN), three protocols
- `benchmark_backbones.py` — 5-backbone benchmark driven by manifest
- `gradcam_ablation.py` — bg-removal crops + Grad-CAM
- `make_artifacts.py` — T1 + sanity grid
- `aggregate_results.py` — T3/T4/T5 + domain-gap figure
