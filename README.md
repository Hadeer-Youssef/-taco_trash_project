# Cross-Domain Waste Classification: Harmonized TACO + TrashNet Benchmark

A unified benchmark and study for waste-image classification that bridges two
siloed datasets — **TrashNet** (clean, studio-shot, single-object) and **TACO**
(cluttered, in-the-wild, detection) — onto a shared **6-class material** space and
a **binary recyclability** label.

**Contribution.** The unified 60→6→2 taxonomy is the vehicle; the finding is a
**cross-domain generalization** result: classifiers trained on clean TrashNet
collapse on wild TACO, and joint training + domain-randomizing augmentation +
a background-removal / Grad-CAM ablation close and *explain* the gap (background
shortcut learning). A recyclability head turns it into a deployable smart-bin model.

See [`methodology.md`](methodology.md) for the full write-up.

## Repo layout

| Path | Purpose |
|---|---|
| `label_mapping.csv` | TACO 60 categories → 6 materials → binary recyclability (keyed on `taco_category_id`, full coverage) |
| `coverage_report.md` | Ontology coverage report (Table T2) |
| `manifest.csv` | Single source of truth: path, material, recyclable, domain (clean/wild), split |
| `waste_pipeline.py` | Harmonization, COCO→material crops, `WasteNet` (+DANN), three protocols |
| `benchmark_backbones.py` | Five backbones (ResNet50, DenseNet121, MobileNetV2, VGG13, Inception_v3) × three protocols |
| `gradcam_ablation.py` | Background-removal crop set + Grad-CAM shortcut-learning evidence |
| `make_artifacts.py` | Dataset table (T1) + sanity crop grid |
| `aggregate_results.py` | Assembles result tables (T3–T5) + domain-gap figure |
| `update_Taco_classification/cv_helper_functions.py` | `try_square_crop`, `mask_and_square_crop` (crop + mask) |
| `*.ipynb` | TrashNet transfer-learning notebook + TACO preprocessing notebook |
| `RUNBOOK.md` | Step-by-step commands to reproduce every table/figure |

## Data (not committed — download / regenerate)

The raw datasets and generated crops are excluded from git (size). To reproduce:

1. **TrashNet** — download the resized dataset into `trashnet_dataset-resized/`
   with subfolders `cardboard/ glass/ metal/ paper/ plastic/ trash/`.
2. **TACO** — download images into `Taco_data/batch_1..15/` and the COCO
   `Taco_data/annotations.json` (from <https://github.com/pedropro/TACO>).
3. Regenerate the harmonized crops + manifest:
   ```bash
   python -c "import waste_pipeline as wp; \
     m=wp.load_mapping('label_mapping.csv','Taco_data/annotations.json'); \
     wild=wp.taco_to_crops('Taco_data','Taco_data/annotations.json',m,'crops_material'); \
     clean=wp.index_trashnet('trashnet_dataset-resized'); \
     wp.build_manifest(wild+clean,'manifest.csv')"
   ```

Then follow [`RUNBOOK.md`](RUNBOOK.md) for the experiments (GPU recommended).

## Datasets & credits

- TACO — Trash Annotations in Context (Proença & Simões), COCO-format detection data.
- TrashNet (Yang & Thung), 6-class material classification.
