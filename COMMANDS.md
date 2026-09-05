# Command Log — every experiment, with/without comparisons

All commands assume you're in the repo root on a GPU box (Kaggle/Colab),
with `manifest.csv` path-normalized (forward slashes) and `git lfs pull`
already run. Each run writes:
- `results_<tag>.json` — metrics (feed into `aggregate_results.py`)
- `results_<tag>.log` — full console log (timestamps, per-epoch loss/acc/F1,
  class weights if used, training time, GPU util, inference latency)
- `reports_<tag>/report_<backbone>.png` — the 5-panel visual report

Run everything for one backbone before moving to the next if you want to
inspect results incrementally (see the earlier model-by-model workflow) —
just drop `--backbones <others>` from any line below to isolate one model.

BACKBONES = resnet50, densenet121, mobilenet_v2, inception_v3 (vgg13 dropped)

---

## 0. Baseline sanity run (no flags) — one backbone, quick smoke check

```bash
python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2 --epochs 2 --out results_smoke.json --report_dir reports_smoke
```

---

## 1. Core protocols — all 4 backbones, defaults (aug_mode=auto, no class weighting, no finetune_head)

```bash
# P1 in_domain (aug_mode auto-picks "mild")
python benchmark_backbones.py --protocol in_domain --epochs 15 --out results_in_domain.json --report_dir reports_in_domain

# P2 cross_domain — HEADLINE result (aug_mode auto-picks "strong")
python benchmark_backbones.py --protocol cross_domain --epochs 15 --out results_cross_domain.json --report_dir reports_cross_domain

# P3 joint (aug_mode auto-picks "strong")
python benchmark_backbones.py --protocol joint --epochs 15 --out results_joint.json --report_dir reports_joint
```

Compare: `material_acc` / `material_macro_f1` should drop clearly from
in_domain → cross_domain (the domain gap), and joint should sit between
cross_domain and in_domain (recovering some of the gap).

---

## 2. WITH vs WITHOUT `--class_weighted` (trash-class imbalance)

Run both, same protocol/backbone/epochs, only this flag differs.

```bash
# WITHOUT class weighting (baseline)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --out results_cross_resnet50_noweight.json --report_dir reports_cross_resnet50_noweight

# WITH class weighting
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --class_weighted --out results_cross_resnet50_weighted.json --report_dir reports_cross_resnet50_weighted
```

Compare: check `non_recyclable_recall` and per-class `trash` recall in
`material_report` — weighting should raise both if it's helping. The log
for the WITH run also prints the exact per-class counts/weights used
(`class_weights_from` logs this before training starts).

Repeat for `joint` (TACO's imbalance is worse there — plastic-heavy):
```bash
python benchmark_backbones.py --protocol joint --backbones resnet50 --epochs 15 --out results_joint_resnet50_noweight.json --report_dir reports_joint_resnet50_noweight
python benchmark_backbones.py --protocol joint --backbones resnet50 --epochs 15 --class_weighted --out results_joint_resnet50_weighted.json --report_dir reports_joint_resnet50_weighted
```

---

## 3. WITH vs WITHOUT `--finetune_head` (frozen linear probe vs. richer head + partial unfreeze)

Run for **every** backbone (all 4 support it) on the headline protocol.

```bash
# resnet50
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --out results_cross_resnet50_frozen.json --report_dir reports_cross_resnet50_frozen
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --finetune_head --out results_cross_resnet50_finetuned.json --report_dir reports_cross_resnet50_finetuned

# densenet121
python benchmark_backbones.py --protocol cross_domain --backbones densenet121 --epochs 15 --out results_cross_densenet121_frozen.json --report_dir reports_cross_densenet121_frozen
python benchmark_backbones.py --protocol cross_domain --backbones densenet121 --epochs 15 --finetune_head --out results_cross_densenet121_finetuned.json --report_dir reports_cross_densenet121_finetuned

# mobilenet_v2
python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2 --epochs 15 --out results_cross_mobilenet_v2_frozen.json --report_dir reports_cross_mobilenet_v2_frozen
python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2 --epochs 15 --finetune_head --out results_cross_mobilenet_v2_finetuned.json --report_dir reports_cross_mobilenet_v2_finetuned

# inception_v3
python benchmark_backbones.py --protocol cross_domain --backbones inception_v3 --epochs 15 --out results_cross_inception_v3_frozen.json --report_dir reports_cross_inception_v3_frozen
python benchmark_backbones.py --protocol cross_domain --backbones inception_v3 --epochs 15 --finetune_head --out results_cross_inception_v3_finetuned.json --report_dir reports_cross_inception_v3_finetuned
```

Compare: `material_acc`, `material_macro_f1`, and `history.train_time_sec`
(finetuning trains more parameters, so expect it to be slower per epoch —
check whether the accuracy gain is worth the extra `train_time_sec` and
`inference_latency_ms_per_image`).

---

## 4. WITH vs WITHOUT augmentation strength (`--aug_mode`)

`auto` already picks the sensible default per protocol (mild for in_domain,
strong for cross_domain/joint) — these runs override it explicitly to
measure the effect directly.

```bash
# cross_domain: off vs mild vs strong (auto default)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode off    --out results_cross_resnet50_aug_off.json    --report_dir reports_cross_resnet50_aug_off
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode mild   --out results_cross_resnet50_aug_mild.json   --report_dir reports_cross_resnet50_aug_mild
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode strong --out results_cross_resnet50_aug_strong.json --report_dir reports_cross_resnet50_aug_strong

# in_domain: mild (auto default) vs strong — does strong aug cost accuracy with no domain gap to fight?
python benchmark_backbones.py --protocol in_domain --backbones resnet50 --epochs 15 --aug_mode mild   --out results_indomain_resnet50_aug_mild.json   --report_dir reports_indomain_resnet50_aug_mild
python benchmark_backbones.py --protocol in_domain --backbones resnet50 --epochs 15 --aug_mode strong --out results_indomain_resnet50_aug_strong.json --report_dir reports_indomain_resnet50_aug_strong
```

Compare: for cross_domain, expect off < mild < strong on `material_acc`
(more domain-randomization closes more of the gap). For in_domain, check
whether `strong` actually *underperforms* `mild` — that's the hypothesis
this comparison is built to test.

---

## 5. WITH vs WITHOUT early stopping (`--patience`)

```bash
# WITHOUT: fixed epoch count (baseline, whatever epoch it ends on is final)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --out results_cross_resnet50_fixed_epochs.json --report_dir reports_cross_resnet50_fixed_epochs

# WITH: large epoch budget + early stopping, keeps best-val-loss weights
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 100 --patience 5 --out results_cross_resnet50_earlystop.json --report_dir reports_cross_resnet50_earlystop
```

Compare: check `history.best_epoch` and `history.best_val_loss` in the
early-stop run's JSON vs. the fixed-epoch run's final-epoch numbers —
early stopping should match or beat the fixed-epoch run without needing to
guess the right epoch count in advance.

---

## 6. Ablations from RUNBOOK.md Phase 3 (background removal, DANN, material-rule)

```bash
# (a) augmentation off — quantifies how much domain-randomizing aug closes the gap
python benchmark_backbones.py --protocol cross_domain --aug_mode off --backbones resnet50 --epochs 15 --out results_p2_noaug.json --report_dir reports_p2_noaug

# (b) class-weighted (long-tail) joint
python benchmark_backbones.py --protocol joint --class_weighted --backbones resnet50 --epochs 15 --out results_joint_weighted.json --report_dir reports_joint_weighted

# (c) background-removal ablation
python gradcam_ablation.py generate
# then point a benchmark run at manifest_masked.csv (train on clean, eval on masked wild):
python benchmark_backbones.py --manifest manifest_masked.csv --protocol cross_domain --backbones resnet50 --epochs 15 --out results_p2_masked.json --report_dir reports_p2_masked

# (d) DANN and material-rule-vs-binary-head (waste_pipeline.py, not benchmark_backbones.py)
python waste_pipeline.py --taco_images Taco_data --taco_ann Taco_data/annotations.json --trashnet trashnet_dataset-resized --protocol cross_domain --dann
python waste_pipeline.py --taco_images Taco_data --taco_ann Taco_data/annotations.json --trashnet trashnet_dataset-resized --protocol cross_domain --material_rule

# Grad-CAM money figure (needs a clean-trained resnet50 checkpoint)
python gradcam_ablation.py gradcam --ckpt resnet50_clean.pt --n 8
```

---

## 7. Full grid — every backbone x every flag combination (headline protocol only)

For a complete WITH/WITHOUT matrix on the headline `cross_domain` protocol,
across all 4 backbones, run section 2–4 blocks substituting each backbone
name in place of `resnet50`. As a checklist per backbone:

- [ ] baseline (no flags)
- [ ] `--class_weighted`
- [ ] `--finetune_head`
- [ ] `--finetune_head --class_weighted` (combined)
- [ ] `--aug_mode off`
- [ ] `--aug_mode strong` (should match baseline, since strong is cross_domain's auto default — run anyway to confirm)

---

## 8. Assemble comparison tables after all runs

```bash
python aggregate_results.py \
  --in_domain results_in_domain.json --cross results_cross_domain.json --joint results_joint.json \
  --ablation no_aug=results_p2_noaug.json --ablation bg_removed=results_p2_masked.json
```

For the WITH/WITHOUT comparisons in sections 2–5 (not natively covered by
`aggregate_results.py`'s table format), compare the saved JSONs directly,
e.g.:

```python
import json
a = json.load(open("results_cross_resnet50_frozen.json"))["results"]["resnet50"]
b = json.load(open("results_cross_resnet50_finetuned.json"))["results"]["resnet50"]
print("frozen   :", a["material_acc"], a["material_macro_f1"], a["non_recyclable_recall"])
print("finetuned:", b["material_acc"], b["material_macro_f1"], b["non_recyclable_recall"])
print("train_time frozen/finetuned:", a["history"]["train_time_sec"], b["history"]["train_time_sec"])
```

---

## Naming convention used above

`results_<protocol>_<backbone>_<variant>.json` /
`reports_<protocol>_<backbone>_<variant>/` — keep this consistent for any
new comparison so `aggregate_results.py` and ad-hoc comparison scripts can
find files by pattern later.
