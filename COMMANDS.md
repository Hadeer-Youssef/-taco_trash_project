# Command Log — every experiment, with/without comparisons

All commands assume you're in the repo root on a GPU box (Kaggle/Colab),
with `manifest.csv` path-normalized (forward slashes) and `git lfs pull`
already run.

## Output layout (auto-organized — no manual naming needed)

Every run automatically writes to a **unique folder per backbone per
configuration**, so different flag combinations never overwrite each other
and you never have to invent `--out`/`--report_dir`/`--ckpt_dir` names by
hand:

```
runs/<backbone>/<protocol>_aug-<mode>[_ft][_cw][_lr-none]/
    report_<backbone>.png   # 5-panel visual report
    checkpoint.pt           # best-epoch weights + metadata (predict.py-ready)
    results.json            # this backbone's metrics for this exact config
runs/_summaries/<tag>.json  # combined metrics for every backbone in this command
runs/_summaries/<tag>.log   # full console log for this command
```

The tag suffixes: `ft` = `--finetune_head` was on, `cw` = `--class_weighted`
was on, `lr-none` = `--lr_schedule none` (cosine decay is the default and
omitted from the tag). E.g. `--protocol cross_domain --backbones resnet50
--finetune_head --class_weighted` writes to
`runs/resnet50/cross_domain_aug-strong_ft_cw/`.

Only pass `--out` / `--report_dir` / `--ckpt_dir` / `--log_file` if you
specifically want to override this auto-layout with a fixed path.

Run everything for one backbone before moving to the next if you want to
inspect results incrementally — just drop `--backbones <others>` from any
line below to isolate one model.

BACKBONES = resnet50, densenet121, mobilenet_v2, inception_v3 (vgg13 dropped)

---

## 0. Baseline sanity run — one backbone, quick smoke check

```bash
python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2 --epochs 2
```
**Output:** `runs/mobilenet_v2/cross_domain_aug-strong/{report_mobilenet_v2.png, checkpoint.pt, results.json}` plus `runs/_summaries/cross_domain_aug-strong.{json,log}`.

---

## 1. Core protocols — all 4 backbones, defaults

Defaults: `aug_mode=auto` (mild for in_domain, strong for cross_domain/joint), `lr_schedule=cosine`, no class weighting, no finetune_head.

```bash
# P1 in_domain (aug_mode auto-picks "mild")
python benchmark_backbones.py --protocol in_domain --epochs 15

# P2 cross_domain — HEADLINE result (aug_mode auto-picks "strong")
python benchmark_backbones.py --protocol cross_domain --epochs 15

# P3 joint (aug_mode auto-picks "strong")
python benchmark_backbones.py --protocol joint --epochs 15
```
**Output:** three folders, one per protocol, each containing all 4 backbones' subfolders, e.g. `runs/resnet50/in_domain_aug-mild/`, `runs/densenet121/in_domain_aug-mild/`, etc., plus one `runs/_summaries/<protocol>_aug-<mode>.json` per command bundling all 4 backbones' metrics together (feed straight into `aggregate_results.py`).

Compare: `material_acc` / `material_macro_f1` should drop clearly from
in_domain → cross_domain (the domain gap), and joint should sit between
cross_domain and in_domain (recovering some of the gap).

---

## 2. WITH vs WITHOUT `--class_weighted` (trash-class imbalance)

```bash
# WITHOUT class weighting (baseline)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15

# WITH class weighting
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --class_weighted
```
**Output:** `runs/resnet50/cross_domain_aug-strong/` (without) vs. `runs/resnet50/cross_domain_aug-strong_cw/` (with) — the `_cw` suffix keeps them side by side automatically.

Compare: check `non_recyclable_recall` and per-class `trash` recall in
`material_report` — weighting should raise both if it's helping. The log
also prints the exact per-class counts/weights used before training starts.

Repeat for `joint` (TACO's imbalance is worse there — plastic-heavy):
```bash
python benchmark_backbones.py --protocol joint --backbones resnet50 --epochs 15
python benchmark_backbones.py --protocol joint --backbones resnet50 --epochs 15 --class_weighted
```
**Output:** `runs/resnet50/joint_aug-strong/` vs. `runs/resnet50/joint_aug-strong_cw/`.

---

## 3. WITH vs WITHOUT `--finetune_head` (frozen linear probe vs. richer head + partial unfreeze)

Run for **every** backbone (all 4 support it) on the headline protocol.

```bash
python benchmark_backbones.py --protocol cross_domain --backbones resnet50      --epochs 15
python benchmark_backbones.py --protocol cross_domain --backbones resnet50      --epochs 15 --finetune_head

python benchmark_backbones.py --protocol cross_domain --backbones densenet121   --epochs 15
python benchmark_backbones.py --protocol cross_domain --backbones densenet121   --epochs 15 --finetune_head

python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2  --epochs 15
python benchmark_backbones.py --protocol cross_domain --backbones mobilenet_v2  --epochs 15 --finetune_head

python benchmark_backbones.py --protocol cross_domain --backbones inception_v3  --epochs 15
python benchmark_backbones.py --protocol cross_domain --backbones inception_v3  --epochs 15 --finetune_head
```
**Output:** for each backbone, `runs/<backbone>/cross_domain_aug-strong/` (frozen) vs. `runs/<backbone>/cross_domain_aug-strong_ft/` (finetuned).

Compare: `material_acc`, `material_macro_f1`, and `history.train_time_sec`
(finetuning trains more parameters, so expect it to be slower per epoch —
check whether the accuracy gain is worth the extra training time and
`inference_latency_ms_per_image`).

---

## 4. WITH vs WITHOUT augmentation strength (`--aug_mode`)

`auto` already picks the sensible default per protocol — these runs
override it explicitly to measure the effect directly.

```bash
# cross_domain: off vs mild vs strong (auto default)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode off
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode mild
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode strong

# in_domain: mild (auto default) vs strong vs off — does more aug cost accuracy with no domain gap to fight?
python benchmark_backbones.py --protocol in_domain --backbones resnet50 --epochs 15 --aug_mode off
python benchmark_backbones.py --protocol in_domain --backbones resnet50 --epochs 15 --aug_mode mild
python benchmark_backbones.py --protocol in_domain --backbones resnet50 --epochs 15 --aug_mode strong
```
**Output:** `runs/resnet50/cross_domain_aug-off/`, `.../cross_domain_aug-mild/`, `.../cross_domain_aug-strong/` (and the `in_domain_aug-*` equivalents) — all three variants sit side by side under `runs/resnet50/`, nothing overwritten.

Compare: for cross_domain, expect off < mild < strong on `material_acc`
(more domain-randomization closes more of the gap). For in_domain, this
repo's own results have already shown `off` beating `mild`/`strong` — see
the README's design-rationale sections for why that's the expected,
not surprising, outcome when train and test share one domain.

---

## 5. WITH vs WITHOUT early stopping (`--patience`) + the new LR scheduler

```bash
# WITHOUT patience: fixed epoch count (whatever epoch it ends on is final)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15

# WITH: large epoch budget + early stopping, keeps best-val-loss weights, cosine LR decay (default)
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 100 --patience 5

# same, but with the OLD flat learning rate (no decay) for comparison
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 100 --patience 5 --lr_schedule none
```
**Output:** `runs/resnet50/cross_domain_aug-strong/` (15-epoch baseline),
`runs/resnet50/cross_domain_aug-strong/` gets overwritten by the 100-epoch
run since they share a tag — **use `--aug_mode` or add a manual `--out`
suffix if you want to keep both**; the `_lr-none` suffix already keeps the
flat-LR variant separate: `runs/resnet50/cross_domain_aug-strong_lr-none/`.

Compare: check `results.json`'s `history.best_epoch` / `history.best_val_loss`
against the fixed-epoch run's final-epoch numbers, and check the per-epoch
`lr=` value printed in the log to confirm cosine decay is active (it should
shrink smoothly from `--lr`, default 3e-4, toward ~0 by the final epoch).

---

## 6. Ablations from RUNBOOK.md Phase 3 (background removal, DANN, material-rule)

```bash
# (a) augmentation off — quantifies how much domain-randomizing aug closes the gap
python benchmark_backbones.py --protocol cross_domain --backbones resnet50 --epochs 15 --aug_mode off

# (b) class-weighted (long-tail) joint
python benchmark_backbones.py --protocol joint --backbones resnet50 --epochs 15 --class_weighted

# (c) background-removal ablation
python gradcam_ablation.py generate
# then point a benchmark run at manifest_masked.csv (train on clean, eval on masked wild):
python benchmark_backbones.py --manifest manifest_masked.csv --protocol cross_domain --backbones resnet50 --epochs 15

# (d) DANN and material-rule-vs-binary-head (waste_pipeline.py, not benchmark_backbones.py)
python waste_pipeline.py --taco_images Taco_data --taco_ann Taco_data/annotations.json --trashnet trashnet_dataset-resized --protocol cross_domain --dann
python waste_pipeline.py --taco_images Taco_data --taco_ann Taco_data/annotations.json --trashnet trashnet_dataset-resized --protocol cross_domain --material_rule

# Grad-CAM money figure (needs a clean-trained resnet50 checkpoint from step 1 or 3 above)
python gradcam_ablation.py gradcam --ckpt runs/resnet50/in_domain_aug-mild/checkpoint.pt --n 8
```
**Output:** (a) → `runs/resnet50/cross_domain_aug-off/`; (b) → `runs/resnet50/joint_aug-strong_cw/`; (c) → `runs/resnet50/cross_domain_aug-strong/` (from `manifest_masked.csv`, so use a distinct `--manifest`-derived comparison, not overwrite (a)'s clean-manifest run — pass `--out`/`--runs_dir` explicitly here if you need both kept); (d) → `waste_pipeline.py`'s own output files (see that script's `--out`/`--help`); Grad-CAM → whatever `gradcam_ablation.py` names its figure output (check that script directly).

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

All of these land in distinct auto-named folders under `runs/<backbone>/` —
no manual bookkeeping needed even for the full 6-way grid per backbone.

---

## 8. Assemble comparison tables after all runs

```bash
python aggregate_results.py \
  --in_domain runs/_summaries/in_domain_aug-mild.json \
  --cross     runs/_summaries/cross_domain_aug-strong.json \
  --joint     runs/_summaries/joint_aug-strong.json \
  --ablation no_aug=runs/_summaries/cross_domain_aug-off.json
```
**Output:** whatever `aggregate_results.py`'s own `--out`/default naming produces (Tables T3–T5 + the domain-gap figure — check that script for its exact output filenames).

For the WITH/WITHOUT comparisons in sections 2–5 (not natively covered by
`aggregate_results.py`'s table format), compare each backbone's
`results.json` directly, e.g.:

```python
import json
a = json.load(open("runs/resnet50/cross_domain_aug-strong/results.json"))["results"]
b = json.load(open("runs/resnet50/cross_domain_aug-strong_ft/results.json"))["results"]
print("frozen   :", a["material_acc"], a["material_macro_f1"], a["non_recyclable_recall"])
print("finetuned:", b["material_acc"], b["material_macro_f1"], b["non_recyclable_recall"])
print("train_time frozen/finetuned (sec):", a["history"]["train_time_sec"], b["history"]["train_time_sec"])
```

---

## 9. Checkpoints — saving and running inference

Every training run saves one checkpoint at
`runs/<backbone>/<tag>/checkpoint.pt` (best-epoch weights, restored via
`--patience`'s best-state tracking when that flag is used). Pass `--no_ckpt`
to skip saving if disk space is tight during a big sweep.

Each `.pt` file bundles everything `predict.py` needs to run standalone:
backbone name, `finetune_head` flag, input size, the `MATERIALS` list, the
`MATERIAL_RECYCLABLE` rule, and the run's final metrics (`material_acc`,
`recyclability_acc`, `non_recyclable_recall`, `best_epoch`) alongside the
model weights.

**Run inference on one or more images:**
```bash
python predict.py --ckpt runs/resnet50/cross_domain_aug-strong/checkpoint.pt --image path/to/photo.jpg
```
**Output (printed to console):**
```
Loaded resnet50 (cross_domain, aug_mode=strong, finetune_head=False) --
material_acc=0.XXX, recyclability_acc=0.XXX, non_recyclable_recall=0.XXX

path/to/photo.jpg
  material   : plastic  (confidence 87.3%)
  state      : RECYCLABLE
```

**Multiple images + full per-material probability breakdown:**
```bash
python predict.py --ckpt runs/resnet50/cross_domain_aug-strong/checkpoint.pt \
    --image img1.jpg img2.jpg img3.jpg --show_probs
```
**Output:** the same block as above repeated per image, plus (with `--show_probs`) all 6 materials' probabilities sorted highest-first under each image.

**Compare checkpoints from different runs on the same image** (e.g. frozen
vs. finetuned, or across backbones) by just swapping `--ckpt`:
```bash
python predict.py --ckpt runs/resnet50/cross_domain_aug-strong/checkpoint.pt     --image sample.jpg
python predict.py --ckpt runs/resnet50/cross_domain_aug-strong_ft/checkpoint.pt  --image sample.jpg
```

The report PNG's bottom-right sample-prediction grid already shows this
same recyclable/non-recyclable state inline with both the true and
predicted material label on every tile, spread across all 6 material
classes — so `predict.py` is for running the model on **new** images
outside the manifest, not for eyeballing test-set examples.

---

## Naming convention (automatic — nothing to remember)

`runs/<backbone>/<protocol>_aug-<mode>[_ft][_cw][_lr-none]/` — every distinct
configuration is kept separate automatically. If you need a manual override
for any single run (e.g. to preserve two runs that would otherwise share a
tag, like the epochs=15-vs-100 case in section 5), pass an explicit
`--out`/`--report_dir`/`--ckpt_dir`/`--log_file`.
