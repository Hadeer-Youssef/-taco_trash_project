# Phase 1 (P1, in-domain) — Material Classification on TrashNet: Method, Experiments and Results

Status: complete. Scope: protocol **P1 (in_domain)** of the harmonized TACO + TrashNet
benchmark — training and testing on the clean (TrashNet) domain only. Protocols P2
(cross-domain) and P3 (joint) are reported separately.

All numbers in this document come from the run logs / `results.json` files produced by
`benchmark_backbones.py` (legacy runs) and `run_benchmark.py` + `ensemble_eval.py`
(`waste_bench` package). Where a number comes from a different evaluation path
(e.g. fp32 ensemble evaluation vs. AMP training-time evaluation) this is stated.

---

## 1. Summary

**Goal.** Predict the material of a waste item (6 classes) from a photograph, and from
the predicted material derive whether the item is recyclable or not.

**Main findings**

1. A frozen ImageNet backbone with a trained head plateaus at **88.3 %** test accuracy.
   Fine-tuning only the last two resolution stages (leaving the generic early layers
   frozen) raises ResNet-50 to **93.3 %**; label smoothing, a lower backbone learning
   rate and flip test-time augmentation bring it to **94.6 %**.
2. With the same recipe, the architecture matters: **ConvNeXt-Tiny reaches
   97.7 % ± 0.7** (3 seeds), against 93.8–94.6 % for ResNet-50, DenseNet-121,
   Inception-v3, MobileNet-V2 and EfficientNetV2-S.
3. An ensemble of the three ConvNeXt-Tiny seeds reaches **98.8 %** (3 errors out of 257)
   with every non-recyclable item detected (15/15). Mixing in the weaker backbones
   *lowers* the ensemble (96.5 %), because plain averaging lets the weaker members
   outvote the best one.
4. Several techniques that look promising did **not** help on this dataset:
   domain-randomizing / geometric augmentation, a paper-derived ECOC output head,
   stronger label smoothing, fully unfreezing the backbone, and a higher learning rate
   for the small MobileNet-V2.
5. The test set is small (257 images; one image = 0.39 % accuracy and the 15
   non-recyclable images make one image = 6.7 % recall). Differences below ~1.5 % are
   within noise; Section 9 lists the validity threats.

**Headline table** (test split, material accuracy; mean ± std over 3 seeds unless stated)

| System | Test acc. | Macro-F1 | Non-recyclable recall | 95 % Wilson CI (acc.) |
|---|---|---|---|---|
| ResNet-50, frozen backbone + custom head | 0.883 ± 0.012 | 0.867 | 0.756 | [0.838, 0.917] |
| ResNet-50, last two stages fine-tuned | 0.933 ± 0.004 | 0.922 | 0.844 | [0.897, 0.958] |
| ResNet-50, final recipe (+ TTA) | 0.946 ± 0.004 | 0.937 | 0.911 | [0.911, 0.967] |
| **ConvNeXt-Tiny, final recipe (+ TTA)** | **0.977 ± 0.007** | **0.974** | **0.978** | [0.950, 0.989] |
| **ConvNeXt-Tiny, 3-seed ensemble** | **0.988** (single value) | 0.990 | 1.000 | [0.966, 0.996] |

The intervals of the last two rows overlap with those of the ResNet-50 rows; the
ordering is supported by seed consistency (every ConvNeXt seed ≥ 0.973, every ResNet-50
seed ≤ 0.949) and by the validation split (+0.8 %), but one split of 257 images cannot
establish it beyond doubt (Section 9).

---

## 2. Task definition

* **Primary task:** 6-way material classification:
  `cardboard, glass, metal, paper, plastic, trash`.
* **Derived task:** binary recyclability. The model never predicts recyclability
  directly. It is derived from the predicted material by a fixed rule:
  cardboard, glass, metal, paper, plastic → recyclable; trash → non-recyclable.
  The ground-truth recyclability label comes from the dataset's manual labelling
  policy (`label_mapping.csv`).
* **Why material first:** an explicit material prediction makes every recyclability
  error attributable to a specific material mistake, and allows both material-level
  and recyclability-level metrics from one model.

---

## 3. Data

Source: **TrashNet** (Yang & Thung), 2,527 studio photographs on a white background,
512 × 384, single object per image. In the harmonized benchmark this is the *clean*
domain (`domain == clean` in `manifest.csv`). Splits are fixed in the manifest's
`split` column (≈ 80 / 10 / 10 within every class).

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| cardboard | 322 | 40 | 41 | 403 |
| glass | 400 | 50 | 51 | 501 |
| metal | 328 | 41 | 41 | 410 |
| paper | 475 | 59 | 60 | 594 |
| plastic | 385 | 48 | 49 | 482 |
| trash | 109 | 13 | 15 | 137 |
| **Total** | **2,019** | **251** | **257** | **2,527** |

Properties relevant to interpretation:

* **Class imbalance:** `trash` is ~3.5× rarer than the other classes and is the only
  non-recyclable class. Inverse-frequency class weights are therefore used
  (cardboard 1.045, glass 0.841, metal 1.026, paper 0.708, plastic 0.874, trash 3.087).
* **Recyclability is almost constant:** 94.6 % of TrashNet images are recyclable, so
  an "always recyclable" classifier scores **0.942** recyclability accuracy on the
  test split. Recyclability accuracy is therefore reported only alongside this
  baseline, and **non-recyclable recall** (15 test images) is used as the
  operationally meaningful recyclability metric.

---

## 4. Evaluation protocol

### 4.1 Metrics

| Metric | Meaning |
|---|---|
| `material_acc` | 6-class accuracy on the test split (headline) |
| `material_macro_f1` | macro-averaged F1 over the 6 classes (penalizes ignoring `trash`) |
| `recyclability_acc` | accuracy of the rule-derived recyclable / non-recyclable label |
| `non_recyclable_recall` | fraction of truly non-recyclable test items detected |
| `val_acc` | accuracy on the validation split (the model-selection split) |

Statistical resolution of the test split: **1 image = 0.39 %** material accuracy and
**6.7 %** non-recyclable recall. Example: recall 0.733 vs 0.867 is 11 vs 13 images out
of 15.

### 4.2 Model selection

* **Early stopping and best-epoch restoration use the validation split** (251 images)
  for in-domain runs; the test split is used only for the reported number.
* The legacy runs (Stage 0) selected the best epoch on the **test** split. This was
  identified as a methodological flaw and corrected (Stage 1); legacy numbers are
  listed separately and marked.
* Each configuration of the final recipe was repeated with **3 seeds** (42, 1, 2).
  Seeds control weight initialisation of the new head, data order and augmentation.

### 4.3 Reporting conventions

* `±` denotes the sample standard deviation over the seeds (n = 3).
* Confidence intervals are approximate 95 % Wilson intervals computed from the
  correct-count out of 257 (they ignore seed variance and treat the split as the only
  source of uncertainty).
* Per-run numbers are the training-time evaluation (mixed precision, `results.json`).
  `ensemble_eval.py` re-evaluates checkpoints in fp32, which changed one image for
  one ResNet-50 run (0.949 → 0.953).

---

## 5. Methods

### 5.1 Transfer learning setup

All backbones are initialised from **ImageNet-1k pretrained weights** (torchvision) and
receive a newly initialised classifier.

| Backbone | Year | Input | Weights |
|---|---|---|---|
| ResNet-50 | 2015 | 224 | IMAGENET1K_V2 |
| DenseNet-121 | 2016 | 224 | IMAGENET1K_V1 |
| MobileNet-V2 | 2018 | 224 | IMAGENET1K_V1 |
| Inception-v3 | 2015 | 299 | IMAGENET1K_V1 (aux. head trained, weight 0.4) |
| EfficientNetV2-S | 2021 | 224 | IMAGENET1K_V1 |
| ConvNeXt-Tiny | 2022 | 224 | IMAGENET1K_V1 |

EfficientNetV2-S was trained on larger resolutions (300–384); 224 was used so that all
backbones see comparable input and the comparison isolates the architecture.

### 5.2 Classification head

* **Linear** head (baseline) or a **custom MLP head** (final recipe): for every hidden
  width `h`, `Linear → BatchNorm → ReLU → Dropout(0.3)`, then a final `Linear → 6`.
  Final recipe: `in → 512 → 256 → 6`.
* ConvNeXt keeps its pretrained `LayerNorm2d + Flatten` in front of the new head.

### 5.3 Partial fine-tuning (the core of the strategy)

Instead of choosing between "frozen" and "fully fine-tuned", the depth of adaptation is
a controlled variable (`--unfreeze`). Early layers (edges, colours, textures) stay
frozen and keep the ImageNet representation; only the last stages adapt.

| Level | ResNet-50 | ConvNeXt-Tiny |
|---|---|---|
| `none` | head only | head only |
| `last` | last bottleneck's conv1 + bn1 | last block of stage 4 |
| `stage` | `layer4` | stage 4 |
| `two_stages` | `layer3` + `layer4` | stages 3 + 4 (with their down-sampling) |
| `all` | every layer | every layer |

(Equivalent definitions exist for DenseNet-121, MobileNet-V2, Inception-v3 and
EfficientNetV2-S; see `waste_bench/models/backbones.py`.)

Note that `two_stages` already trains ≈ 94 % of ResNet-50's parameters (the late stages
hold most weights) while keeping the first half of the network's *depth* frozen; the
parameter count is therefore not a good measure of "how much was frozen".

### 5.4 Optimisation and regularisation

| Component | Setting |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | head 3e-4, cosine-annealed to ~0 over the epoch budget |
| Backbone learning rate | `lr × backbone_lr_mult` (0.1 / 0.3), same schedule |
| Batch size / epochs | 32 / 30, early stopping with patience 8 on validation loss |
| Loss | cross-entropy with inverse-frequency class weights |
| Label smoothing | 0.1 (final recipe) |
| Augmentation | `off`: RandomResizedCrop(scale 0.6–1) + horizontal flip; `mild`: + small rotation (±30°), vertical flip, translate/shear; `strong`: + colour jitter, grayscale, blur |
| Precision | mixed precision (AMP) |
| Test-time augmentation | average of softmax(image) and softmax(horizontally flipped image) |

Final recipe (used for all backbone comparisons):
`two_stages`, head 512→256, lr 3e-4 (cosine), backbone_lr_mult 0.3, label smoothing
0.1, class weights, augmentation off, TTA on.

### 5.5 Ensembling

Probabilities (after TTA) of the selected checkpoints are averaged with equal weights;
the class is the arg-max. `ensemble_eval.py` evaluates every individual model, one
seed-ensemble per backbone, and the ensemble of all selected checkpoints on both the
validation and test splits, and lists the test images that are misclassified by every
single model ("hard core").

### 5.6 Implementation

The experiments use the `waste_bench` package (config → data → models → engine →
reporting) with deterministic seeding, per-epoch validation, automatic run folders
(`runs/<backbone>/<config tag>/`), checkpoints that carry their full configuration, and
an experiment-comparison tool (`compare_runs.py`). The earlier script
`benchmark_backbones.py` is kept unchanged.

---

## 6. Experimental log

Experiments are presented in the order they were carried out. Each stage lists what
was changed, the result, and the decision taken.

### Stage 0 — Legacy baseline and a paper-derived component (selection on test)

Setup: ResNet-50, AdamW, class weights, up to 100 epochs with patience 5; the
"fine-tuned head" runs use an MLP head (2048→96→6) with the last bottleneck unfrozen.
**Best epoch selected on the test split** — these numbers are optimistic. The exact
learning-rate settings of the individual legacy runs were not recorded; the table is
indicative only.

| Run | Aug. | Test acc. | Macro-F1 | Non-rec. recall |
|---|---|---|---|---|
| Frozen backbone, linear head | mild | 0.833 | 0.812 | 0.800 |
| Fine-tuned head | mild | 0.887 | 0.876 | 0.733 |
| Fine-tuned head | off | 0.914 | 0.900 | 0.933 |
| Fine-tuned head, further tuned | off | 0.922 | 0.915 | 0.867 |

Observation: with a mostly frozen backbone, augmentation *reduced* accuracy
(0.887 vs 0.914), consistent with the hypothesis that in-domain training has no domain
gap for distortion-based augmentation to close.

**ECOC head** (Wang 2024, Heliyon 10 e29966; reported +3.68 % on TrashNet with a
from-scratch CNN): an ordinal error-correcting-output-code head (5 binary classifiers,
Hamming decoding) replacing the softmax head on the same ResNet-50:

Both runs: fine-tuned head, class weights, no augmentation, lr 1e-4 (flat), batch 32,
patience 5, selection on test.

| Head | Test acc. | Macro-F1 | Non-rec. recall |
|---|---|---|---|
| Softmax (same configuration without ECOC) | 0.922 | 0.904 | 0.800 |
| Ordinal ECOC | 0.872 | 0.858 | 0.667 |

Decision: **ECOC dropped.** Its BCE validation loss was numerically lower than the
softmax cross-entropy (0.15 vs 0.27) while accuracy was 5 % lower — the loss measures
per-bit confidence, not decoded-class accuracy, and the ordinal codebook assumes an
order among classes that materials do not have. A symmetric one-vs-rest codebook was
implemented but **not benchmarked** (out of scope after this result).

### Stage 1 — Methodological corrections

| Issue found | Correction |
|---|---|
| Epoch selection on the test split | Select on the held-out validation split (251 images); test used only for the final number |
| No measure of run-to-run noise | 3 seeds per configuration; mean ± std reported |
| LR schedule cut off by early stopping (lr still ≈ 1e-3 when stopping) | Shorter budget (30 epochs) so cosine annealing actually completes |
| Sample-grid in reports showed one class (manifest is class-sorted) | Class-balanced sample collection |
| Reproducibility | Config stored in every `results.json` and checkpoint |

Sanity check: re-running the legacy recipe in the new code (selection on test)
gave **0.911** (val 0.916), within the later-measured seed noise of the legacy
number (0.914–0.922).

### Stage 2 — Frozen backbone with a custom head (noise baseline)

ResNet-50, head 512→256, `unfreeze=none`, lr 1e-3, 60 epochs, patience 10, class
weights, no augmentation, selection on val.

| Seed | Test acc. | Macro-F1 | Non-rec. recall | Val acc. | Best epoch |
|---|---|---|---|---|---|
| 42 | 0.872 | 0.854 | 0.733 | 0.908 | 3 |
| 1 | 0.895 | 0.877 | 0.800 | 0.908 | 4 |
| 2 | 0.883 | 0.871 | 0.733 | 0.928 | 20 |
| **mean ± std** | **0.883 ± 0.012** | 0.867 | 0.756 | 0.915 | 9 |

Findings: (i) run-to-run noise is ≈ ±1.2 %, i.e. 3 images; (ii) train accuracy
reaches 97 % within 13 epochs while validation stays ≈ 90 % — the frozen ImageNet
features are not discriminative enough for the material distinctions (overfitting of
the head, not of the backbone); (iii) the selected epoch varies from 3 to 20 because
the learning rate had not decayed.

### Stage 3 — Depth of fine-tuning (ResNet-50)

Common: head 512→256, lr 3e-4 cosine, backbone_lr_mult 0.1, 30 epochs, patience 8,
class weights, no augmentation.

| `--unfreeze` | Trainable params | Test acc. | Macro-F1 | Non-rec. recall | Val acc. | Best ep. | Time |
|---|---|---|---|---|---|---|---|
| `none` (Stage 2) | 1.18 M | 0.883 ± 0.012 | 0.867 | 0.756 | 0.915 | 9 | 2.4 min |
| `last` | 2.23 M | 0.918 | 0.899 | 0.733 | 0.916 | 16 | 2.9 min |
| `stage` | 16.1 M | 0.918 | 0.910 | 0.867 | 0.932 | 8 | 2.0 min |
| `two_stages` (3 seeds) | 23.2 M | **0.933 ± 0.004** | 0.922 | 0.844 | 0.955 | 16 | 2.8 min |
| `all` | 24.7 M | 0.938 | 0.928 | 0.867 | 0.964 | 18 | 4.4 min |

Findings: accuracy increases monotonically with depth (validation: 0.915 → 0.916 →
0.932 → 0.955); the gain from `none` to `two_stages` (+5.0 %) is ≈ 4× the measured
noise. `all` is within noise of `two_stages` at ~1.6× the cost. Seed variance also
dropped (±0.004 vs ±0.012), i.e. adapting the late stages made training more stable.
Decision: **`two_stages`**.

### Stage 4 — Recipe ablations (ResNet-50, `two_stages`, seed 42 unless stated)

| Change vs. Stage-3 `two_stages` (acc. 0.930, val 0.952 at seed 42) | Test acc. | Macro-F1 | Non-rec. recall | Val acc. | Verdict |
|---|---|---|---|---|---|
| `aug_mode = mild` | 0.934 | 0.921 | 0.867 | **0.904** | rejected (val −4.8 %) |
| `label_smoothing = 0.1` | 0.946 | 0.943 | 1.000 | 0.964 | **kept** (test and val improve) |
| `backbone_lr_mult = 0.3` | 0.949 | 0.938 | 0.867 | 0.952 | kept (test only; val unchanged) |
| `label_smoothing = 0.1 + backbone_lr_mult = 0.3` (3 seeds) | 0.944 ± 0.008 | 0.934 | 0.867 | 0.952 | **final recipe** |
| `label_smoothing = 0.2 + backbone_lr_mult = 0.3` | 0.949 | 0.941 | 0.933 | 0.936 | rejected (val lower; best epoch 5) |

Findings: (i) once most of the network trains, the model overfits (train acc. 99 %
by epoch 8–10), so regularisation matters — label smoothing is the only change with
agreeing validation and test gains; (ii) augmentation still does not help in-domain
(validation falls by 12 images); (iii) combining the two accepted changes gave a
+1.1 % mean gain over Stage 3, borderline relative to the ±0.8 % seed std, with no
validation gain — a small effect.

### Stage 5 — Test-time augmentation

Same trained models, evaluation averaged over image and horizontal flip.

| | Test acc. | Macro-F1 | Non-rec. recall |
|---|---|---|---|
| ResNet-50 seed 42, no TTA | 0.938 | 0.928 | 0.867 |
| ResNet-50 seed 42, TTA | 0.949 | 0.947 | 0.933 |

TTA is a paired comparison on identical weights: **+1.1 % (3 images)** at 1.7× the
inference time. Decision: kept. ResNet-50 with the final recipe and TTA, 3 seeds:
**0.946 ± 0.004** (0.949 / 0.942 / 0.946; val 0.960).

### Stage 6 — Backbone comparison (final recipe, TTA)

| Backbone | Seeds | Test acc. | Macro-F1 | Non-rec. recall | Val acc. | Train time | Inference |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 3 | 0.946 ± 0.004 | 0.937 | 0.911 | 0.960 | 2–3 min | 3.9 ms |
| DenseNet-121 | 3 | 0.941 ± 0.002 | 0.926 | 0.822 | 0.955 | 2–3 min | 4.5 ms |
| Inception-v3 (299 px) | 3 | 0.943 ± 0.010 | 0.929 | 0.889 | 0.956 | 2.5–5 min | 4.7 ms |
| MobileNet-V2 | 1 | 0.938 | 0.926 | 0.867 | 0.948 | 3 min | 18 ms\* |
| EfficientNetV2-S | 1 | 0.942 | 0.929 | 1.000 | 0.960 | 2.9 min | 14 ms\* |
| **ConvNeXt-Tiny** | 3 | **0.977 ± 0.007** | **0.974** | **0.978** | 0.964 | 2.6–3.5 min | 15.8 ms\* |

Inference time is per image with TTA (two forward passes), batch 32, mixed precision.
\* Backbones built from depthwise convolutions are ≈ 3–5× slower at inference than
the others in this setup (GPU utilisation 9–35 %); the cause was not isolated and the
figures should be re-measured before being quoted (Section 9).

Depth sweeps of the two new backbones (seed 42):

| Backbone | `last` | `stage` | `two_stages` | `all` |
|---|---|---|---|---|
| ConvNeXt-Tiny | 0.938 | 0.957 | **0.973** (0.977 ± 0.007, 3 seeds) | 0.977 |
| EfficientNetV2-S | 0.872 | 0.942 | 0.942 | – |

Findings:

* The four older backbones are statistically indistinguishable (0.938–0.946); the
  plateau at ≈ 94 % is architecture-independent.
* ConvNeXt-Tiny is the only clear exception: +3.1 % over ResNet-50 on test but only
  +0.4–0.9 % on validation. Pooling validation and test (508 images, per-model mean
  error counts) gives ≈ 97.0 % vs ≈ 94.8–95.3 % for the older backbones, i.e. ≈ +2 %
  — the test-only +3 % partly reflects the particular split.
* The depth trend of ResNet-50 repeats for ConvNeXt (0.938 → 0.957 → 0.973); `all`
  adds nothing.
* MobileNet-V2 at lr 1e-3 / 50 epochs was worse and unstable (0.926, val 0.936);
  it had not converged at 30 epochs at lr 3e-4 (best epoch 30/30), which suggests
  it is under-trained rather than capacity-limited, but this was not pursued.

### Stage 7 — Ensembles and error analysis

Evaluated with `ensemble_eval.py` (fp32, TTA, equal-weight averaging). Ensembles
were defined before looking at results (per-backbone seed ensembles and "all
checkpoints"); no subset search was performed.

| Ensemble | Models | Val acc. | Test acc. | Macro-F1 | Non-rec. recall | Test errors |
|---|---|---|---|---|---|---|
| ResNet-50 seeds | 3 | 0.956 | 0.957 | 0.948 | 0.867 | 11 |
| DenseNet-121 seeds | 3 | 0.964 | 0.942 | 0.930 | 0.867 | 15 |
| Inception-v3 seeds | 3 | 0.960 | 0.957 | 0.941 | 0.867 | 11 |
| All ResNet/DenseNet/Inception (3 × 3) | 9 | 0.972 | 0.969 | 0.961 | 0.933 | 8 |
| ConvNeXt-Tiny seeds | 3 | 0.964 | **0.988** | **0.990** | **1.000** | **3** |
| All of the above + ConvNeXt | 12 | 0.972 | 0.965 | 0.958 | 0.933 | 9 |

Hard core (test images misclassified by *every* member):

| Ensemble | Hard-core images | True → predicted |
|---|---|---|
| 9 older models | 2 | cardboard → metal, plastic → metal |
| ConvNeXt seeds (3) | 3 | cardboard → metal, paper → plastic, plastic → glass |
| 12 models | 1 | cardboard → metal |

Findings:

* The earlier hypothesis that the ≈ 94 % plateau is a ceiling imposed by ambiguous
  labels is **refuted**: only 1–3 images are wrong in every model; 50 images are
  wrong in at least one of 12 models. Most errors are model-specific variance.
* Ensembling same-architecture seeds helps (ResNet-50: 0.946 → 0.957; ConvNeXt:
  0.977 → 0.988), but on the validation split the ConvNeXt seed-ensemble does not
  improve on a single model (0.964 vs 0.964). The test gain (+1.1 %, 3 images) is
  small and may be partly split-specific.
* Mixing a strong backbone with weaker ones is harmful under equal weighting
  (0.988 → 0.965): the three ConvNeXt models hold only 25 % of the votes.
* None of the three remaining ConvNeXt-ensemble errors is a missed non-recyclable
  item (non-recyclable recall 15/15).

---

## 7. Consolidated improvement path (ResNet-50 → ConvNeXt-Tiny)

| Step | Change | Test acc. | Δ | Evidence strength |
|---|---|---|---|---|
| 0 | Frozen backbone + custom head | 0.883 ± 0.012 | – | 3 seeds |
| 1 | Fine-tune last two stages (lr ×0.1 on backbone) | 0.933 ± 0.004 | +5.0 | 3 seeds; ≈ 4× noise |
| 2 | + label smoothing 0.1, backbone lr ×0.3 | 0.944 ± 0.008 | +1.1 | 3 seeds; borderline, val flat |
| 3 | + flip TTA | 0.946 ± 0.004 | +0.2 | paired on seed 42: +1.1 |
| 4 | ResNet-50 → ConvNeXt-Tiny (same recipe) | 0.977 ± 0.007 | +3.1 | 3 seeds; val +0.8 |
| 5 | 3-seed ensemble | 0.988 | +1.1 | single value; val unchanged |

The two large steps are *adapting the late stages of a pretrained network* (step 1)
and *a stronger pretrained architecture* (step 4). The rest are small.

---

## 8. Discussion

**Why fine-tuning depth mattered.** With a frozen backbone the head memorised the
training set while validation stalled at ≈ 90 %, which indicates that generic ImageNet
features separate objects but not the material cues (texture, reflectance) needed
here. Letting the last two stages adapt, with a reduced backbone learning rate to
avoid destroying the pretrained features, removed most of this gap while the first
half of the network stayed frozen.

**Why augmentation did not help.** In-domain train and test images are drawn from
the same studio distribution; geometric and photometric distortions add variance
without addressing any shift. Augmentation is expected to matter in P2/P3, where the
test domain differs.

**Why ECOC failed here.** The reported gain of the source paper was obtained with a
small CNN trained from scratch; on a strong pretrained backbone the softmax head is
already adequate, and the ordinal code imposes an arbitrary class order on unordered
materials.

**Why the newer backbone helped.** ConvNeXt-Tiny's larger-kernel depthwise design and
LayerNorm may give features that transfer better to fine material texture, but the
experiments here do not isolate the cause; EfficientNetV2-S, also recent, did not
help at 224 px.

**Relation to the literature.** Wang (2024) reports 98.81 % on TrashNet with 10-fold
cross-validation and a CNN trained from scratch; the earlier baselines it cites reach
94–97 %. The present 97.7–98.8 % is on a single fixed 257-image test split and is
**not directly comparable**; a k-fold evaluation over all 2,527 images would be
needed for a like-for-like comparison.

---

## 9. Threats to validity and limitations

1. **Single small test split.** 257 images; the 95 % intervals of the best
   configurations overlap with those of ResNet-50 (Section 1). Seed consistency and
   the validation split support the ranking but are not a significance test.
2. **Test-set reuse.** The test split was inspected throughout development (≈ 40
   training runs). The recipe was chosen mainly on validation agreement, but the
   backbone, depth and ensemble choices were also informed by test results, so the
   best numbers carry a small optimistic bias (we estimate ≲ 1 %).
3. **Validation optimism.** The validation split picks the best epoch, so
   `val_acc` is itself slightly optimistic, equally for all models.
4. **Post-hoc ensemble choice.** The ConvNeXt-only ensemble was singled out after
   seeing that the 12-model ensemble was worse on test; on validation the 12-model
   ensemble was nominally better (0.972 vs 0.964; 2 images).
5. **Only 3 seeds** (1 for MobileNet-V2 and EfficientNetV2-S).
6. **Hyper-parameter coverage.** Learning rate, batch size, head width and
   dropout were only coarsely explored; backbones may favour different settings
   (MobileNet-V2 did not converge at the shared lr).
7. **Recyclability labels** follow a single, facility-agnostic policy, and in
   TrashNet recyclability is almost equivalent to "class ≠ trash"; only 15 test
   items are non-recyclable, giving wide recall intervals (e.g. 14/15 =
   0.933, 95 % CI [0.70, 0.99]).
8. **Single domain.** P1 does not test robustness to cluttered real-world images,
   which is the subject of P2/P3.
9. **Latency numbers** were measured with a short warm-up and without isolating
   data loading; those of depthwise-convolution backbones should be re-measured
   before being reported.

---

## 10. Reproducibility

Environment: single Kaggle GPU, PyTorch/torchvision pretrained weights, AMP, 4 data
workers. Typical cost: 2–5 minutes per run.

Final recipe (one run):

```bash
python run_benchmark.py --protocol in_domain --backbones convnext_tiny \
  --epochs 30 --patience 8 --class_weighted --aug_mode off \
  --head_hidden 512 256 --lr 3e-4 --unfreeze two_stages \
  --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --amp --workers 4 --seed 42
```

Seeds 1 and 2 add `--seed 1` / `--seed 2`. Ensemble and error analysis:

```bash
python ensemble_eval.py --protocol in_domain --match two_stages --match ls0.1 \
  --match tta --backbones convnext_tiny --tta --workers 4 --out_dir ens_convnext
```

Comparison table of every run: `python compare_runs.py --protocol in_domain --csv p1_all.csv`.

Artifacts per run (`runs/<backbone>/<tag>/`): `report_<backbone>.png`
(loss/accuracy curves, per-class report, confusion matrix, ROC, class-balanced sample
grid), `checkpoint.pt` (weights + configuration), `results.json` (metrics, history,
timing, config). Summaries and logs: `runs/_summaries/`.

Source files: `waste_bench/` (package), `run_benchmark.py`, `ensemble_eval.py`,
`compare_runs.py`, `run_predict.py`; legacy: `benchmark_backbones.py`, `predict.py`.

---

## 11. Next steps (outside P1)

1. P3 (joint) and P2 (cross-domain) with the frozen P1 recipe, comparing ConvNeXt-Tiny
   and ResNet-50 — to test whether the in-domain ranking and the optimal fine-tuning
   depth persist under domain shift (deeper fine-tuning may hurt there).
2. k-fold cross-validation over the clean domain for a number comparable with the
   literature.
3. Re-measure inference latency of the depthwise-convolution backbones
   (larger warm-up, channels-last, fp32 vs fp16).
4. Weighted or val-selected ensembles instead of equal averaging, if a small
   deployable ensemble is required.
