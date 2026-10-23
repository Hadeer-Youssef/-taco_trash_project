# Phase 2 (P2 cross-domain, P3 joint) — Recyclability from Clean and Wild Images: Dual-Head Method, Experiments and Results

Status: complete for the planned experiments. Scope: protocols **P2 (cross_domain)** and
**P3 (joint)** of the harmonized TACO + TrashNet benchmark, plus the dual
(material + recyclability) head introduced for them. Protocol P1 (in-domain) is
reported in `PHASE1_REPORT.md`.

All numbers in this document come from the run logs / `results.json` files written by
`run_dual.py` (`waste_bench/cli_dual.py`), on a Kaggle GPU, with ConvNeXt-Tiny and the
Phase 1 recipe. Numbers marked *derived* are computed from other logged numbers
(stated where used). Standard deviations are population standard deviations over seeds
(`numpy.std`, ddof = 0) over **3 seeds** (42, 1, 2) unless a row says "1 seed".

---

## 1. Summary

**Goal.** Decide whether a photographed waste item is recyclable. Phase 1 predicted the
material (6 classes) and derived recyclability by a fixed rule (everything except `trash`
is recyclable). Phase 2 tests that on realistic images and replaces the rule with a
directly trained recyclability output.

**Main findings**

1. **The material → recyclability rule does not hold on wild (TACO) images.** In wild,
   recyclability is an item-level property, not a material-level one: 73.9 % of wild
   `plastic` items are non-recyclable (bags, wrappers, …). Even with perfect material
   labels the rule scores 0.568 accuracy / 0.343 non-recyclable recall on the wild
   test split (Section 3.3). It is a rule that Phase 1's data (TrashNet) could not
   reveal as wrong.
2. **A dual head fixes this where the training data allows it.** One ConvNeXt-Tiny with
   6 material + 2 recyclability outputs reaches, on the P3 wild test split,
   recyclability accuracy **0.872 ± 0.009** (rule on the same predictions:
   0.534 ± 0.023; "always non-recyclable": 0.657) and AUC **0.922 ± 0.008**, while
   material accuracy is 0.718 ± 0.013 (Section 6.1–6.2).
3. **There is a large domain gap.** Trained on clean images only (P2), the same recipe
   reaches material accuracy **0.407 ± 0.017** on wild images (clean in-domain, P1:
   0.977) — *below* the 0.485 of always predicting `plastic` — and recyclability AUC
   **0.699 ± 0.013** (Section 6.3).
4. **P2's recyclability limit is the training data, not the loss weighting.** In the
   clean training set the only non-recyclable class is `trash` (5.4 %). Removing the
   17:1 class weight lets the head fit the training set (train acc 0.99) but leaves the
   AUC unchanged (0.714 ± 0.019 vs 0.699 ± 0.013); it only moves the decision threshold
   and lowers test accuracy to 0.443, below the trivial baseline (Section 6.4).
5. **The recyclability loss weight is not a sensitive parameter** (0.5 / 1 / 2 give
   0.875 / 0.872 / 0.873 recyclability accuracy at 1 seed; Section 6.5).

**Headline table** (ConvNeXt-Tiny, flip TTA, mean ± std over 3 seeds)

| Protocol | Test set | Material acc. | Recyclability acc. | AUC (non-rec.) | Trivial baseline |
|---|---|---|---|---|---|
| P1 in-domain (single head, Phase 1) | clean test, 257 | 0.977 ± 0.007 | – (rule) | – | – |
| **P3 joint, direct head** | wild test, 417 | **0.718 ± 0.013** | **0.872 ± 0.009** | **0.922 ± 0.008** | 0.657 |
| P3 joint, rule on predicted material | wild test, 417 | 0.718 ± 0.013 | 0.534 ± 0.023 | – | 0.657 |
| **P2 cross-domain, direct head** | wild all, 4132 | **0.407 ± 0.017** | **0.673 ± 0.008** | **0.699 ± 0.013** | 0.633 |
| P2 cross-domain, rule on predicted material | wild all, 4132 | 0.407 ± 0.017 | 0.622 ± 0.003 | – | 0.633 |

The trivial baseline is the accuracy of always predicting the majority recyclability
class of that test set (non-recyclable in both). P1's recyclability is not comparable
(different domain, 94.6 % recyclable).

---

## 2. Task definition and what changed since Phase 1

* **Primary task (unchanged):** 6-way material classification
  (`cardboard, glass, metal, paper, plastic, trash`).
* **Recyclability (changed):** Phase 1 derived it from the predicted material. Here the
  model has a **second output** that predicts recyclability directly (2 classes:
  non-recyclable = 0, recyclable = 1), trained on the manual recyclability label of the
  manifest. The old rule is still evaluated on the same predictions, so every run
  reports both side by side.
* **Why:** the rule is exact in TrashNet (where `trash` is the only non-recyclable
  class) and wrong in TACO (Section 3.3). The dual head also keeps the material
  prediction, so material-level and recyclability-level metrics still come from one
  model.

---

## 3. Data

### 3.1 Domains

| Domain | Source | Images | Non-recyclable share | Splits (train / val / test) |
|---|---|---|---|---|
| clean | TrashNet: studio photographs, white background, one object | 2,527 | 5.4 % | 2,019 / 251 / 257 |
| wild | TACO: item crops cut out of photographs taken outdoors | 4,132 | **63.3 %** | 3,304 / 411 / 417 |

Both domains are harmonized into the same 6 materials and the same recyclable /
non-recyclable label (`manifest.csv`, columns `material`, `recyclable`, `domain`,
`split`).

### 3.2 Material and recyclability by domain (counts)

Clean (TrashNet): every material is 100 % recyclable except `trash` (137 images),
which is 100 % non-recyclable.

Wild (TACO, all 4,132 images):

| Material | Recyclable | Non-recyclable | Non-recyclable share |
|---|---|---|---|
| cardboard | 172 | 78 | 31 % |
| glass | 204 | 0 | 0 % |
| metal | 497 | 6 | 1 % |
| paper | 119 | 125 | 51 % |
| **plastic** | 524 | **1,481** | **74 %** |
| trash | 0 | 926 | 100 % |

Wild is dominated by `plastic` (2,005 images = **48.5 %**): always predicting
`plastic` gives 0.485 material accuracy on the P2 test set (0.482 on the P3 test set:
201 / 417).

### 3.3 Why the Phase 1 rule fails on wild

The rule predicts "non-recyclable" only for `trash`. Evaluated with the **perfect**
material (ground-truth material as the "prediction"):

| Wild split | n | Rule accuracy | Rule non-recyclable recall |
|---|---|---|---|
| test (P3 test set) | 417 | 0.568 | 0.343 |
| all (P2 test set) | 4,132 | 0.591 | 0.354 |

Most of what the rule misses is `plastic`: in the wild test split 161 of 201 plastic
items are non-recyclable. This "perfect-material rule" number is **not an upper bound**
for the rule applied to predictions: material errors can accidentally repair the
recyclability label (a non-recyclable plastic bag predicted as `trash` becomes correct).
In P2 the rule on predicted materials (0.622) is indeed higher than on perfect materials
(0.591).

### 3.4 Prior and label-semantics shift between domains

* **Prior shift:** non-recyclable is 5.4 % of the clean training set and 63.3 % of wild.
* **Label-semantics shift:** in clean data "non-recyclable" means "looks like a
  TrashNet `trash` image"; in wild it is mostly a property of the item (soiled or
  composite plastics, wrappers). A model trained on clean data only has no examples of
  the wild meaning (Section 7.2).

---

## 4. Evaluation protocol

### 4.1 Protocols

| Protocol | Train | Selection | Test |
|---|---|---|---|
| P2 cross_domain | clean, all 2,527 | none (see below) | wild, all 4,132 |
| P3 joint | clean train 2,019 + wild train 3,304 = 5,323 | wild val, 411 | wild test, 417 |

P2 has no labelled target-domain split other than its test set. To avoid choosing
anything on the test data, **P2 runs use a fixed number of epochs (8) with no early
stopping or best-epoch restore**; the reported model is the last epoch. (The log line
"model selection on: TEST split" is printed by shared code for this protocol, but with
no `--patience` no selection takes place; the per-epoch "Val" lines in the P2 logs are
the test set monitored for plotting only.) P3 selects the epoch on the wild val split
(early stopping on the combined validation loss, patience 8) and uses the test split
only for the final numbers.

### 4.2 Metrics

| Metric | Meaning |
|---|---|
| `material_acc`, `material_macro_f1` | 6-class accuracy and macro-F1 of the material head |
| `recyclability_acc` | accuracy of the recyclability head (argmax) |
| `non_recyclable_recall` / `_precision` | for class "non-recyclable" (positive), recyclability head |
| `recyclability_auc_nonrec` | ROC AUC of P(non-recyclable) vs. the true label; **threshold-free** |
| `rule_recyclability_acc`, `rule_non_recyclable_recall` | the Phase 1 rule applied to the predicted material |
| majority baseline | accuracy of always predicting non-recyclable (0.657 on P3 test, 0.633 on P2 test) |

**Statistical resolution.** One P3 test image = 0.24 % accuracy; 1 of the 274
non-recyclable P3 test images = 0.4 % recall. The binomial 95 % half-width of an
accuracy near 0.87 on 417 images is about 3.2 points; near 0.67 on 4,132 images about
1.4 points. Seed standard deviations are 0.9–1.3 points on P3.

**Reading recall.** A classifier that always says "non-recyclable" has recall 1.0 and
accuracy equal to the baseline. Recall is therefore reported only with precision, accuracy
and AUC. AUC is the number least affected by the decision threshold and is used as the
headline measure of discrimination.

---

## 5. Method: the dual head

### 5.1 Architecture

The Phase 1 model with an 8-way output instead of 6 (`build_backbone(n_classes=8)`):

```
image 224x224
  -> ConvNeXt-Tiny (ImageNet-1k, stages 1-2 frozen; stages 3, 4 and the norm trained)
  -> LayerNorm2d -> Flatten
  -> Linear 768->512 -> BN -> ReLU -> Dropout 0.3
  -> Linear 512->256 -> BN -> ReLU -> Dropout 0.3
  -> Linear 256->8
        logits[0:6]  material      (cardboard, glass, metal, paper, plastic, trash)
        logits[6:8]  recyclability (0 = non-recyclable, 1 = recyclable)
```

The two outputs share the whole backbone and the 256-d hidden representation; they differ
only in the final linear layer. `build_backbone`, the unfreezing levels and the optimizer
parameter groups are unchanged from Phase 1 (`waste_bench/engine.py` is untouched; the
dual head lives in `engine_dual.py`, `reporting_dual.py`, `cli_dual.py`, `run_dual.py`).

### 5.2 Loss

```
L = CE(material_logits, material) + rec_weight * CE(recyclability_logits, recyclable)
```

Both terms use label smoothing 0.1 and, with `--class_weighted`, inverse-frequency class
weights **computed per label from the training rows** (so the recyclability weights
differ per protocol). `rec_weight = 1` unless stated. Early-stopping criterion
(`monitor = val_loss`): the combined validation loss, with the same class weights.

### 5.3 Training and evaluation settings (all runs in this report)

| Setting | Value |
|---|---|
| Backbone / unfreeze | ConvNeXt-Tiny, `two_stages` (features[5..7]); 26,818,056 trainable parameters |
| Head | hidden 512 → 256, dropout 0.3 |
| Optimizer | AdamW, weight decay 1e-4, cosine schedule; head lr 3e-4, backbone lr 9e-5 (×0.3) |
| Batch / precision | 32, AMP |
| Augmentation | `strong` (auto for non-in-domain protocols: crop, flips, rotation, affine, colour jitter, grayscale, blur) |
| Label smoothing | 0.1 |
| Class weights | on (material and recyclability), except Section 6.4 |
| P3 | ≤ 30 epochs, patience 8, selection on wild val |
| P2 | 8 epochs, no early stopping |
| Test-time | horizontal-flip TTA (probabilities of image and flip averaged, both heads) |
| Seeds | 42, 1, 2 (`set_seed` before each run; GPU/AMP kernels are not bit-deterministic, so a rerun can differ slightly) |

### 5.4 Run tags

`<protocol>_aug-strong_h512-256_uf-two_stages_cw_blr0.3_ls0.1_tta[_s<seed>]_dual[-rw<w>][-norcw]`.
Seed 42 has no suffix. Runs are stored in `runs/convnext_tiny/<tag>/`
(report PNG, `checkpoint.pt`, `results.json`) and summarized in `runs/_summaries/<tag>.json`.

---

## 6. Experiments and results

### 6.1 P3 (joint): material and direct recyclability

Wild test split: 417 images, 274 non-recyclable, 143 recyclable. Best epoch 3 / 7 / 3
(seeds 42 / 1 / 2); training time 6.5–8.6 min.

| Seed | Best epoch | Material acc. | Macro-F1 | Rec. acc. | Non-rec. recall | Non-rec. precision | AUC |
|---|---|---|---|---|---|---|---|
| 42 | 3 | 0.731 | 0.681 | 0.882 | 0.956 | 0.876 | 0.930 |
| 1 | 7 | 0.724 | 0.665 | 0.873 | 0.956 | 0.865 | 0.912 |
| 2 | 3 | 0.700 | 0.650 | 0.861 | 0.953 | 0.853 | 0.924 |
| **mean ± std** | | **0.718 ± 0.013** | 0.665 ± 0.013 | **0.872 ± 0.009** | 0.955 ± 0.001 | 0.865 ± 0.009 | **0.922 ± 0.008** |

The three seeds agree within 3 points on every metric; the direct head clears the trivial
baseline (0.657) by 0.20–0.22 in every seed.

**The detection rate is partly bias.** Recall 0.955 is almost constant over seeds and
the model leans towards "non-recyclable": from recall, precision and n, the number of
recyclable test images wrongly flagged non-recyclable is 37 / 41 / 45 of 143
(*derived*), i.e. **recyclable recall 0.741 / 0.713 / 0.685** (mean 0.713). Report recall
together with these.

Training dynamics: the best epoch is early (3–7 of ≤ 30); afterwards training
recyclability accuracy keeps rising (0.96–0.98 by epoch 11–15) while validation stays at
0.83–0.87. The validation loss is noisy (411 images) and the selected epoch varies by
seed.

### 6.2 P3: direct head vs. the Phase 1 rule

| Seed | Rec. acc. direct | Rec. acc. rule | Non-rec. recall direct | Non-rec. recall rule |
|---|---|---|---|---|
| 42 | 0.882 | 0.504 | 0.956 | 0.255 |
| 1 | 0.873 | 0.537 | 0.956 | 0.318 |
| 2 | 0.861 | 0.561 | 0.953 | 0.369 |
| **mean ± std** | **0.872 ± 0.009** | 0.534 ± 0.023 | **0.955 ± 0.001** | 0.314 ± 0.047 |

The difference (34 points of accuracy) is more than ten times the seed spread. The
rule's accuracy is below the trivial baseline (0.657) in every seed. Earlier
single-head P3 runs (seed 42, from the Kaggle logs of the previous session, not
re-verified here) showed the same: ConvNeXt-Tiny rule accuracy 0.518 and non-recyclable
recall 0.28, with material accuracy 0.743 (mild) / 0.741 (strong augmentation);
ResNet-50 material accuracy 0.681 / 0.707.

**Cost to material accuracy.** The dual model's material accuracy (0.718 ± 0.013) is
lower than the single-head 0.743 (one seed), but the difference is within the seed
spread (0.700–0.731) and was not isolated: a 3-seed single-head P3 control was not run
(Section 10).

### 6.3 P2 (cross-domain): clean → wild

Test set: 4,132 wild images (2,616 non-recyclable, 1,516 recyclable). Training: clean
only. Final epoch of 8; training time 3.4 min.

| Seed | Material acc. | Macro-F1 | Rec. acc. | Non-rec. recall | Non-rec. precision | AUC | Rule rec. acc. | Rule recall |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.392 | 0.372 | 0.667 | 0.839 | 0.696 | 0.685 | 0.621 | 0.508 |
| 1 | 0.399 | 0.370 | 0.668 | 0.888 | 0.683 | 0.697 | 0.620 | 0.517 |
| 2 | 0.430 | 0.378 | 0.685 | 0.933 | 0.684 | 0.716 | 0.626 | 0.516 |
| **mean ± std** | **0.407 ± 0.017** | 0.373 ± 0.003 | **0.673 ± 0.008** | 0.887 ± 0.038 | 0.688 ± 0.006 | **0.699 ± 0.013** | 0.622 ± 0.003 | 0.514 ± 0.004 |

* **Material:** 0.407 on wild against 0.95–0.96 on the clean *training* set (logs) and
  0.977 on the clean test set (P1). It is below the majority-class baseline of 0.485
  (always `plastic`); macro-F1 (0.373) is a fairer summary.
* **Recyclability:** the direct head is 0.040 above the trivial baseline (0.633), 5 points
  above the rule (0.622, which is itself below the baseline). The gain is consistent
  over seeds and larger than the standard error (0.7 points on 4,132 images), but
  small. Precision (0.688) is close to the non-recyclable prior (0.633): the model
  mostly says "non-recyclable".
* Recall varies by seed (0.84–0.93), so accuracy and AUC are the reliable columns.

### 6.4 P2: recyclability class weights (ablation)

The inverse-frequency weight for the recyclability loss in P2 is 9.22 (non-recyclable)
vs. 0.53 (recyclable), a 17:1 ratio, because the clean training set is 5.4 %
non-recyclable. In the main P2 runs the *training* recyclability accuracy stayed low
(0.60–0.64) while material training accuracy was 0.95–0.96. The ablation removes the
recyclability weights only (`--no_rec_class_weights`); material weights are kept.

| | Material acc. | Rec. acc. | Non-rec. recall | Non-rec. precision | AUC | Train rec. acc. (last epoch) |
|---|---|---|---|---|---|---|
| With rec. weights (3 seeds) | 0.407 ± 0.017 | **0.673 ± 0.008** | 0.887 ± 0.038 | 0.688 ± 0.006 | 0.699 ± 0.013 | 0.61–0.63 |
| Without (seeds 42 / 1 / 2) | 0.395 / 0.389 / 0.444 | 0.448 / 0.441 / 0.440 | 0.160 / 0.137 / 0.140 | 0.833 / 0.871 / 0.853 | 0.689 / 0.718 / 0.734 | 0.993 / 0.993 / 0.993 |
| Without (mean ± std) | 0.409 ± 0.025 | 0.443 ± 0.004 | 0.146 ± 0.010 | 0.852 ± 0.016 | **0.714 ± 0.019** | 0.993 |

Interpretation. The suspicion that the weights hurt learning was **half right**: the
weights are why the training accuracy was low (0.63 → 0.99 without them). But the
**AUC does not change** (0.699 ± 0.013 vs. 0.714 ± 0.019; difference 0.015 is below the
seed spread), so the weights did not limit discrimination. They only shift the decision
threshold. Without them, the head (trained on 95 % recyclable) calls almost everything
recyclable on wild (recall 0.15), and accuracy (0.443) falls below the trivial baseline.
With them, the 17:1 weight accidentally compensates for most of the prior shift
(5 % → 63 % non-recyclable), which is why accuracy is higher. Material accuracy is
unaffected (0.407 vs. 0.409). The rule columns also differ slightly (0.622 vs. 0.590 mean
accuracy; the material predictions differ between the two training runs).

### 6.5 P3: recyclability loss weight (ablation, 1 seed each)

| `rec_weight` | Best epoch | Material acc. | Rec. acc. | Non-rec. recall | Non-rec. precision | AUC |
|---|---|---|---|---|---|---|
| 0.5 | 3 | 0.715 | 0.875 | 0.960 | 0.865 | 0.931 |
| **1** (3 seeds) | 3–7 | 0.718 ± 0.013 (0.700–0.731) | 0.872 ± 0.009 (0.861–0.882) | 0.955 | 0.865 | 0.922 ± 0.008 (0.912–0.930) |
| 2 | 4 | 0.695 | 0.873 | 0.945 | 0.872 | 0.920 |

Recyclability accuracy (0.873–0.875) and AUC (0.920–0.931) are inside the `rec_weight = 1`
seed ranges. Material accuracy at `rec_weight = 2` (0.695) is slightly below the lowest
`rec_weight = 1` seed (0.700), a weak signal at one seed. The default stays at 1.

---

## 7. Discussion

### 7.1 Phase 1's headline overstates deployable performance

The same recipe gives material accuracy 0.977 (clean → clean), 0.718 (clean + wild →
wild) and 0.407 (clean → wild). For a system that receives photographs of real waste,
P1 is the least informative number. The comparison is the central result of this phase:
the gap is attributable to the domain, since the architecture, recipe, optimizer and
augmentation policy are identical across the three protocols.

### 7.2 Recyclability is not a function of material in the wild

Sections 3.2–3.3 show that the manual label is item-level in TACO. The Phase 1 rule
scores below the trivial baseline in P3 (0.534 vs. 0.657) and in P2 (0.622 vs. 0.633). A
direct head is the right design for the stated goal; in P3 it lifts accuracy by 34
points and reaches AUC 0.92.

### 7.3 What limits P2

Three observations point to the training data rather than to the model or the loss:

1. Fitting is not the problem: the head reaches 0.99 training recyclability accuracy
   when unweighted.
2. The AUC does not depend on the weighting (Section 6.4).
3. In P3, which adds 3,304 labelled wild images, AUC rises from ≈ 0.70 to 0.92.

The clean training set contains only one non-recyclable concept (`trash`, 137 images);
the wild test set contains mostly another (plastic items). We did not test this causal
explanation directly (a wild-fraction learning curve would); it is the simplest account
of the three observations.

### 7.4 The decision threshold is domain-dependent

At a fixed argmax threshold the P2 accuracy ranges from 0.44 to 0.67 depending on
class weights, with unchanged AUC. Any deployment on a domain with a different prior
needs a calibration or threshold step; reporting AUC next to accuracy is necessary for
honest comparisons across protocols.

---

## 8. Threats to validity

1. **Seeds and sizes.** 3 seeds per configuration, except the two `rec_weight` ablation
   points (1 seed each). The P3 test set is small (417 images: 0.24 % per image;
   ± 3.2 points binomial); P2 has a large test set (4,132) but a single training
   distribution.
2. **P3 split leakage is not ruled out.** The wild items are crops of TACO photographs.
   We did not verify whether crops of the same photograph are confined to one split. If
   they are not, P3 numbers (and the clean + wild vs. wild contrast with P2) are
   optimistic. This should be checked before the P3 numbers are quoted as clean
   generalization.
3. **Model selection noise (P3).** The best epoch varies (3–7) with a 411-image
   validation split; differences between runs of 1–2 points are not interpretable.
4. **P2 has no selection split.** We fixed 8 epochs from the P3 behavior (best epochs
   3–7) and the P2 validation curve was not used. The result is the last epoch and may
   differ from the best epoch of the test curve (the per-epoch test material accuracy
   varied by ±3 points between epochs in the logs).
5. **Label semantics.** The recyclability label of the manifest is manual and
   item-level in TACO (`label_mapping.csv` policy). We did not audit label noise in the
   wild domain. The recyclability AUC is bounded by this noise.
6. **One backbone and recipe.** The recipe is tuned for P1 (clean in-domain); ablations
   in this phase vary only the loss weights. A recipe tuned for P2/P3 (augmentation
   strength, unfreezing depth, domain-adaptation methods) could differ.
7. **Single-head vs. dual-head comparison is incomplete.** The single-head P3 numbers
   are from one seed of an earlier session (Section 6.2), so the cost of the second
   head on material accuracy is not established.
8. **TTA and AMP.** All evaluations use flip TTA and AMP inference; numbers are not
   directly comparable to evaluations without them.

---

## 9. Reproducibility

Code state: `waste_bench` at commit `646ee63` (dual head: `1ce746d`; `--no_rec_class_weights`:
`646ee63`). Entry point `run_dual.py`; the base recipe and the P3 command:

```
BASE = --backbones convnext_tiny --class_weighted --head_hidden 512 256 --lr 3e-4 \
       --unfreeze two_stages --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --amp --workers 4
P3   : python run_dual.py --protocol joint --epochs 30 --patience 8 <BASE> [--seed S] [--rec_weight W]
P2   : python run_dual.py --protocol cross_domain --epochs 8 <BASE> [--seed S] [--no_rec_class_weights]
```

Each run writes `runs/convnext_tiny/<tag>/{report_convnext_tiny.png, checkpoint.pt,
results.json}` and `runs/_summaries/<tag>.{json,log}`. The aggregation cell used for the
tables in Sections 6.1–6.4 reads these `results.json` files and prints mean ± std
(population) per tag group.

Runs reported here (tags without the common prefix):

* P3: `joint_…_tta_dual`, `…_tta_s1_dual`, `…_tta_s2_dual`; `…_dual-rw0.5`, `…_dual-rw2`.
* P2: `cross_domain_…_tta_dual`, `…_s1_dual`, `…_s2_dual`; `…_dual-norcw`, `…_s1_dual-norcw`,
  `…_s2_dual-norcw`.

Dual-head checkpoints are not yet readable by `run_predict.py` / `ensemble_eval.py` (they
build a 6-output model); see Section 10.

---

## 10. Next steps

1. **Wild-fraction learning curve** (new `--wild_fraction`): train on clean + 5 / 10 / 25 /
   50 % of wild train, test on the P3 wild test split. Quantifies how many labelled wild
   images close the gap between P2 (AUC 0.70) and P3 (0.92); this is the most practical
   question for the project goal.
2. **Leakage check for P3** (Section 8, item 2): confirm that crops of one TACO
   photograph stay in one split, or re-split by photograph and rerun P3.
3. **Single-head P3 control with 3 seeds** to measure what the second head costs in
   material accuracy.
4. **Threshold / prior calibration** for P2: evaluate precision–recall and a
   target-prior-adjusted threshold (a decision rule, not retraining), and report AUC, PR
   curves and calibration next to accuracy.
5. **Per-material analysis** of the direct head in wild (where recyclability errors
   concentrate: plastic, paper, cardboard) using the saved checkpoints.
6. **Dual-head inference tooling:** `run_predict_dual` and an ensemble evaluation over the
   three seeds (ensembling raised P1 from 0.977 to 0.988).
7. **Domain adaptation** options for P2 (stronger augmentation, domain-adversarial
   training, feature-statistics alignment) as a separate phase.
