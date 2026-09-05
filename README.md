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

## FAQ — decisions made while building this pipeline

A running Q&A log of the reasoning behind the pipeline's design and the
`benchmark_backbones.py` tooling, kept so the "why" behind each choice
doesn't get lost. See also [`methodology.md`](methodology.md) for the formal
write-up and [`RUNBOOK.md`](RUNBOOK.md) for the commands.

### Setup / infrastructure

**Q: Can the dataset/code be committed to git?**
Yes for code, docs, small CSVs, and the two smaller image folders
(`trashnet_dataset-resized/` 48MB, `crops_material/` 63MB) — tracked via
**Git LFS** since GitHub blocks files over 100MB and warns on large repos.
`Taco_data/` (2.5GB) and `update_Taco_classification/processed_data/` (1.1GB)
stay gitignored — too large even for the free LFS tier (1GB storage/bandwidth).

**Q: Do I have a usable local GPU?**
Checked: an NVIDIA MX130 with 2GB VRAM is present, but it's an entry-level
laptop chip — batch size 32 on ResNet50/DenseNet121/Inception_v3 will OOM,
and even a shrunk batch would train far slower than a free cloud GPU. All
real training runs happen on **Kaggle/Colab** (free T4, 16GB VRAM), not locally.

**Q: How do I get the updated code onto Kaggle?**
`git clone` the GitHub repo into the Kaggle notebook (with internet enabled
in notebook settings, and a GitHub Personal Access Token for the private
repo, passed via Kaggle Secrets — never pasted in plaintext). Run
`git lfs pull` after cloning so the tracked images resolve to real files, not
pointer stubs. `manifest.csv`'s paths need backslash→forward-slash
normalization for Linux (Kaggle) after cloning, since it was generated on
Windows.

### The classification pipeline itself

**Q: What does `benchmark_backbones.py` actually run, end to end?**
For each requested backbone: load ImageNet-pretrained weights → freeze the
backbone (unless `--finetune_head`) → replace the head with a 6-way material
classifier → train on the protocol's train split → evaluate on its test
split → derive binary recyclable/non-recyclable from the *predicted material*
via a fixed lookup table (`MATERIAL_RECYCLABLE`) → save metrics, curves, and
a report figure. See [`RUNBOOK.md`](RUNBOOK.md) for the full phase-by-phase
breakdown (label harmonization → crops/manifest → 3 protocols → ablations →
aggregated tables).

**Q: What do `in_domain`, `cross_domain`, and `joint` mean?**
- `in_domain`: train clean(TrashNet train) → test clean(TrashNet test). Upper-bound sanity check, no domain shift.
- `cross_domain` (**headline**): train clean(all TrashNet) → test wild(all TACO). Directly measures the domain gap.
- `joint`: train clean+wild(train splits) → test wild(test). Shows the benefit of merging both datasets under the unified taxonomy.

**Q: `--no_domain_randomize` — what does the flag's presence/absence actually mean?**
It's a double negative by naming convention: **omit the flag → augmentation is ON** (the normal case). **Pass the flag → augmentation is OFF.** There is no separate "turn augmentation on" flag — silence means yes.

**Q: Should the same augmentation strength apply to every protocol?**
No — this became the `--aug_mode {auto,off,mild,strong}` flag.
`in_domain` has no domain gap to fight, so aggressive color/blur/grayscale
distortion can only cost accuracy there with no generalization payoff; it
auto-selects `mild` (geometric variety only: crop, flip, rotation, small
shift/shear). `cross_domain`/`joint` auto-select `strong` (mild + color
jitter, grayscale, blur) since that's the set actually designed to fight the
clean/wild domain gap. `--aug_mode off/mild/strong` overrides the auto choice.

**Q: I want to weight the loss because `trash` is under-represented — how, and how do I check the weights before trusting a run?**
`--class_weighted` (inverse-frequency weighting, computed from the training
split only). `class_weights_from()` now logs the exact per-class counts and
computed weights before training starts, specifically so an imbalance like
`trash` (137 of 2,527 TrashNet rows vs. 400–600 for other classes) is
visible up front rather than trusted blindly.

**Q: Is `manifest.csv` shuffled, and does that matter?**
No — it's exactly 12 contiguous blocks (grouped by domain, then by material
within each domain; wild's 6 material blocks first, then clean's 6). This
does **not** break training because the training `DataLoader` uses
`shuffle=True`, which re-randomizes batch composition every epoch regardless
of file order. It **did** originally break the report's sample-prediction
grid, though — see below.

**Q: The sample-prediction grid in the report only ever seemed to show one class — why, and how was it fixed?**
Because `eval_model` originally took "the first N images the (unshuffled)
test loader produced" — and since the manifest is class-blocked, that could
mean 12 images of a single material. Fixed by collecting up to 2 samples
*per material class* during the eval pass, so the 12-image grid spans all 6
classes (and both recyclable/non-recyclable) by construction, not by luck of
file order.

**Q: The grid shows a recyclability label next to "true" — is that model output?**
No, and this was corrected after review: the "true" recyclability is the
dataset's **manual** ground-truth label from `manifest.csv` (ultimately
authored in `label_mapping.csv`'s item-aware policy) — never a prediction.
The model itself has no recyclability head in this script; it only predicts
material. An earlier version also showed a "pred: rule-derived recyclability"
line (`MATERIAL_RECYCLABLE[predicted_material]`) next to the prediction —
that was removed because it risked being misread as the model's own
recyclability judgment when it's really just a fixed lookup applied
after the fact. The grid now shows only `true:material (recyclable)` /
`pred:material`.

**Q: Should I port the notebook's offline augmentation (rotate/flip/shift, saved as new files on disk, ~7x more files) into `benchmark_backbones.py`?**
No — recommended against, and instead the existing **online** augmentation
was extended with more variety (added `RandomVerticalFlip`, `RandomRotation`,
`RandomAffine` translate/shear, gated by `--aug_mode`) rather than adopting
the disk-based approach. Reasons: (1) online augmentation gives the same
statistical benefit without ever repeating the exact same augmented image,
since transforms are re-sampled every epoch; (2) it needs no disk space and
no `manifest.csv` regeneration/leakage bookkeeping; (3) it keeps the
pipeline's manifest-driven, deterministic design intact. The log now prints
train/val sample counts plus "effective training views" (`train_rows ×
epochs`) so the exposure increase from augmentation is visible without
needing a literal file-count change.

**Q: Can I compare "linear probe" vs. a richer fine-tuned head across backbones?**
Yes — `--finetune_head` applies a consistent policy across all 4 backbones
(vgg13 was dropped from the benchmark by request): a 2-layer MLP head
(Linear→ReLU→Dropout→Linear) plus unfreezing roughly the last stage of each
backbone (`resnet50`: `layer4[2]`; `mobilenet_v2`: `features[16:19]`;
`densenet121`: `denseblock4`+`norm5`; `inception_v3`: `Mixed_7c`), instead of
a fully-frozen backbone + single linear layer.

**Q: Can training find the best epoch automatically instead of just running a fixed count?**
Yes — `--patience N` (paired with a large `--epochs`) tracks validation loss
each epoch, keeps the best-so-far weights, stops early once N epochs pass
with no improvement, and restores the best epoch's weights at the end.
Caveat: "validation" here is the same test split used for final evaluation
(the manifest has no separate held-out validation set distinct from test),
so this is a convenience for experimentation, not a leakage-free protocol —
worth flagging if it ever appears in a paper's methodology section.

**Q: What extra stats does the benchmark report per model, for comparing them?**
Training wall-clock time, GPU utilization (avg/max %, sampled on a
background thread via `pynvml`; safely reports "n/a" on CPU-only machines
instead of crashing), and inference latency (ms/image) + throughput
(images/sec), the last two measured with proper CUDA synchronization and a
warmup-batch exclusion so cudnn autotuning doesn't skew the numbers. All of
it is logged (console + a `.log` file per run) and saved into the results
JSON alongside the accuracy/F1 metrics.

## Why this subject, before there was a dataset

It's worth separating "why waste classification" from "why TrashNet+TACO" —
the dataset choice is downstream of a more basic reasoning chain, and that
chain is the part worth reusing on a future project even if the subject
changes entirely.

**1. Pick a domain where a wrong answer has a concrete, checkable cost.**
Waste sorting isn't interesting because classification is hard in the
abstract — it's interesting because a wrong sort has a real consequence: a
recyclable item lands in a landfill, or a non-recyclable/contaminated item
lands in the recycling stream and can spoil the whole batch. That gives a
principled reason to care about *which kind* of error matters most
(missing a non-recyclable item is worse than the reverse), instead of
picking an evaluation metric arbitrarily. Contrast this with a subject
where every misclassification is equally inconsequential — there, metric
choice is just a formality, not a design decision.

**2. Ask how the thing would actually be *used*, before looking at data.**
A smart-bin camera or a sorting-line camera sees cluttered, outdoor,
arbitrary-background scenes — not staged studio photos. Asking "what does
the input look like at the point of use" *before* touching a dataset is
what exposes the eventual problem; if you start from the dataset instead,
you inherit its assumptions (staged, clean, single-object) without
noticing they might not hold at deployment.

**3. Only then look at what data exists, and check it against step 2.**
Public waste datasets exist in two flavors: easy-to-collect and clean
(TrashNet: studio shots, single object, 6 tidy classes) or realistic but
harder to use directly (TACO: outdoor, cluttered, 60-category detection
data). Neither alone matches "how it's actually used" (step 2) *and* is
big enough to train on well. That mismatch is not a fact you look up — it's
a fact you only notice because step 2 gave you a question to check the data
against.

**4. Treat the mismatch as the actual research opportunity, not a nuisance.**
The lazy path is "use TrashNet because it's clean and easy" and ignore
that it doesn't represent deployment. The path that produces a paper
instead of a demo is "the mismatch between what's easy to collect and what
deployment looks like is itself worth measuring and explaining" — which is
exactly what the `cross_domain` protocol and the Grad-CAM shortcut-learning
ablation do in this repo.

**5. This reasoning pattern outlives the specific subject.**
"An easy-to-collect benchmark dataset exists; a messy real-world one also
exists; they don't share a label space or visual distribution; a model
trained on the easy one may fail on the real one" is a pattern that repeats
across domains — medical imaging (lab vs. field), agriculture (staged vs.
drone/field photos), retail (catalog vs. security-camera images). Being
able to name *why* trash classification has this structure is a more
durable skill than the trash classifier itself; it's the difference between
having solved one dataset and having a repeatable way to find the next
project.

Only after these five steps do the concrete decisions in this repo follow:
which datasets to harmonize, what taxonomy to define, which protocols to
run, which ablations isolate which effect. Those are documented in the FAQ
above and the paper-framing section below — but they are *answers*, and the
five points here are the *questions* that made those answers necessary.

## From this pipeline to a paper: how to think about it

The project already *is* recyclable-vs-non-recyclable classification — the
6-class material head is a means to that end, not a competing goal. Framed
as a paper, the story is not "we built a classifier" (many exist) but
**"naively classifying trash images does not generalize, and here is why, and
here is how much of the gap you can close and by what mechanism."** A
concrete way to think through that framing:

**1. Start from the deployment problem, not the dataset.**
A real smart-bin / recycling-sorter camera sees *wild* images (TACO-like:
cluttered, outdoor, arbitrary background) — but nearly every public labeled
waste dataset (TrashNet) is *clean* (studio-shot, white background,
single object). This mismatch is the actual problem worth writing about: a
model that looks great on the dataset used to train it can be nearly useless
once deployed, because it silently learned a shortcut (background/lighting)
rather than the object's material.

**2. State the concrete failure mode as a hypothesis, then test it.**
"Models trained on clean data exploit background as a shortcut for material,
and this shortcut collapses on wild images." This is falsifiable:
`cross_domain` (train clean, test wild) vs. `in_domain` (train/test clean)
gives a measurable **domain gap**. The Grad-CAM / background-removal
ablation (`gradcam_ablation.py`) is what turns "accuracy dropped" into "and
here is visual/quantitative evidence of *why* — the model was attending to
background pixels, not the object."

**3. Data can't just be "more" — it has to be *harmonized* before it's useful together.**
TrashNet (6 classes, classification) and TACO (60 categories, detection
boxes) don't share a label space or a task shape out of the box. Two
separate design decisions had to be made deliberately, not
assumed away:
   - **Task compatibility**: TACO is detection, TrashNet is classification —
     resolved by cropping TACO's annotated boxes into object-centric crops
     (lossy in context, but reversible in principle and the only direction
     that doesn't require re-annotating TrashNet for detection).
   - **Label compatibility**: a 60→6→2 ontology (`label_mapping.csv`) maps
     every TACO category to one of 6 materials, then deterministically to
     recyclable/non-recyclable via an explicit, stated **policy** (item-aware:
     rigid clean plastics/metal/glass/paper/cardboard recyclable; films,
     foams, soiled items, cigarettes, organics non-recyclable). A paper must
     own this as a limitation, not hide it — recyclability is
     facility-dependent, so the mapping is *a* defensible policy, not *the*
     truth.

**4. Anticipate the confound before a reviewer finds it.**
TrashNet's recyclable label is almost totally confounded with "is it the
trash class" (94.6% recyclable) — so a model could score well on TrashNet's
binary label by accident, without learning anything material-specific. TACO,
by contrast, splits recyclability *within* materials (e.g. some plastics
recyclable, some not), making it the harder and more meaningful binary
target. This is why the 6-class **material** accuracy is the headline
metric and binary recyclability is reported "alongside, with the caveat" —
not the other way around.

**5. Long-tail class imbalance is a data property to report, not paper over.**
TACO is plastic-heavy (56% of wild rows); TrashNet's `trash` class is the
smallest of its 6. A paper should show the per-class breakdown (already
computed on demand from `manifest.csv`), report macro-F1 alongside accuracy
(macro-F1 penalizes ignoring small classes; raw accuracy doesn't), and treat
`--class_weighted` / augmentation strength as an ablation with a measured
effect, not an unstated fix.

**6. The improvement story should be progressive, each step earning its place.**
Naive merge/single-dataset training fails to transfer → the unified taxonomy
makes a joint model possible at all → the cross-domain numbers reveal *how
much* the gap is → domain-randomizing augmentation, then optionally
DANN, then background-removal, each *close* the gap by a measured amount,
with Grad-CAM providing the visual "why." Every ablation in
[`RUNBOOK.md`](RUNBOOK.md) Phase 3 exists to isolate one factor's
contribution to that closing, not just to report a bigger final number.

**7. The paper's contribution is the benchmark + the diagnosis, not the accuracy number.**
The deliverable worth releasing is the **harmonized dataset artifact**
(`label_mapping.csv` + `manifest.csv` + the crop scripts + coverage report)
so others can reproduce or extend the cross-domain study, plus the diagnosis
of *why* the gap exists (shortcut learning) and *how much* each
intervention closes it — not a claim of state-of-the-art accuracy on either
dataset alone.

## سؤال 1 / Question 1 — دمج الـ dataset كـ contribution واسمه

### بالعربي

**هل ممكن دمج الـ 2 dataset (TACO + TrashNet) كمساهمة (contribution) في البحث، وإديله اسم فريد؟**

نعم، وهو بالفعل مساهمة قوية ومقبولة علميًا. في أبحاث الـ Machine Learning، مش لازم الـ contribution يكون معمارية جديدة — من أقوى الأنواع المقبولة:
1. **Benchmark جديد** (تجميع بيانات بطريقة متسقة يقدر غيرك يستخدمها)
2. **Unified taxonomy** (توحيد تصنيفات مختلفة تحت نظام واحد)
3. **Empirical finding** (اكتشاف تجريبي، زي "الموديل بيفشل لما ينتقل من بيئة لبيئة")

المشروع ده فيه الثلاثة مع بعض: TACO (60 category) وTrashNet (6 classes) اتوحدوا تحت نظام واحد (60→6→2)، ومفيش حد قبل كده وحدهم بالشكل ده وربطهم بـ recyclability بشكل منهجي، والدمج نفسه كشف مشكلة حقيقية (الـ domain gap بين البيانات النضيفة والواقعية).

**أسماء مقترحة للـ dataset الموحد:**
| الاسم | ليه مناسب |
|---|---|
| **CD-Waste** (Cross-Domain Waste) | بيركز على الميزة الأساسية: قياس الفجوة بين البيئتين |
| **TACO-TrashNet Unified Benchmark (TTU)** | مباشر وواضح، موضح إنه دمج الاثنين |
| **WasteHarmonized-6M2** | بيوضح الهيكل الهرمي 60 فئة → 6 مواد → 2 قابلية لإعادة التدوير |

الأقرب لروح المشروع (بما إن `methodology.md` بيركز على cross-domain finding كمساهمة أساسية) هو **CD-Waste** أو **TACO-TrashNet Unified Benchmark**.

### In English

**Can merging the two datasets (TACO + TrashNet) itself be a paper contribution, with its own unique name?**

Yes — and it is already a strong, well-accepted kind of contribution. In ML research, a contribution doesn't have to be a new architecture. Some of the most accepted contribution types are:
1. **A new benchmark** (a consistently assembled dataset others can reuse)
2. **A unified taxonomy** (reconciling incompatible label systems into one)
3. **An empirical finding** (e.g. "a model trained in one setting fails when transferred to another")

This project has all three: TACO's 60 categories and TrashNet's 6 classes are reconciled into one 60→6→2 taxonomy, no prior harmonization of these two specific datasets exists in this systematic form, and the harmonization itself is what surfaced a real, measurable problem (the clean/wild domain gap).

**Suggested names for the unified dataset:**
| Name | Why it fits |
|---|---|
| **CD-Waste** (Cross-Domain Waste) | Foregrounds the core contribution: measuring the gap between the two domains |
| **TACO-TrashNet Unified Benchmark (TTU)** | Direct and explicit about being a merge of the two |
| **WasteHarmonized-6M2** | Makes the 60→6-material→2-recyclability hierarchy explicit in the name |

Given that `methodology.md` centers the cross-domain finding as the headline contribution, **CD-Waste** or **TACO-TrashNet Unified Benchmark** fit best.

## سؤال 2 / Question 2 — تصميم الـ modeling وعلاقته بالهدف النهائي

### بالعربي

**الهدف النهائي هو موديل واحد مدرّب على الـ 2 datasets، يطلع recyclable/non-recyclable بناءً على اسم الـ class — فإزاي أوضح إن تصميم الـ modeling الحالي هو الطريق الصح لده؟**

**1. ليه الموديل بيتنبأ بالمادة (6 classes) مش recyclable/non-recyclable مباشرة؟**
لو الموديل اتنبأ بالتصنيف الثنائي على طول، أي غلط هيبقى غير قابل للتفسير (مش هتعرف ليه غلط)، وممكن يطلع صح بالصدفة حتى لو لخبط بين موادّ مختلفة. لما الموديل يتنبأ بالمادة الأول، وبعدين تُشتق الـ recyclability من اسم المادة عن طريق قاعدة ثابتة:
```python
MATERIAL_RECYCLABLE = [1, 1, 1, 1, 1, 0]  # cardboard, glass, metal, paper, plastic → recyclable | trash → non-recyclable
pr = [MATERIAL_RECYCLABLE[i] for i in pm]
```
القرار النهائي بيبقى **قابل للتفسير بالكامل**: أي غلط في recyclability تقدر ترجعه لغلط محدد في التنبؤ بالمادة. وده **حرفيًا** الطريقة اللي هدفك النهائي وصفها: "recyclable بناءً على اسم الـ class."

**2. ليه فيه 4 معماريات مختلفة (ResNet50, DenseNet121, MobileNetV2, Inception_v3)؟**
عشان تثبت إن النتيجة مش نتيجة حظ معمارية معينة، ولإنك تقدر توصي بأنسب موديل للنشر الفعلي (MobileNetV2 أخف وأسرع لجهاز smart bin محدود الموارد، ResNet50 أدق لكن أتقل).

**3. ليه Backbone متجمّد (frozen) في الأساس؟**
البيانات المدموجة (~6,659 صورة) صغيرة نسبيًا مقارنة بمليون صورة ImageNet. تجميد أغلب الطبقات بيمنع الـ overfitting، والاستفادة بالمعرفة العامة اللي الموديل اتعلمها مسبقًا عن الأشكال والقوام.

**4. ليه فيه `--finetune_head` كخيار إضافي؟**
عشان تختبر: هل الموديل محتاج يتعلم أكتر من مجرد ربط فيتشرز جاهزة، ولا محتاج كمان يعدّل آخر طبقات الفهم البصري نفسها عشان يتكيف مع بيانات الـ TACO الواقعية المختلفة بصريًا عن ImageNet؟

**5. الترتيب اللي بيوصلك فعليًا للموديل النهائي:**
- `in_domain` → تتأكد إن المعمارية أصلاً بتتعلم كويس في أسهل حالة
- `cross_domain` → تكتشف وتقيس المشكلة (هل فيه اعتماد على الخلفية؟)
- **`joint`** → هنا فعليًا بتدرب على الـ 2 datasets **مع بعض**، وده أقرب موديل لهدفك النهائي
- `class_weighted` + `aug_mode strong` → تضمن إن الموديل مش بيهمل المواد القليلة (زي trash) وبيتعلم من تنوع بصري حقيقي

**خلاصة:** الموديل النهائي هو ناتج بروتوكول `joint` مع أفضل backbone وأفضل إعدادات، والقرار النهائي بيتاخد من اسم المادة اللي اتنبأ بيها — بالظبط زي ما هو مطلوب.

### In English

**The end goal is one model, trained on both datasets, that outputs recyclable/non-recyclable based on the predicted class name — how do I explain that the current modeling design is the right path to that?**

**1. Why does the model predict material (6 classes) instead of recyclable/non-recyclable directly?**
A direct binary head makes errors uninterpretable (you can't tell *why* it was wrong) and can be right by accident even when it confuses two different materials. By predicting material first and deriving recyclability from a fixed rule:
```python
MATERIAL_RECYCLABLE = [1, 1, 1, 1, 1, 0]  # cardboard, glass, metal, paper, plastic → recyclable | trash → non-recyclable
pr = [MATERIAL_RECYCLABLE[i] for i in pm]
```
the final decision becomes **fully explainable** — any recyclability error traces back to a specific material misprediction. This is *literally* the mechanism your end goal describes: "recyclable, based on the class name."

**2. Why benchmark 4 different backbones (ResNet50, DenseNet121, MobileNetV2, Inception_v3)?**
To prove the finding isn't an artifact of one architecture's quirks, and to be able to recommend the right model for real deployment (MobileNetV2 lighter/faster for a resource-constrained smart bin; ResNet50 more accurate but heavier).

**3. Why is the backbone frozen by default?**
The merged dataset (~6,659 images) is small relative to the million-image ImageNet the backbone was pretrained on. Freezing most layers avoids overfitting and lets the model reuse general shape/texture knowledge instead of memorizing the small dataset.

**4. Why does `--finetune_head` exist as an extra option?**
To test whether the model needs more than just re-combining frozen features — whether the last few visual-understanding layers themselves need to adapt to TACO's visually different, real-world data.

**5. The sequence that actually leads to the end-goal model:**
- `in_domain` → confirms the architecture can learn the 6 materials at all, in the easiest setting
- `cross_domain` → surfaces and measures the real problem (is it relying on background?)
- **`joint`** → this is where training actually happens on **both datasets together** — the closest thing to the end-goal model
- `class_weighted` + `aug_mode strong` → ensures the model doesn't ignore rare materials (like trash) and learns from real visual variety

**Bottom line:** the end-goal model is the output of the `joint` protocol with the best-performing backbone and settings, with the final recyclable/non-recyclable decision derived directly from the predicted material's class name — exactly as intended.

## سؤال 3 / Question 3 — ليه معملناش TACO لوحدها بما إنها أقرب للواقع؟

### بالعربي

**TACO أقرب للواقع، فليه معملناش الشغل عليها لوحدها؟ TrashNet ساعدت في إيه بالظبط؟**

TACO لوحدها فيها مشكلتين حقيقيتين:

**1. ندرة البيانات لكل فئة (data scarcity):** من `manifest.csv`، فئات زي **glass (163 صورة تدريب)** و**paper (195 صورة)** قليلة جدًا لدرجة إن الموديل مش هيقدر يتعلم منها كويس لوحدها — هيحصل overfitting، ومش هيعمم صح. ده غير إن الـ 60 category الأصلية كانت متبعترة أكتر قبل التوحيد لـ 6 materials.

**2. TACO مصممة أصلاً كـ detection مش classification:** الصور فيها أجسام متعددة وbounding boxes، فاضطرينا نعمل crop لكل box عشان نحولها لتاسك تصنيف — والعملية دي بذاتها بتفقد معلومات (السياق حوالين الجسم، الدقة الأصلية).

**TrashNet ساعدت في حل المشكلتين دول:**
- **حلت ندرة البيانات:** لما ضفناها لـ TACO في بروتوكول `joint`، الموديل بقى عنده حجم بيانات أكبر بكتير يتعلم منه شكل كل مادة (322–475 صورة تدريب لكل مادة في TrashNet).
- **أعطت مرجع نظيف للمادة:** لما الموديل يشوف زجاج نضيف في استوديو (TrashNet) وزجاج متكسر وسط زبالة (TACO)، بيتعلم إن خصائص المادة نفسها (شفافية، لمعان) بتفضل ثابتة حتى لو البيئة اتغيرت.
- **الأهم: كشفت المشكلة الحقيقية اللي البحث كله مبني عليها.** لو استخدمنا TACO بس، ما كناش هنقدر نقيس حاجة اسمها **domain gap** أصلاً — لأن مفيش حاجة "نضيفة" نقارن بيها. TrashNet بتمثل الحالة السهلة (clean domain)، وTACO بتمثل الحالة الواقعية الصعبة (wild domain). لازم الاتنين مع بعض عشان نحسب:
```
Domain gap = (دقة الموديل على البيانات النظيفة) − (دقة نفس الموديل على البيانات الواقعية)
```
لو استخدمنا TACO بس، مكانش هيبقى عندنا "قبل" نقارنه بـ"بعد"، ومكانش هيظهر الاكتشاف الأساسي للبحث (إن الموديل بيعتمد على الخلفية كـ shortcut).

**خلاصة:** TrashNet مكانتش الهدف، كانت أداة ضرورية — لحل ندرة البيانات، ولإعطاء مرجع نظيف للمادة، والأهم لإنها المقارنة الوحيدة اللي بتسمحلنا نثبت ونقيس وجود المشكلة أصلاً.

### In English

**TACO is closer to real-world use, so why not just work on it alone? What did TrashNet actually help with?**

TACO alone has two real problems:

**1. Severe per-class data scarcity:** from `manifest.csv`, classes like **glass (163 training images)** and **paper (195 images)** are too few for a model to learn them well in isolation — it would overfit and generalize poorly. This is on top of the original 60 categories being even more fragmented before being unified into 6 materials.

**2. TACO was designed for detection, not classification:** its images contain multiple objects with bounding boxes, so each box had to be cropped out to turn it into a classification task — and that cropping itself loses information (surrounding context, original resolution).

**TrashNet solved both of these:**
- **Fixed the data-scarcity problem:** merging it with TACO under the `joint` protocol gave the model a much larger pool to learn each material's appearance from (TrashNet has 322–475 training images per material).
- **Provided a clean reference for the material itself:** seeing a clean studio glass object (TrashNet) alongside a broken, cluttered glass object in a street scene (TACO) teaches the model that a material's core visual properties (transparency, shine) stay consistent even as context changes.
- **Most importantly, it's what made the paper's core finding measurable at all.** Using TACO alone, there would be no **domain gap** to measure — nothing "clean" to compare against. TrashNet represents the easy case (clean domain), TACO the hard, realistic case (wild domain). Both are needed to compute:
```
Domain gap = (accuracy on clean data) − (accuracy of the same model on wild data)
```
Without TrashNet, there would be no "before" to compare against an "after," and the paper's central finding — that the model leans on background as a shortcut — would never have surfaced.

**Bottom line:** TrashNet wasn't the goal — it was a necessary tool: it fixed data scarcity, gave a clean reference for what a material actually looks like, and, most importantly, made it possible to prove and measure that the problem exists in the first place.
