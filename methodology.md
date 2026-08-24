# Cross-Domain Waste Classification: A Harmonized TACO + TrashNet Benchmark for Recyclability Prediction

## 1. Problem & Contribution

Existing waste-image datasets are siloed: **TrashNet** is clean, single-object, studio-shot classification data, while **TACO** is cluttered, in-the-wild detection data with a fine-grained 60-category taxonomy. Models trained on one fail on the other because they exploit background shortcuts (clean white vs. natural scenes) rather than learning material properties.

**Our two contributions:**
1. **(Idea 2) A unified taxonomy and benchmark.** We define an ontology that maps TACO's 60 categories and TrashNet's 6 classes into a shared 6-class material space, then deterministically into a binary recyclability label. We release the mapping table, the harmonized dataset, and baselines.
2. **(Idea 1) A cross-domain generalization study.** Using domain tags (clean vs. wild), we measure the domain gap and show how joint training plus domain-randomizing augmentation closes it.

The end deliverable is a model that, given a **new** image, predicts the material and outputs **recyclable / non-recyclable**.

## 2. Datasets

| | TACO | TrashNet |
|---|---|---|
| Images | ~1,500 | ~2,527 |
| Annotations | ~4,800 (COCO masks/boxes) | folder labels |
| Task | detection/segmentation | classification |
| Domain | wild / cluttered | clean / studio |
| Taxonomy | 60 cats, 28 supercats | 6 classes |

## 3. Label Harmonization (the core method)

We use a **three-level hierarchy** so recyclability is a *function of material*, not an independent guess:

```
TACO 60 categories ─┐
                    ├─► 6 material classes ─► binary recyclability
TrashNet 6 classes ─┘   (cardboard, glass,     (recyclable /
                         metal, paper,           non_recyclable)
                         plastic, trash)
```

The mapping is provided as `label_mapping.csv` (TACO supercategory → material → recyclability + notes). **Policy decision (state explicitly in the paper):** recyclability is contextual and facility-dependent, so we adopt an item-aware policy where rigid PET/HDPE plastics, clean metals, glass, paper, and corrugated cardboard are recyclable, while films, foams (EPS/styrofoam), food-soiled items, lined paper cups, cigarettes, and organics are non-recyclable. This is the most defensible single policy; the paper should acknowledge it as a limitation and a configurable parameter.

## 4. Making the Tasks Compatible

TACO is detection; TrashNet is classification. We unify on **classification** by cropping each TACO annotated object using its bounding box, yielding object-centric crops comparable to TrashNet samples. This is reversible and loss-free for our purpose, and avoids the harder direction (annotating TrashNet for detection). A minimum crop size (e.g., 32px) filters degenerate boxes.

## 5. Experimental Protocols

Each sample carries a `domain` tag (`clean` for TrashNet, `wild` for TACO). We evaluate three protocols:

1. **In-domain** — train clean → test clean. Sanity / upper-bound for the easy setting.
2. **Cross-domain (headline)** — train clean → test wild. Directly quantifies the domain gap. We expect a large drop here; closing it is the contribution.
3. **Joint** — train clean+wild → test wild. Shows the benefit of merging the datasets under the unified taxonomy.

Report both **material accuracy (6-class)** and **recyclability accuracy (binary)** for every protocol.

## 6. Closing the Domain Gap

Two complementary techniques, increasing in sophistication:
- **Domain-randomizing augmentation (baseline):** strong color jitter, random grayscale, Gaussian blur, random-resized crop. Forces the model off background cues. (Implemented in the pipeline.)
- **Domain-adversarial training (DANN):** a gradient-reversal domain classifier on top of the shared backbone so features become domain-invariant. (Hook left in the architecture; add as the "improved" condition.)
- Optional: background removal / segmentation-aware training as an ablation to directly test the shortcut-learning hypothesis.

## 7. Model

ResNet-50 (ImageNet-pretrained) shared backbone with **two heads**: a 6-class material head and a 2-class recyclability head, trained jointly with summed cross-entropy. At inference, you may either (a) trust the binary head directly, or (b) predict material then apply the deterministic material→recyclability rule. Comparing (a) vs (b) is itself a useful ablation.

## 8. Evaluation Metrics

- Material: accuracy, macro-F1, per-class F1, confusion matrix.
- Recyclability: accuracy, macro-F1, precision/recall (recall on *non-recyclable* matters most — a false "recyclable" contaminates the stream).
- Domain gap = (in-domain acc) − (cross-domain acc). Report reduction after adaptation.

## 9. Expected Narrative for the Paper

1. Naive merge fails / single-dataset models don't transfer (motivation).
2. Unified taxonomy makes the merge meaningful (Contribution 1).
3. Cross-domain numbers reveal the gap (Contribution 2, the finding).
4. Joint training + augmentation + DANN progressively close it (the result).
5. Recyclability head turns it into a deployable smart-bin classifier (impact).

## 10. Limitations to Pre-empt

- Recyclability rules vary by municipality; ours is one policy.
- Cropping discards spatial context.
- TACO class imbalance (long tail) — report per-class, consider re-weighting.
- Cropped TACO objects still differ in resolution/quality from TrashNet.

## 11. Files

- `label_mapping.csv` — the 60→6→2 ontology.
- `waste_pipeline.py` — harmonization, cropping, three protocols, training, evaluation.

Run example:
```
python waste_pipeline.py \
  --taco_images data/TACO/images --taco_ann data/TACO/annotations.json \
  --trashnet data/trashnet --protocol cross_domain --epochs 15
```
