# Improving P1 (in_domain) — code layout and a step-by-step experiment plan

`benchmark_backbones.py` is unchanged (the previous version). The new code is
the `waste_bench/` package, run through `run_benchmark.py` with the same flags
plus the new options below. Both write to `runs/` in the same layout, and
checkpoints from either load in `predict.py`.

## 1. Code layout

```
run_benchmark.py          entry point (new)            benchmark_backbones.py  old version, untouched
compare_runs.py           one table of every run in runs/, mean ± std over seeds
waste_bench/
  constants.py            MATERIALS, MATERIAL_RECYCLABLE, ImageNet stats
  config.py               RunConfig: every option in one dataclass; run_tag() folder names
  data.py                 manifest -> (train, val, test) per protocol; transforms; loaders
  models/backbones.py     pretrained backbones; --unfreeze depth; lr param groups; BN freezing
  models/heads.py         linear / MLP / ECOC heads
  engine.py               training loop, early stopping, evaluation, TTA, mixed precision
  reporting.py            report PNG + console summary
  utils.py                logging, seeding, timing, GPU utilisation
  cli.py                  flags -> RunConfig -> train/evaluate each backbone -> save
```

Data flows one way: `cli` builds a `RunConfig` → `data` gives loaders → `models`
builds the network → `engine` trains and evaluates → `reporting` and `cli` save
outputs. To add a new option you touch `config.py` (field + tag), `cli.py`
(flag), and the one module the option belongs to.

## 2. Two things to know before comparing numbers

**Your previous in_domain numbers picked the best epoch on the test set.** The
clean data has a val split (251 images) that the old code never used; early
stopping and "restored best epoch" were decided on the 257 test images, and
then the test score on those same images was reported. That makes every old
in_domain number slightly optimistic. `run_benchmark.py` now selects on
**val** for in_domain (and joint) and uses test only for the final number.
`--select_on test` reproduces the old behaviour. cross_domain has no labelled
val split in the target domain, so it still selects on test and says so in the
log.

**The test set is small, so small differences are noise.** 1 test image =
0.39% material_acc, and there are only 15 trash images, so 1 image = 6.7%
non_recyclable_recall. For example, 0.800 vs 0.667 recall is 12/15 vs 10/15
images. Before trusting a gain, measure how much the same config moves when
only the seed changes (step 2). The log now prints this resolution line on
every run.

## 3. New options (all default to the old behaviour)

| Option | What it does | Why it may help in_domain |
|---|---|---|
| `--unfreeze none/last/stage/two_stages/all` | how much of the pretrained backbone trains (`last` = old `--finetune_head` depth) | biggest expected lever: `last` for resnet50 is only 2 small layers |
| `--backbone_lr_mult 0.1` | unfrozen backbone layers get lr × 0.1, the new head keeps the full lr | lets deep fine-tuning adjust pretrained features without wrecking them |
| `--label_smoothing 0.1` | softer targets for cross-entropy | fights over-confidence; your train acc hits 98% while val stays ~90% |
| `--freeze_bn` | keeps ImageNet BatchNorm statistics in frozen layers | by default PyTorch keeps updating them even when the layer is "frozen" |
| `--monitor val_f1` | early stopping on val macro-F1 instead of val loss | the number you actually report; needed with `--ecoc_head` |
| `--tta` | average prediction of image + its horizontal flip | usually a small free gain, no retraining needed |
| `--seed N` | repeat a config with another seed | measures noise (step 2) |
| `--amp` | mixed precision on GPU | ~1.5–2x faster; not part of the folder name |
| `--select_on val/test` | which split picks the best epoch | `test` only to reproduce old runs |

Every non-default option is added to the run folder name (e.g.
`in_domain_aug-off_ft_uf-stage_cw_lr1e-4_blr0.1_s1`), so nothing overwrites.

## 4. The plan — one change at a time, keep the winner

Work on **resnet50 only** until step 9, then apply the winning recipe to the
other backbones. At each step run the listed variants, look at the table, keep
the best, and carry it into the next step. On Kaggle put `!` in front of each
command. All steps below use `--amp` for speed except step 0.

`BASE` below means the current best flags; start with the ones from your best
run so far:

```bash
BASE="--protocol in_domain --backbones resnet50 --epochs 100 --patience 5 --finetune_head --aug_mode off --class_weighted --lr 1e-4 --lr_schedule none"
```
On Kaggle, define it as a Python variable in a cell and use `{BASE}` —
the notebook fills it into `!` commands:
```python
BASE = "--protocol in_domain --backbones resnet50 --epochs 100 --patience 5 --finetune_head --aug_mode off --class_weighted --lr 1e-4 --lr_schedule none"
!python run_benchmark.py {BASE} --amp
```
When a step picks a winner, append its flag to `BASE` so the next step builds on it.

After every step:
```bash
python compare_runs.py --protocol in_domain --backbone resnet50
```

**Rule for keeping a change:** its mean accuracy must beat the current best by
more than the seed noise from step 2 (roughly 1 std, and at least ~1%). If two
options tie, keep the simpler/faster one. Look at `acc`, `f1`, and `nr_rec`
together; don't accept a gain in accuracy that costs a lot of trash recall.

**Step 0 — check the new code reproduces the old result** (select on test, like before)
```bash
python run_benchmark.py $BASE --select_on test
```
Expect material_acc ≈ 0.92, same as your old run. If it differs a lot, stop and report it.

**Step 1 — honest baseline** (select on val)
```bash
python run_benchmark.py $BASE --amp
```
This is the number every later step has to beat. It may be a bit lower than step 0; that's expected (step 0 was optimistic).

**Step 2 — measure noise**
```bash
python run_benchmark.py $BASE --amp --seed 1
python run_benchmark.py $BASE --amp --seed 2
```
`compare_runs.py` groups the three seeds and shows `acc ± acc_std`. That std is your noise band.

**Step 3 — how much to fine-tune (expected biggest gain)**
```bash
python run_benchmark.py $BASE --amp --unfreeze stage      --backbone_lr_mult 0.1
python run_benchmark.py $BASE --amp --unfreeze two_stages --backbone_lr_mult 0.1
python run_benchmark.py $BASE --amp --unfreeze all        --backbone_lr_mult 0.1
```
Deeper is slower per epoch; check the `time` column too.

**Step 4 — augmentation again** (deeper fine-tuning overfits more, so mild aug may now win)
```bash
python run_benchmark.py <best so far> --aug_mode mild
```

**Step 5 — learning rate and schedule**
```bash
python run_benchmark.py <best> --lr_schedule cosine
python run_benchmark.py <best> --lr 3e-4
python run_benchmark.py <best> --bs 16
```

**Step 6 — regularisation**
```bash
python run_benchmark.py <best> --label_smoothing 0.1
python run_benchmark.py <best without --class_weighted>
```

**Step 7 — BatchNorm statistics** (only matters while some layers are still frozen; skip if step 3 picked `all`)
```bash
python run_benchmark.py <best> --freeze_bn
```

**Step 8 — head type**
```bash
python run_benchmark.py <best> --ecoc_head --ecoc_codebook onevsrest --monitor val_f1
python run_benchmark.py <best> --monitor val_f1
```
(Drop `--finetune_head` and keep `--unfreeze <level>` to try a plain linear head at the same depth.)

**Step 9 — test-time augmentation**
```bash
python run_benchmark.py <best> --tta
```

**Step 10 — apply the recipe to every backbone**
```bash
python run_benchmark.py <best, with --backbones resnet50 densenet121 mobilenet_v2 inception_v3>
```
`two_stages`/`all` mean the same kind of depth on each backbone (see `UNFREEZE_TARGETS` in `waste_bench/models/backbones.py`).

**Step 11 — final numbers for the paper**
Run the final recipe with `--seed 1` and `--seed 2` for every backbone and report
`acc ± acc_std` from `compare_runs.py`, not a single run.

## 5. Keeping track

```bash
python compare_runs.py --protocol in_domain --backbone resnet50 --baseline <step-1 tag> --csv p1_results.csv
```
`d_acc` shows each config's gain over the step-1 baseline. `selected_on` shows
`test (old run)` for results from `benchmark_backbones.py`; don't compare those
directly with val-selected runs.
