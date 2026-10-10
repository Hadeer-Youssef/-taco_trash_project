"""
waste_bench — cross-domain waste classification benchmark
=========================================================
Package layout (each module owns one concern):

  constants.py        label space (MATERIALS, MATERIAL_RECYCLABLE), ImageNet stats
  config.py           RunConfig: every CLI option in one dataclass + run_tag()
  data.py             manifest -> (train, val, test) rows, transforms, DataLoaders
  models/backbones.py pretrained backbones + how deep to unfreeze
  models/heads.py     classification heads: linear / MLP / ECOC
  engine.py           training loop, early stopping, evaluation, TTA, AMP
  reporting.py        per-model PNG report + console summary
  utils.py            logging, seeding, timing, GPU utilization sampling
  cli.py              argparse -> RunConfig -> run every backbone -> save outputs

Entry points: `python benchmark_backbones.py ...` (unchanged CLI) and
`python compare_runs.py` (tabulates everything under runs/).
"""
