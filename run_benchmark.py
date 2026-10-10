"""
Cross-domain backbone benchmark — new entry point (waste_bench package)
=======================================================================
Same flags as benchmark_backbones.py plus the new P1 options (--unfreeze,
--backbone_lr_mult, --label_smoothing, --freeze_bn, --monitor, --select_on,
--seed, --tta, --amp). benchmark_backbones.py is left unchanged as the
previous version; both write to runs/ in the same layout, and checkpoints
from either load in predict.py.

    python run_benchmark.py --protocol in_domain --backbones resnet50 --epochs 15

Implementation: see waste_bench/__init__.py for the module layout.

Protocols:
  P1 in_domain    : train clean/train -> select on clean/val -> test clean/test
  P2 cross_domain : train clean/all   -> test wild/all   <-- headline
  P3 joint        : train clean+wild/train -> select on wild/val -> test wild/test
"""

from waste_bench.cli import main

if __name__ == "__main__":
    main()
