"""
Dual-head benchmark: material (6 classes) + recyclability (2 classes) from one backbone
=======================================================================================
Same flags as run_benchmark.py plus --rec_weight. run_benchmark.py is unchanged and
still produces the single-head runs; dual runs are written to runs/<backbone>/<tag>_dual/.

    python run_dual.py --protocol joint --backbones convnext_tiny --epochs 30 --patience 8 \
        --class_weighted --head_hidden 512 256 --lr 3e-4 --unfreeze two_stages \
        --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --amp --workers 4
"""

from waste_bench.cli_dual import main

if __name__ == "__main__":
    main()
