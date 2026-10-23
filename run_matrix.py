"""
Cross-domain generalization matrix (dual head, photo-grouped TACO split)
=======================================================================
Train on clean, wild or both; every run is tested on BOTH the clean and the wild
test split. Uses manifest_grouped.csv by default (make_grouped_manifest.py).

    python run_matrix.py --train_on wild --backbones convnext_tiny --epochs 30 --patience 8 \
        --class_weighted --head_hidden 512 256 --lr 3e-4 --unfreeze two_stages \
        --backbone_lr_mult 0.3 --label_smoothing 0.1 --tta --amp --workers 4

    --wild_fraction 0.1   keep 10 % of the wild training photos (data-efficiency curve)
    --val_on wild         select on wild val even when training on both (old P3 setting)

Summarize every matrix run with:  python matrix_table.py
"""

from waste_bench.cli_matrix import main

if __name__ == "__main__":
    main()
