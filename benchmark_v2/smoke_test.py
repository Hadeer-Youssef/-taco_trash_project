"""
Smoke test for benchmark_v2: manifest -> splits -> DataLoaders -> real batches.
Run from the REPO ROOT (manifest paths are repo-relative):

    python benchmark_v2/smoke_test.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_v2.waste_bench.data import load_manifest, select_rows, make_loader  # noqa: E402

BATCHES = 2
BS = 8


def check_protocol(df, protocol):
    tr, va, te = select_rows(df, protocol)
    print(f"\n=== {protocol} ===")
    print(f"  splits: train={len(tr)}  val={len(va)}  test={len(te)}")

    # photo integrity across protocols: wild val/test photos must not appear in any train row
    wild_train_photos = {r["photo"] for r in select_rows(df, "joint")[0] if r["domain"] == "wild"}
    leaked = sum(1 for r in va + te if r["domain"] == "wild" and r["photo"] in wild_train_photos)
    print(f"  wild val/test crops sharing a photo with wild train: {leaked}  (v1: ~80.8%)")

    loader = make_loader(tr, 224, True, "strong", BS, num_workers=0)
    for i, (x, y, rec) in enumerate(loader):
        if i >= BATCHES:
            break
        print(f"  train batch {i + 1}: images {tuple(x.shape)}, material {y.tolist()}, rec {rec.tolist()}")

    # test loader unshuffled, eval-mode, no augmentation
    te_loader = make_loader(te, 224, False, "strong", BS, num_workers=0)
    x, y, rec = next(iter(te_loader))
    assert tuple(x.shape)[1:] == (3, 224, 224)
    assert set(y.tolist()) <= set(range(6)) and set(rec.tolist()) <= {0, 1}
    print(f"  test batch ok: {tuple(x.shape)}, rec mean={sum(rec.tolist()) / len(rec):.2f}")
    return len(tr), len(va), len(te)


def main():
    df = load_manifest(os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_v2.csv"))
    print(f"manifest_v2: {len(df)} rows, columns: {list(df.columns)}")
    for proto in ("in_domain", "cross_domain", "joint"):
        check_protocol(df, proto)
    assert (df.groupby("photo").split.nunique() == 1).all(), "photo leaked across splits"
    amb = df.ambiguous.mean()
    print(f"\nALL SMOKE CHECKS PASSED — manifest intact, loaders working, photo integrity verified.")
    print(f"ambiguous-labelled crops share: {amb:.1%}")


if __name__ == "__main__":
    main()
