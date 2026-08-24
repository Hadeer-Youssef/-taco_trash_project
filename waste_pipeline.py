"""
Cross-Domain Waste Classification Pipeline  (v2 — harmonized & wired)
====================================================================
Combines TACO (wild/cluttered, COCO detection) + TrashNet (clean studio,
classification) into ONE harmonized, domain-tagged benchmark on a shared
6-class MATERIAL space + a binary recyclability label, then trains/evaluates:
    image -> material (6-class)  ->  recyclable / non_recyclable

This is the connective tissue between the three previously-disjoint pieces:
  - label_mapping.csv  (TACO 60 categories -> 6 materials -> binary recyclability)
  - update_Taco_classification/  (TACO COCO crops, here re-derived by MATERIAL)
  - last-update-trash-classify-amira notebook (TrashNet transfer-learning backbones)

KEY FIX vs the original draft
-----------------------------
The mapping is now keyed on TACO **category_id** (resolved from category NAME,
with supercategory fallback), NOT on the supercategory string. The old code keyed
on supercategory and silently dropped/mis-mapped most of the 4784 annotations.
label_mapping.csv v2 carries: taco_category_id, taco_category, taco_supercategory,
material_6class, recyclability_binary, n_annotations, mapped_via, notes.

Pipeline stages
  0. load_mapping()      -> {category_id: (material_idx, recyclable)}
  1. taco_to_crops()     -> COCO -> MATERIAL crops via try_square_crop (domain='wild')
  2. index_trashnet()    -> folder labels (domain='clean')
  3. build_manifest()    -> ONE stratified manifest.csv driving every experiment
  4. WasteDataset / WasteNet (+ optional DANN gradient-reversal head)
  5. three protocols: in_domain / cross_domain / joint
  6. evaluate both material (6-class) and recyclability (binary); report non_rec recall
  7. ablation hooks: background-removal crops + Grad-CAM

Dependencies: torch, torchvision, pandas, numpy, opencv-python, scikit-image,
              Pillow, scikit-learn
"""

import os, json, csv, argparse, random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# Reuse the project's exact cropping helpers (centroid square crop + black border,
# matching the TrashNet preprocessing) instead of a plain PIL bbox crop.
import sys
_HELPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_Taco_classification")
if _HELPER_DIR not in sys.path:
    sys.path.insert(0, _HELPER_DIR)
from cv_helper_functions import try_square_crop, mask_and_square_crop  # noqa: E402

# ----------------------------------------------------------------------------
# 1. LABEL HARMONIZATION  (key on category_id)
# ----------------------------------------------------------------------------
MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MAT2IDX = {m: i for i, m in enumerate(MATERIALS)}

# Material -> binary recyclability lookup. This is the DETERMINISTIC "material-then-rule"
# baseline used ONLY for the Phase-3 ablation. The single ADOPTED policy is the
# item-aware one stored per-category in label_mapping.csv (recyclability_binary),
# which is what the binary head is trained on.
MATERIAL_RECYCLABLE = {
    "cardboard": 1, "glass": 1, "metal": 1, "paper": 1, "plastic": 1, "trash": 0,
}


def load_mapping(csv_path, taco_ann_json=None):
    """Return dict: taco category_id -> (material_idx, is_recyclable).

    label_mapping.csv v2 already carries taco_category_id, so we key on it directly.
    If an older CSV (without the id column) is passed, fall back to resolving
    category NAME -> id against the COCO annotations (name match, then supercat).
    """
    df = pd.read_csv(csv_path)
    if "taco_category_id" in df.columns:
        m = {}
        for _, r in df.iterrows():
            mat = MAT2IDX[str(r["material_6class"]).strip()]
            rec = 1 if str(r["recyclability_binary"]).strip() == "recyclable" else 0
            m[int(r["taco_category_id"])] = (mat, rec)
        return m

    # ---- legacy fallback: resolve names against COCO ----
    assert taco_ann_json, "Old CSV without taco_category_id needs taco_ann_json to resolve."
    ann = json.load(open(taco_ann_json, encoding="utf-8"))
    name_map = {}
    for _, r in df.iterrows():
        key = str(r["taco_supercategory"]).strip().lower()
        name_map[key] = (MAT2IDX[str(r["material_6class"]).strip()],
                         1 if str(r["recyclability_binary"]).strip() == "recyclable" else 0)
    out = {}
    for c in ann["categories"]:
        nm, sp = c["name"].strip().lower(), c["supercategory"].strip().lower()
        out[c["id"]] = name_map.get(nm, name_map.get(sp, (MAT2IDX["trash"], 0)))
    return out


# ----------------------------------------------------------------------------
# 2. TACO COCO -> MATERIAL CLASSIFICATION CROPS  (domain='wild')
# ----------------------------------------------------------------------------
def taco_to_crops(taco_img_dir, taco_ann_json, mapping, out_dir,
                  min_size=32, widen_param=0.05, newsize=224, masked=False):
    """Crop each annotated TACO object with try_square_crop, label by harmonized
    MATERIAL. Writes crops to out_dir/<material>/ and returns an index list.

    masked=True multiplies the RGB crop by the segmentation mask (background -> black)
    to produce the background-removal ablation set (reuses mask_and_square_crop).
    """
    annotations = json.load(open(taco_ann_json, encoding="utf-8"))
    cat_of_ann = {a["id"]: a["category_id"] for a in annotations["annotations"]}
    bbox_of_ann = {a["id"]: a["bbox"] for a in annotations["annotations"]}
    os.makedirs(out_dir, exist_ok=True)

    index = []
    for ann_id, cat_id in cat_of_ann.items():
        if cat_id not in mapping:
            continue  # should not happen with v2 CSV (full coverage)
        mat_idx, rec = mapping[cat_id]
        w, h = bbox_of_ann[ann_id][2], bbox_of_ann[ann_id][3]
        if w < min_size or h < min_size:
            continue
        try:
            crop = try_square_crop(ann_id, annotations, widen_param, newsize, taco_img_dir)
            if masked:
                mask = mask_and_square_crop(ann_id, annotations, widen_param, newsize, taco_img_dir)
                crop = (crop * (mask > 0)).astype(np.uint8)
        except (StopIteration, AttributeError, ValueError, cv2.error):
            continue
        mat_name = MATERIALS[mat_idx]
        save_dir = os.path.join(out_dir, mat_name)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"taco_{ann_id}.jpg")
        Image.fromarray(crop).save(path)
        index.append({"path": path, "material": mat_idx, "recyclable": rec, "domain": "wild"})
    return index


# ----------------------------------------------------------------------------
# 3. TRASHNET INDEX (already classification, clean background)  domain='clean'
# ----------------------------------------------------------------------------
# Adopted policy: TrashNet has no film/foam split, so we use the material-level
# rule for its generic classes (documented asymmetry / limitation in the paper).
TRASHNET_RECYCLABLE = {
    "cardboard": 1, "glass": 1, "metal": 1, "paper": 1, "plastic": 1, "trash": 0,
}


def index_trashnet(root):
    """root has subfolders cardboard/ glass/ metal/ paper/ plastic/ trash/."""
    index = []
    for mat_name in MATERIALS:
        d = os.path.join(root, mat_name)
        if not os.path.isdir(d):
            continue
        rec = TRASHNET_RECYCLABLE[mat_name]
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index.append({"path": os.path.join(d, f), "material": MAT2IDX[mat_name],
                              "recyclable": rec, "domain": "clean"})
    return index


# ----------------------------------------------------------------------------
# 3b. MANIFEST — one stratified table that drives every experiment
# ----------------------------------------------------------------------------
def build_manifest(records, out_csv, seed=42, fracs=(0.8, 0.1, 0.1)):
    """Stratify by (domain, material) into train/val/test; write a single manifest."""
    rng = random.Random(seed)
    buckets = {}
    for r in records:
        buckets.setdefault((r["domain"], r["material"]), []).append(r)
    rows = []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * fracs[0]); n_va = int(n * fracs[1])
        for i, r in enumerate(items):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            rows.append({**r, "split": split})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "material", "recyclable", "domain", "split"])
        w.writeheader(); w.writerows(rows)
    return rows


# ----------------------------------------------------------------------------
# 4. DATASET
# ----------------------------------------------------------------------------
class WasteDataset(Dataset):
    def __init__(self, index, train=True, domain_randomize=True):
        self.index = index
        aug = []
        if train:
            aug = [transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                   transforms.RandomHorizontalFlip()]
            if domain_randomize:  # the augmentation ABLATION toggles this
                aug += [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                        transforms.RandomGrayscale(0.1),
                        transforms.GaussianBlur(3, sigma=(0.1, 2.0))]
            self.tf = transforms.Compose([transforms.Resize((256, 256)), *aug,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.tf = transforms.Compose([transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index[i]
        img = Image.open(r["path"]).convert("RGB")
        dom = 0 if r["domain"] == "clean" else 1
        return self.tf(img), r["material"], r["recyclable"], dom


# ----------------------------------------------------------------------------
# 5. MODEL: shared backbone + material head + binary head (+ optional DANN)
# ----------------------------------------------------------------------------
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class WasteNet(nn.Module):
    def __init__(self, n_materials=6, use_dann=False):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feat_dim = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.material_head = nn.Linear(self.feat_dim, n_materials)
        self.binary_head = nn.Linear(self.feat_dim, 2)
        self.use_dann = use_dann
        if use_dann:  # gradient-reversal domain classifier (clean vs wild)
            self.domain_head = nn.Sequential(
                nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x, dann_lambda=0.0):
        f = self.backbone(x)
        mat, binr = self.material_head(f), self.binary_head(f)
        if self.use_dann:
            dom = self.domain_head(grad_reverse(f, dann_lambda))
            return mat, binr, dom
        return mat, binr


# ----------------------------------------------------------------------------
# 6. TRAIN / EVAL
# ----------------------------------------------------------------------------
def run_epoch(model, loader, opt, device, train=True, dann_lambda=0.0,
              material_weight=None):
    model.train(train)
    ce = nn.CrossEntropyLoss(weight=material_weight)
    ce_plain = nn.CrossEntropyLoss()
    tot = 0.0
    for x, mat, rec, dom in loader:
        x, mat, rec, dom = x.to(device), mat.to(device), rec.to(device), dom.to(device)
        with torch.set_grad_enabled(train):
            out = model(x, dann_lambda) if model.use_dann else model(x)
            if model.use_dann:
                ml, bl, dl = out
                loss = ce(ml, mat) + ce_plain(bl, rec) + ce_plain(dl, dom)
            else:
                ml, bl = out
                loss = ce(ml, mat) + ce_plain(bl, rec)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, use_material_rule=False):
    """Report material (6-class) + recyclability (binary). If use_material_rule,
    derive recyclability from argmax material via MATERIAL_RECYCLABLE instead of
    the binary head (the direct-head vs material-then-rule ablation)."""
    model.eval()
    ym, pm, yr, pr = [], [], [], []
    rule = [MATERIAL_RECYCLABLE[m] for m in MATERIALS]
    for x, mat, rec, dom in loader:
        x = x.to(device)
        out = model(x)
        ml, bl = out[0], out[1]
        pred_mat = ml.argmax(1).cpu()
        pm += pred_mat.tolist(); ym += mat.tolist()
        if use_material_rule:
            pr += [rule[i] for i in pred_mat.tolist()]
        else:
            pr += bl.argmax(1).cpu().tolist()
        yr += rec.tolist()
    print("\n--- MATERIAL (6-class) ---")
    print(classification_report(ym, pm, target_names=MATERIALS, zero_division=0))
    print("--- RECYCLABILITY (binary) ---  [non_recyclable recall is safety-critical]")
    print(classification_report(yr, pr, target_names=["non_recyclable", "recyclable"],
                                zero_division=0))
    return {"material_cm": confusion_matrix(ym, pm).tolist(),
            "binary_cm": confusion_matrix(yr, pr).tolist()}


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def split(idx, frac=0.8, seed=42):
    random.Random(seed).shuffle(idx)
    k = int(len(idx) * frac)
    return idx[:k], idx[k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="label_mapping.csv")
    ap.add_argument("--taco_images", required=True)
    ap.add_argument("--taco_ann", required=True)
    ap.add_argument("--trashnet", required=True)
    ap.add_argument("--crops_out", default="crops_material")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--protocol", choices=["in_domain", "cross_domain", "joint"],
                    default="cross_domain")
    ap.add_argument("--masked", action="store_true", help="background-removal ablation")
    ap.add_argument("--no_domain_randomize", action="store_true", help="aug ablation")
    ap.add_argument("--dann", action="store_true")
    ap.add_argument("--material_rule", action="store_true",
                    help="derive recyclability from material rule instead of binary head")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = load_mapping(args.mapping, args.taco_ann)

    wild = taco_to_crops(args.taco_images, args.taco_ann, mapping, args.crops_out,
                         masked=args.masked)
    clean = index_trashnet(args.trashnet)
    print(f"wild(TACO) crops: {len(wild)} | clean(TrashNet): {len(clean)}")

    rows = build_manifest(wild + clean, args.manifest)
    df = pd.DataFrame(rows)
    clean_rows = df[df.domain == "clean"].to_dict("records")
    wild_rows = df[df.domain == "wild"].to_dict("records")

    def by_split(recs, s):
        return [r for r in recs if r["split"] == s]

    if args.protocol == "in_domain":
        train_idx, test_idx = by_split(clean_rows, "train"), by_split(clean_rows, "test")
    elif args.protocol == "cross_domain":
        train_idx, test_idx = clean_rows, wild_rows           # clean -> wild
    else:  # joint
        train_idx = by_split(clean_rows, "train") + by_split(wild_rows, "train")
        test_idx = by_split(wild_rows, "test")

    dr = not args.no_domain_randomize
    tr = DataLoader(WasteDataset(train_idx, True, domain_randomize=dr), args.bs,
                    shuffle=True, num_workers=4)
    te = DataLoader(WasteDataset(test_idx, False), args.bs, num_workers=4)

    model = WasteNet(use_dann=args.dann).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for e in range(args.epochs):
        lam = (2.0 / (1.0 + np.exp(-10 * e / args.epochs)) - 1.0) if args.dann else 0.0
        loss = run_epoch(model, tr, opt, device, True, dann_lambda=lam)
        print(f"epoch {e+1}/{args.epochs}  train_loss={loss:.4f}  dann_lambda={lam:.3f}")
    evaluate(model, te, device, use_material_rule=args.material_rule)
    torch.save(model.state_dict(), f"wastenet_{args.protocol}.pt")


if __name__ == "__main__":
    main()
