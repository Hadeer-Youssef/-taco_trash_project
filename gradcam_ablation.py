"""
Shortcut-learning evidence (Phase 3, the "money figure")
========================================================
1. Generate the background-removal ablation crop set (RGB * segmentation mask)
   by reusing the project's try_square_crop + mask_and_square_crop, producing a
   parallel manifest (crops_material_masked/ + manifest_masked.csv).
2. Grad-CAM a clean-trained model on WILD crops, before vs after background
   removal, to show attention shifting off the background.

If cross-domain accuracy rises on the masked set, that is causal evidence the
clean->wild gap is a background shortcut, not a material-recognition failure.

Run generate first (CPU ok), then gradcam on a GPU box with a trained checkpoint:
    python gradcam_ablation.py generate
    python gradcam_ablation.py gradcam --ckpt resnet50_clean.pt --n 8
"""
import os, sys, json, argparse, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "update_Taco_classification"))
from cv_helper_functions import try_square_crop, mask_and_square_crop  # noqa

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def generate_masked(taco_img_dir, taco_ann, mapping_csv, out_dir, manifest_out,
                    min_size=32, widen=0.05, size=224):
    import waste_pipeline as wp
    mapping = wp.load_mapping(mapping_csv, taco_ann)
    annotations = json.load(open(taco_ann, encoding="utf-8"))
    cat_of = {a["id"]: a["category_id"] for a in annotations["annotations"]}
    bbox_of = {a["id"]: a["bbox"] for a in annotations["annotations"]}
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for ann_id, cid in cat_of.items():
        if cid not in mapping:
            continue
        mat_idx, rec = mapping[cid]
        w, h = bbox_of[ann_id][2], bbox_of[ann_id][3]
        if w < min_size or h < min_size:
            continue
        try:
            crop = try_square_crop(ann_id, annotations, widen, size, taco_img_dir)
            mask = mask_and_square_crop(ann_id, annotations, widen, size, taco_img_dir)
            crop = (crop * (mask > 0)).astype(np.uint8)
        except Exception:
            continue
        d = os.path.join(out_dir, MATERIALS[mat_idx]); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"taco_{ann_id}.jpg")
        Image.fromarray(crop).save(p)
        rows.append({"path": p, "material": mat_idx, "recyclable": rec,
                     "domain": "wild", "split": "test"})
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["path", "material", "recyclable", "domain", "split"])
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"masked crops: {len(rows)} -> {manifest_out}")


def _load_resnet(ckpt, device):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    return m.to(device).eval()


def gradcam(ckpt, n, device):
    """Minimal Grad-CAM on ResNet50 layer4, masked vs unmasked wild crops."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    model = _load_resnet(ckpt, device)
    target = model.layer4[-1]
    acts, grads = {}, {}
    target.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
    target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

    df = pd.read_csv("manifest.csv")
    wild = df[df.domain == "wild"].sample(n, random_state=0)
    dfm = pd.read_csv("manifest_masked.csv")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), _NORM])

    def cam_for(path):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        out = model(x); cls = out.argmax(1)
        model.zero_grad(); out[0, cls].backward()
        w = grads["v"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts["v"]).sum(1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
        return cam, int(cls)

    fig, axes = plt.subplots(n, 4, figsize=(8, n * 2))
    for i, (_, r) in enumerate(wild.iterrows()):
        orig = np.array(Image.open(r["path"]).convert("RGB").resize((224, 224)))
        cam, cls = cam_for(r["path"])
        axes[i, 0].imshow(orig); axes[i, 0].set_title("wild", fontsize=7)
        axes[i, 1].imshow(orig); axes[i, 1].imshow(_up(cam), alpha=0.5, cmap="jet")
        axes[i, 1].set_title(f"CAM->{MATERIALS[cls]}", fontsize=7)
        mp = dfm[dfm.path.str.contains(os.path.basename(r["path"]))]
        if len(mp):
            mpath = mp.iloc[0]["path"]
            mimg = np.array(Image.open(mpath).convert("RGB").resize((224, 224)))
            mcam, mcls = cam_for(mpath)
            axes[i, 2].imshow(mimg); axes[i, 2].set_title("bg-removed", fontsize=7)
            axes[i, 3].imshow(mimg); axes[i, 3].imshow(_up(mcam), alpha=0.5, cmap="jet")
            axes[i, 3].set_title(f"CAM->{MATERIALS[mcls]}", fontsize=7)
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    fig.suptitle("Grad-CAM: clean-trained model on wild crops, before/after background removal")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("fig_gradcam.png", dpi=130)
    print("wrote fig_gradcam.png")


def _up(cam):
    from PIL import Image as I
    return np.array(I.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--taco_images", default="Taco_data")
    g.add_argument("--taco_ann", default="Taco_data/annotations.json")
    g.add_argument("--mapping", default="label_mapping.csv")
    g.add_argument("--out", default="crops_material_masked")
    g.add_argument("--manifest_out", default="manifest_masked.csv")
    c = sub.add_parser("gradcam")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "generate":
        generate_masked(args.taco_images, args.taco_ann, args.mapping, args.out, args.manifest_out)
    else:
        gradcam(args.ckpt, args.n, dev)
