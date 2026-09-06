"""
Inference on a saved checkpoint (Phase 5)
==========================================
Loads a checkpoint written by benchmark_backbones.py's --ckpt_dir (the
best-epoch weights of one backbone/protocol run) and predicts, for one or
more images, the 6-class material and the derived recyclable/non-recyclable
state -- printing both, the same way the training-time report does.

Usage:
    python predict.py --ckpt checkpoints/cross_domain_resnet50.pt --image path/to/photo.jpg
    python predict.py --ckpt checkpoints/cross_domain_resnet50.pt --image img1.jpg img2.jpg
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from benchmark_backbones import build_backbone, attach_ecoc_head, MATERIALS, MATERIAL_RECYCLABLE


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = build_backbone(ckpt["backbone"], finetune_head=ckpt["finetune_head"])
    ecoc_head = None
    if ckpt.get("ecoc_head"):
        model = attach_ecoc_head(model, ckpt["backbone"])
        from benchmark_backbones import _FINAL_LAYER_ATTR
        ecoc_head = getattr(model, _FINAL_LAYER_ATTR[ckpt["backbone"].lower()])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    ckpt["_ecoc_head"] = ecoc_head
    return model, ckpt


def make_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_image(model, ckpt, image_path, device):
    tf = make_transform(ckpt["input_size"])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    ecoc_head = ckpt.get("_ecoc_head")
    if ecoc_head is not None:
        pred_t, probs_t = ecoc_head.decode(logits)
        pred_idx, probs = int(pred_t[0]), probs_t[0].cpu()
    else:
        probs = F.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(probs.argmax())
    materials = ckpt["materials"]
    material_recyclable = ckpt["material_recyclable"]
    recyclable = bool(material_recyclable[pred_idx])
    return {
        "image": image_path,
        "material": materials[pred_idx],
        "material_confidence": float(probs[pred_idx]),
        "recyclable": recyclable,
        "state": "recyclable" if recyclable else "non-recyclable",
        "all_material_probs": {materials[i]: float(probs[i]) for i in range(len(materials))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint saved by benchmark_backbones.py --ckpt_dir")
    ap.add_argument("--image", nargs="+", required=True, help="one or more image paths")
    ap.add_argument("--show_probs", action="store_true", help="print the full per-material probability breakdown")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    print(f"Loaded {ckpt['backbone']} ({ckpt['protocol']}, aug_mode={ckpt['aug_mode']}, "
          f"finetune_head={ckpt['finetune_head']}) -- "
          f"material_acc={ckpt['material_acc']:.3f}, recyclability_acc={ckpt['recyclability_acc']:.3f}, "
          f"non_recyclable_recall={ckpt['non_recyclable_recall']:.3f}\n")

    for image_path in args.image:
        r = predict_image(model, ckpt, image_path, device)
        print(f"{r['image']}")
        print(f"  material   : {r['material']}  (confidence {r['material_confidence']*100:.1f}%)")
        print(f"  state      : {r['state'].upper()}")
        if args.show_probs:
            for mat, p in sorted(r["all_material_probs"].items(), key=lambda kv: -kv[1]):
                print(f"    {mat:<10s} {p*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
