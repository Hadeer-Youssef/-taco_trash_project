"""
Inference on a saved checkpoint — new version (waste_bench package)
===================================================================
Like predict.py, but rebuilds the model from everything the checkpoint
records (custom head widths, unfreeze depth, ECOC codebook), so it also
loads checkpoints trained with --head_hidden. predict.py is unchanged and
still works for checkpoints without a custom head.

    python run_predict.py --ckpt runs/resnet50/<tag>/checkpoint.pt --image photo.jpg [--show_probs]
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from waste_bench.data import build_transform
from waste_bench.models import build_backbone, attach_ecoc_head, get_head


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt["backbone"]
    model, _ = build_backbone(name, finetune_head=ckpt.get("finetune_head", False),
                              unfreeze=ckpt.get("unfreeze"),
                              head_hidden=ckpt.get("head_hidden"),
                              head_dropout=ckpt.get("head_dropout", 0.3))
    ecoc_head = None
    if ckpt.get("ecoc_head"):
        model = attach_ecoc_head(model, name, codebook_type=ckpt.get("ecoc_codebook", "ordinal"))
        ecoc_head = get_head(model, name)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt, ecoc_head


@torch.no_grad()
def predict_image(model, ckpt, ecoc_head, image_path, device):
    x = build_transform(ckpt["input_size"], train=False)(Image.open(image_path).convert("RGB"))
    out = model(x.unsqueeze(0).to(device))
    logits = out[0] if isinstance(out, tuple) else out
    if ecoc_head is not None:
        pred, probs = ecoc_head.decode(logits)
        pred, probs = int(pred[0]), probs[0].cpu()
    else:
        probs = F.softmax(logits, dim=1)[0].cpu()
        pred = int(probs.argmax())
    materials = ckpt["materials"]
    recyclable = bool(ckpt["material_recyclable"][pred])
    return {"material": materials[pred], "confidence": float(probs[pred]),
            "state": "recyclable" if recyclable else "non-recyclable",
            "probs": {materials[i]: float(probs[i]) for i in range(len(materials))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--show_probs", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt, ecoc_head = load_checkpoint(args.ckpt, device)
    print(f"Loaded {ckpt['backbone']} ({ckpt.get('tag', ckpt['protocol'])}) -- "
          f"material_acc={ckpt['material_acc']:.3f}, "
          f"non_recyclable_recall={ckpt['non_recyclable_recall']:.3f}\n")
    for path in args.image:
        r = predict_image(model, ckpt, ecoc_head, path, device)
        print(path)
        print(f"  material   : {r['material']}  (confidence {r['confidence']*100:.1f}%)")
        print(f"  state      : {r['state'].upper()}")
        if args.show_probs:
            for mat, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
                print(f"    {mat:<10s} {p*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
