"""Classification heads: linear / MLP (built in backbones.py) and ECOC.

ECOC is from Wang, "Application research of image classification algorithm
based on deep learning in household garbage sorting", Heliyon 10 (2024)
e29966 — see paper_household_garbage_ecoc_capsa.md. Here it replaces the
softmax head on top of this project's pretrained backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# attribute holding each backbone's final classification layer
FINAL_LAYER_ATTR = {
    "resnet50": "fc", "densenet121": "classifier",
    "mobilenet_v2": "classifier", "inception_v3": "fc",
    "convnext_tiny": "classifier", "efficientnet_v2_s": "classifier",
}


def mlp_head(in_features, hidden, n_classes, dropout):
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )


def custom_head(in_features, hidden_sizes, n_classes, dropout=0.3):
    """Your own classifier on top of the pretrained features: one block per
    entry in hidden_sizes (Linear -> BatchNorm -> ReLU -> Dropout), then a
    final Linear to n_classes. E.g. hidden_sizes=[512, 256] on resnet50 gives
    2048 -> 512 -> 256 -> 6. BatchNorm keeps the new layers' activations on a
    stable scale while the pretrained features underneath are fixed or change
    slowly; dropout limits overfitting on ~2,000 training images."""
    layers, width = [], in_features
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, n_classes))
    return nn.Sequential(*layers)


def ordinal_codebook(n_classes):
    """The paper's scheme: L = n_classes-1 classifiers, classifier j asks "is the
    class index > j?". Adjacent classes differ by 1 bit, distant ones by many —
    which assumes the classes are ordered. Materials have no natural order, so
    this can make some class pairs artificially hard to separate."""
    L = n_classes - 1
    return torch.tensor([[1.0 if j < c else 0.0 for j in range(L)] for c in range(n_classes)])


def onevsrest_codebook(n_classes):
    """L = n_classes classifiers, one per class; every class pair differs by
    exactly 2 bits, so no pair is favoured by index order."""
    return torch.eye(n_classes, dtype=torch.float32)


CODEBOOKS = {"ordinal": ordinal_codebook, "onevsrest": onevsrest_codebook}


class EcocHead(nn.Module):
    """L small binary classifiers (1 hidden layer, 10 tanh units, as in the
    paper), each producing one logit. forward() returns (B, L) logits for BCE
    training; decode() maps them to a class by nearest code (Hamming/L1)."""

    def __init__(self, in_features, n_classes, hidden=10, codebook_type="ordinal"):
        super().__init__()
        assert codebook_type in CODEBOOKS, f"unknown codebook_type {codebook_type}"
        self.n_classes = n_classes
        self.codebook_type = codebook_type
        codebook = CODEBOOKS[codebook_type](n_classes)
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(codebook.shape[1])
        ])
        self.register_buffer("codebook", codebook)

    def forward(self, x):
        return torch.cat([clf(x) for clf in self.classifiers], dim=1)

    def decode(self, logits):
        """Returns (pred class index, per-class score); the score is a softmax
        over negative code distance, usable wherever class probabilities are
        expected (ROC curves, confidence, TTA averaging)."""
        bits = torch.sigmoid(logits.float())
        dist = torch.cdist(bits, self.codebook.to(bits.device), p=1)
        return dist.argmin(dim=1), F.softmax(-dist, dim=1)


def get_head(model, name):
    return getattr(model, FINAL_LAYER_ATTR[name.lower()])


def attach_ecoc_head(model, name, n_classes=6, codebook_type="ordinal"):
    """Replaces whichever final layer build_backbone() installed (Linear or the
    MLP) with an EcocHead fed by the same backbone features. The input width is
    the FIRST Linear's in_features (e.g. 2048 for resnet50) — the MLP's last
    Linear only sees its hidden width (e.g. 96)."""
    if name.lower() == "convnext_tiny":
        # its classifier is LayerNorm2d -> Flatten -> head; swapping the whole
        # attribute for an EcocHead would drop the norm and the flatten
        raise NotImplementedError("--ecoc_head is not supported for convnext_tiny")
    attr = FINAL_LAYER_ATTR[name.lower()]
    old_head = getattr(model, attr)
    first_linear = old_head if isinstance(old_head, nn.Linear) else \
        [m for m in old_head.modules() if isinstance(m, nn.Linear)][0]
    head = EcocHead(first_linear.in_features, n_classes, codebook_type=codebook_type)
    setattr(model, attr, head)
    return model
