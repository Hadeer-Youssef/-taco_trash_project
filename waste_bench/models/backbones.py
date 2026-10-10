"""Pretrained backbones, how deep to unfreeze them, and optimizer param groups."""

import torch.nn as nn
from torchvision import models

from .heads import FINAL_LAYER_ATTR, mlp_head, custom_head

INPUT_SIZE = {"resnet50": 224, "densenet121": 224, "mobilenet_v2": 224, "inception_v3": 299}

# Modules to unfreeze at each depth, from the output side inwards. "none" trains
# only the head (linear probe); "all" fine-tunes every layer.
#   last       : the setting --finetune_head has always used
#   stage      : the whole last resolution stage
#   two_stages : the last two stages
UNFREEZE_TARGETS = {
    "resnet50": {
        "last": lambda m: [m.layer4[2].conv1, m.layer4[2].bn1],
        "stage": lambda m: [m.layer4],
        "two_stages": lambda m: [m.layer3, m.layer4],
    },
    "densenet121": {
        "last": lambda m: [m.features.denseblock4, m.features.norm5],
        "stage": lambda m: [m.features.denseblock4, m.features.norm5],  # same as last
        "two_stages": lambda m: [m.features.denseblock3, m.features.transition3,
                                 m.features.denseblock4, m.features.norm5],
    },
    "mobilenet_v2": {
        "last": lambda m: [m.features[i] for i in (16, 17, 18)],
        "stage": lambda m: [m.features[i] for i in range(14, 19)],
        "two_stages": lambda m: [m.features[i] for i in range(11, 19)],
    },
    "inception_v3": {
        "last": lambda m: [m.Mixed_7c],
        "stage": lambda m: [m.Mixed_7a, m.Mixed_7b, m.Mixed_7c],
        "two_stages": lambda m: [m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
                                 m.Mixed_7a, m.Mixed_7b, m.Mixed_7c],
    },
}

# MLP head (hidden width, dropout) used when finetune_head is on
MLP_HEAD = {"resnet50": (96, 0.5), "densenet121": (256, 0.3),
            "mobilenet_v2": (256, 0.3), "inception_v3": (256, 0.3)}


def _load_pretrained(name):
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if name == "densenet121":
        return models.densenet121(weights="IMAGENET1K_V1")
    if name == "mobilenet_v2":
        return models.mobilenet_v2(weights="IMAGENET1K_V1")
    if name == "inception_v3":
        return models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    raise ValueError(f"unknown backbone {name}")


def _feature_width(m, name):
    head = getattr(m, FINAL_LAYER_ATTR[name])
    return head[1].in_features if name == "mobilenet_v2" else head.in_features


def build_backbone(name, n_classes=6, freeze_features=True, finetune_head=False, unfreeze=None,
                   head_hidden=None, head_dropout=0.3):
    """ImageNet-pretrained backbone with a fresh n_classes head.

    finetune_head : MLP head (in -> hidden -> n_classes) instead of a linear one
    head_hidden   : list of hidden widths for a custom head (see
                    heads.custom_head); takes precedence over finetune_head's
                    fixed MLP. None keeps the previous heads.
    unfreeze      : none | last | stage | two_stages | all ; None means
                    "last" with finetune_head and "none" without it, which is
                    exactly what earlier versions of this script did.
    Returns (model, input_size)."""
    name = name.lower()
    m = _load_pretrained(name)
    if freeze_features:
        for p in m.parameters():
            p.requires_grad = False

    in_features = _feature_width(m, name)
    if head_hidden:
        head = custom_head(in_features, head_hidden, n_classes, head_dropout)
    elif finetune_head:
        hidden, dropout = MLP_HEAD[name]
        head = mlp_head(in_features, hidden, n_classes, dropout)
    elif name == "mobilenet_v2":
        head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, n_classes))
    else:
        head = nn.Linear(in_features, n_classes)
    setattr(m, FINAL_LAYER_ATTR[name], head)
    if name == "inception_v3" and m.AuxLogits is not None:
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, n_classes)

    level = unfreeze or ("last" if finetune_head else "none")
    if level == "all":
        for p in m.parameters():
            p.requires_grad = True
    elif level != "none":
        for module in UNFREEZE_TARGETS[name][level](m):
            for p in module.parameters():
                p.requires_grad = True
    return m, INPUT_SIZE[name]


def _head_modules(model, name):
    mods = [getattr(model, FINAL_LAYER_ATTR[name.lower()])]
    if name.lower() == "inception_v3" and model.AuxLogits is not None:
        mods.append(model.AuxLogits.fc)
    return mods


def param_groups(model, name, lr, backbone_lr_mult=1.0):
    """Optimizer groups: the freshly initialised head trains at `lr`, unfrozen
    pretrained layers at `lr * backbone_lr_mult`. A multiplier < 1 (e.g. 0.1)
    lets deep fine-tuning adapt pretrained features gently instead of wiping
    them out with a learning rate sized for a random head."""
    head_ids = {id(p) for mod in _head_modules(model, name) for p in mod.parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": lr}]
    if body:
        groups.append({"params": body, "lr": lr * backbone_lr_mult})
    return groups


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_train_mode(model, freeze_bn=False):
    """model.train(), optionally keeping BatchNorm layers whose weights are
    frozen in eval mode. By default PyTorch still updates the running
    mean/variance of a "frozen" BN layer on every training batch, so the
    frozen backbone is not really frozen; freeze_bn keeps ImageNet statistics."""
    model.train()
    if freeze_bn:
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and \
                    not any(p.requires_grad for p in mod.parameters()):
                mod.eval()
