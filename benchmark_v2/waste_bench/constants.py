"""Label space and fixed numbers shared by every module."""

import torch

MATERIALS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
N_CLASSES = len(MATERIALS)

# Rule used to derive recyclability from a *predicted* material (index-aligned
# to MATERIALS). The model never predicts recyclability directly.
MATERIAL_RECYCLABLE = [1, 1, 1, 1, 1, 0]

BACKBONES = ["resnet50", "densenet121", "mobilenet_v2", "inception_v3"]
# every backbone --backbones accepts; BACKBONES stays the default set so a
# command that does not name backbones behaves as before
ALL_BACKBONES = BACKBONES + ["convnext_tiny", "efficientnet_v2_s"]

IMAGENET_MEAN_LIST = [0.485, 0.456, 0.406]
IMAGENET_STD_LIST = [0.229, 0.224, 0.225]
IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN_LIST).view(3, 1, 1)
IMAGENET_STD = torch.tensor(IMAGENET_STD_LIST).view(3, 1, 1)
