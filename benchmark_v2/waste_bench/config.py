"""RunConfig: every option of a benchmark run in one place, plus run_tag()."""

from dataclasses import dataclass, field, asdict

from .constants import BACKBONES

PROTOCOLS_WITH_VAL = ("in_domain", "taco_only", "joint", "cross_domain")
UNFREEZE_LEVELS = ["auto", "none", "last", "stage", "two_stages", "all"]


@dataclass
class RunConfig:
    manifest: str = "manifest_v2.csv"
    protocol: str = "cross_domain"
    backbones: list = field(default_factory=lambda: list(BACKBONES))
    select_on: str = "auto"

    finetune_head: bool = False
    head_hidden: list = None
    head_dropout: float = 0.3
    unfreeze: str = "auto"
    freeze_bn: bool = False
    ecoc_head: bool = False
    ecoc_codebook: str = "ordinal"

    epochs: int = 15
    bs: int = 32
    lr: float = 3e-4
    lr_schedule: str = "cosine"
    backbone_lr_mult: float = 1.0
    label_smoothing: float = 0.0
    class_weighted: bool = False
    patience: int = None
    monitor: str = "val_loss"
    aug_mode: str = "auto"
    seed: int = 42
    amp: bool = False
    workers: int = 2

    tta: bool = False

    runs_dir: str = "benchmark_v2/runs"
    out: str = None
    report_dir: str = None
    log_file: str = None
    ckpt_dir: str = None
    no_report: bool = False
    no_ckpt: bool = False

    def resolved_aug_mode(self):
        if self.aug_mode == "auto":
            return "mild" if self.protocol in ("in_domain", "taco_only") else "strong"
        return self.aug_mode

    def resolved_select_on(self):
        if self.select_on != "auto":
            return self.select_on
        return "val" if self.protocol in PROTOCOLS_WITH_VAL else "test"

    def resolved_unfreeze(self):
        if self.unfreeze != "auto":
            return self.unfreeze
        return "last" if self.finetune_head else "none"

    def to_dict(self):
        d = asdict(self)
        d["aug_mode_resolved"] = self.resolved_aug_mode()
        d["select_on_resolved"] = self.resolved_select_on()
        d["unfreeze_resolved"] = self.resolved_unfreeze()
        return d


def _fmt_float(x):
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")


def run_tag(cfg: RunConfig):
    parts = [cfg.protocol, f"aug-{cfg.resolved_aug_mode()}"]
    if cfg.finetune_head:
        parts.append("ft")
    if cfg.head_hidden:
        parts.append("h" + "-".join(str(h) for h in cfg.head_hidden))
        if cfg.head_dropout != 0.3:
            parts.append(f"hd{cfg.head_dropout:g}")
    if cfg.unfreeze != "auto":
        parts.append(f"uf-{cfg.unfreeze}")
    if cfg.class_weighted:
        parts.append("cw")
    if cfg.ecoc_head:
        parts.append("ecoc" if cfg.ecoc_codebook == "ordinal" else f"ecoc-{cfg.ecoc_codebook}")
    if cfg.freeze_bn:
        parts.append("fbn")
    if cfg.lr_schedule != "cosine":
        parts.append(f"lr-{cfg.lr_schedule}")
    if cfg.lr != 3e-4:
        parts.append(f"lr{_fmt_float(cfg.lr)}")
    if cfg.backbone_lr_mult != 1.0:
        parts.append(f"blr{cfg.backbone_lr_mult:g}")
    if cfg.bs != 32:
        parts.append(f"bs{cfg.bs}")
    if cfg.label_smoothing:
        parts.append(f"ls{cfg.label_smoothing:g}")
    if cfg.monitor != "val_loss":
        parts.append("mon-f1")
    if cfg.select_on != "auto":
        parts.append(f"sel-{cfg.select_on}")
    if cfg.tta:
        parts.append("tta")
    if cfg.seed != 42:
        parts.append(f"s{cfg.seed}")
    return "_".join(parts)
