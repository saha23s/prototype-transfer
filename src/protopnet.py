"""21-class HSN AudioProtoPNet: build from the pretrained XCL model, save/load, losses, push.

The model is the original AudioProtoNetForSequenceClassification (remote code, pinned revision) with
config.num_classes = 21, initialised from the pretrained checkpoint:
  backbone       <- pretrained ConvNeXt backbone
  prototypes     <- the 5 prototypes of each HSN species (105 total)
  last layer     <- the matching rows of the class-specific, non-negative last layer (+ bias)
so the trained prototypes stay comparable with the pretrained ones.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.model import HSN_SPECIES_LIST, MODEL_REVISION

logger = logging.getLogger("protopnet")
PRETRAINED = "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL"


def hsn_config(revision: str = MODEL_REVISION):
    cfg = AutoConfig.from_pretrained(PRETRAINED, revision=revision, trust_remote_code=True)
    cfg.num_classes = len(HSN_SPECIES_LIST)
    cfg.id2label = {i: sp for i, sp in enumerate(HSN_SPECIES_LIST)}
    cfg.label2id = {sp: i for i, sp in enumerate(HSN_SPECIES_LIST)}
    return cfg


def build_hsn_protopnet(revision: str = MODEL_REVISION) -> nn.Module:
    """21-class model initialised from the pretrained 9736-class XCL model."""
    full = AutoModelForSequenceClassification.from_pretrained(PRETRAINED, revision=revision, trust_remote_code=True)
    cfg = hsn_config(revision)
    model = AutoModelForSequenceClassification.from_config(cfg, trust_remote_code=True)
    model.model.load_state_dict(full.model.state_dict())
    gid = [int(full.config.label2id[sp]) for sp in HSN_SPECIES_LIST]
    K = full.config.prototypes_per_class
    pidx = torch.tensor([g * K + k for g in gid for k in range(K)])
    with torch.no_grad():
        model.head.prototype_vectors.copy_(full.head.prototype_vectors[pidx])
        model.head.last_layer.weight.copy_(full.head.last_layer.weight[gid])
        model.head.last_layer.bias.copy_(full.head.last_layer.bias[gid])
    del full
    return model


def save_checkpoint(model: nn.Module, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "species": HSN_SPECIES_LIST,
                "revision": MODEL_REVISION, "extra": extra or {}}, path)


def load_checkpoint(path: Path | str, device: str = "cpu") -> nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["species"] == HSN_SPECIES_LIST, "species order mismatch"
    model = AutoModelForSequenceClassification.from_config(hsn_config(ck["revision"]), trust_remote_code=True)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------------------
def orthogonality_loss(head: nn.Module) -> torch.Tensor:
    """||P_c P_c^T - I||_F summed over classes, P_c = L2-normalised prototypes of class c.
    (The remote get_prototype_orthogonalities references a missing attribute, so we compute it here.)"""
    K = head.prototypes_per_class
    P = nn.functional.normalize(head.prototype_vectors.flatten(1), dim=-1).view(head.num_classes, K, -1)
    G = P @ P.transpose(1, 2) - torch.eye(K, device=P.device)
    return torch.linalg.matrix_norm(G).sum()


def param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    return {
        "backbone": list(model.model.parameters()),
        "prototypes": [model.head.prototype_vectors],
        "last_layer": list(model.head.last_layer.parameters()),
    }


def set_trainable(model: nn.Module, groups: set[str]) -> None:
    for name, params in param_groups(model).items():
        for p in params:
            p.requires_grad_(name in groups)


# ---------------------------------------------------------------------------
@torch.no_grad()
def push_prototypes(model: nn.Module, dataset: Dataset, device: str, batch_size: int = 32,
                    num_workers: int = 8) -> list[dict[str, Any]]:
    """Project every prototype onto its most similar latent patch among training clips of its class
    (standard ProtoPNet push). Returns, per prototype, the source clip / grid cell / activation."""
    model.eval()
    head = model.head
    C, K = head.num_classes, head.prototypes_per_class
    best_act = torch.full((C * K,), -1.0)
    best_vec = head.prototype_vectors.detach().clone().cpu()
    best_src: list[dict | None] = [None] * (C * K)
    cls_of = torch.arange(C * K) // K
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in tqdm(loader, desc="push"):
        x = batch["spectrogram"].to(device)
        y = batch["target"] > 0.5                                   # (B, C)
        out = model(input_values=x, output_prototypical_activations=True)
        acts = out.prototype_activations.float().cpu()             # (B, P, H, W)
        feats = head.add_on_layers(out.last_hidden_state).float().cpu()  # (B, D, H, W)
        B, P, H, W = acts.shape
        valid = y[:, cls_of]                                        # (B, P): clip contains prototype's class
        flat = acts.view(B, P, -1).masked_fill(~valid[..., None], -2.0)
        val, arg = flat.max(dim=-1)                                 # (B, P)
        bval, bclip = val.max(dim=0)                                # (P,)
        for p in torch.nonzero(bval > best_act).flatten().tolist():
            b, a = bclip[p].item(), arg[bclip[p], p].item()
            r, c = divmod(a, W)
            best_act[p] = bval[p]
            best_vec[p] = feats[b, :, r, c].view_as(best_vec[p])
            best_src[p] = {"audio_path": batch["audio_path"][b], "row": r, "col": c, "activation": float(bval[p])}
    pushed = best_act > -1
    with torch.no_grad():
        head.prototype_vectors.copy_(torch.where(pushed[:, None, None, None], best_vec, head.prototype_vectors.cpu()).to(device))
    logger.info(f"Pushed {int(pushed.sum())}/{C * K} prototypes (others had no training positives)")
    return [{"prototype": p, "species": HSN_SPECIES_LIST[p // K], "k": p % K + 1, **(best_src[p] or {"pushed": False})}
            for p in range(C * K)]
