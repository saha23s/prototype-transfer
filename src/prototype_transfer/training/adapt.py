"""Source-free adaptation loop for the 21-class AudioProtoPNet (methods: prototype_transfer.method.sfda).

Follows the reference loop (chirp/projects/sfda/adapt.py): only unlabeled clips; a fixed number of epochs with no
selection and no early stopping; the last incomplete batch is kept; pseudo-labels come from a clean forward pass (no
noise), either of all clips before each epoch (dropout_student, notela, shot) or of the current batch (pseudo_label).

Two things differ because the backbone is a ConvNeXt, not the authors' EfficientNet-B1:
  trainable parameters  they train BatchNorm scale and shift; this model has no BatchNorm, so the scale and shift of
                        the backbone's LayerNorms are trained. Prototypes, last layer and all other weights stay frozen.
  student noise         their network has stochastic depth with survival probability 0.8; the pretrained ConvNeXt is
                        configured with none, so a drop probability of 0.2 is switched on in every residual block for
                        the update step. Their additional dropout on the pooled embedding is NOT reproduced: here the
                        classifier reads the feature map through cosine similarity to prototypes, not a pooled vector.
The clip embedding used by notela and shot is the backbone's final feature map averaged over frequency and time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers.models.convnext.modeling_convnext import ConvNextDropPath, ConvNextLayer

from prototype_transfer.data.preprocessing import AudioPreprocessor
from prototype_transfer.method import sfda

logger = logging.getLogger(__name__)


class UnlabeledSoundscapes(Dataset):
    """(spectrogram [1, 256, 626], index) of 5 s soundscape clips. Only the audio_path column is read: no labels."""

    def __init__(self, manifest: Path | str) -> None:
        self.paths = pd.read_parquet(manifest, columns=["audio_path"])["audio_path"].tolist()
        self.preprocessor = AudioPreprocessor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.preprocessor.process_file(self.paths[i], is_soundscape=True), i


def norm_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Scale and shift of the backbone LayerNorms on the prediction path.

    The backbone's final `layernorm` only feeds `pooler_output`, which the prototype head does not read; it is left out.
    """
    params = [p for name, module in model.model.backbone.named_modules()
              if "LayerNorm" in type(module).__name__ and name != "layernorm"
              for p in (module.weight, module.bias)]
    if not params:
        raise ValueError("the backbone has no LayerNorm to adapt")
    return params


def add_stochastic_depth(model: nn.Module, drop_prob: float) -> int:
    """Set the drop probability of every ConvNeXt residual block (active in train mode only). Returns the block count.

    The modules have no parameters, so checkpoints stay loadable without them.
    """
    blocks = [m for m in model.modules() if isinstance(m, ConvNextLayer)]
    if not blocks:
        raise ValueError("no ConvNeXt blocks found")
    for block in blocks:
        block.drop_path = ConvNextDropPath(drop_prob) if drop_prob > 0 else nn.Identity()
    return len(blocks)


def _forward(model: nn.Module, x: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits [B, C] and clip embeddings [B, D] in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(input_values=x)
    return out.logits.float(), out.last_hidden_state.float().mean(dim=(-2, -1))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean (eval-mode) probabilities [N, C] and embeddings [N, D] of all clips, ordered by dataset index."""
    model.eval()
    probs, feats, index = [], [], []
    for x, i in loader:
        logits, emb = _forward(model, x.to(device), device)
        probs.append(torch.sigmoid(logits))
        feats.append(emb)
        index.append(i)
    order = torch.argsort(torch.cat(index)).to(device)
    return torch.cat(probs)[order], torch.cat(feats)[order]


def adapt(model: nn.Module, dataset: Dataset, method: str, cfg: dict, *, epochs: int, batch_size: int,
          drop_path: float, device: str, seed: int = 0, num_workers: int = 0,
          on_epoch: Callable[[dict], None] | None = None) -> list[dict]:
    """Adapt `model` in place on unlabeled clips and return one log record per epoch.

    Args:
        model: 21-class AudioProtoPNet (build_hsn_protopnet).
        dataset: yields (spectrogram, index) with indices 0..len-1.
        method: a key of sfda.METHODS.
        cfg: the method's settings (sfda.METHODS[method], possibly overridden).
        drop_path: stochastic-depth probability of the update step; used only if cfg["dropout"].

    Raises:
        FloatingPointError: on a non-finite loss or gradient, before the optimizer step.
    """
    add_stochastic_depth(model, drop_path if cfg["dropout"] else 0.0)
    params = norm_parameters(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    model.to(device)

    kw = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": device == "cuda"}
    train_dl = DataLoader(dataset, shuffle=True, generator=torch.Generator().manual_seed(seed), **kw)
    eval_dl = DataLoader(dataset, shuffle=False, **kw)
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_dl)) if cfg["cosine"] else None
    logger.info(f"{method}: {len(dataset)} unlabeled clips, {len(train_dl)} steps/epoch, {epochs} epochs; training "
                f"{sum(p.numel() for p in params)} LayerNorm parameters; settings {cfg}")

    history = []
    for epoch in range(1, epochs + 1):
        targets = None
        if method in sfda.PER_EPOCH_LABELS and (method != "shot" or cfg["beta"] != 0):
            probs, feats = predict(model, eval_dl, device)
            targets = sfda.epoch_targets(method, probs, feats, cfg)
        total, seen, label_sum, label_n = 0.0, 0, 0.0, 0
        for x, index in train_dl:
            x = x.to(device, non_blocking=True)
            batch_targets = None if targets is None else targets[index.to(targets.device)]
            if method == "pseudo_label":
                model.eval()
                with torch.no_grad():
                    batch_targets = sfda.threshold_labels(torch.sigmoid(_forward(model, x, device)[0]), cfg["threshold"])
            model.train(cfg["dropout"])   # train mode only switches stochastic depth on: the backbone has no BatchNorm
            logits, _ = _forward(model, x, device)
            loss = sfda.loss_fn(method, logits, batch_targets, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss ({float(loss)}) in epoch {epoch}; stopping")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
                raise FloatingPointError(f"non-finite gradient in epoch {epoch}; stopping before the optimizer step")
            opt.step()
            if sched is not None:
                sched.step()
            total += float(loss.detach()) * len(x)
            seen += len(x)
            if batch_targets is not None:
                label_sum += float(batch_targets.sum())
                label_n += batch_targets.numel()
        record = {"epoch": epoch, "loss": total / seen, "lr": opt.param_groups[0]["lr"],
                  # mean pseudo-label over (clip, species); for pseudo_label, the share of entries selected
                  "mean_target": label_sum / label_n if label_n else None}
        history.append(record)
        logger.info(f"[epoch {epoch}/{epochs}] " + " ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))
        if on_epoch is not None:
            on_epoch(record)
    model.eval()
    return history
