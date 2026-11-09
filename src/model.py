"""AudioProtoPNet wrapper for BirdSet HSN 21-species evaluation and explainability.

Wraps DBD-research-group/AudioProtoPNet-5-BirdSet-XCL:
  - Maps 21 HSN species to pretrained class indices and prototype vectors
  - Computes multi-label probabilities and class-specific prototype activations
  - Extracts spatial activation heatmaps for soundscape vs focal analysis
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger("model")

HSN_SPECIES_LIST: list[str] = [
    "gcrfin", "whcspa", "amepip", "sposan", "rocwre",
    "brebla", "daejun", "foxspa", "clanut", "moublu",
    "casfin", "mallar3", "herthr", "amerob", "yerwar",
    "yelwar", "dusfly", "mouchi", "orcwar", "warvir", "norfli",
]


class HSNProtoPNetClassifier(nn.Module):
    """Wrapper around AudioProtoNetForSequenceClassification specialized for HSN 21 species."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
        device: str | torch.device = "cpu",
        species_list: Sequence[str] = HSN_SPECIES_LIST,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.species_list = list(species_list)
        self.num_target_classes = len(self.species_list)

        logger.info(f"Loading pretrained AudioProtoPNet from {pretrained_model_name_or_path}...")
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            config=self.config,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.prototypes_per_class = self.config.prototypes_per_class  # 5
        self.id2label = self.config.id2label
        self.label2id = {v: int(k) for k, v in self.id2label.items()}

        # Map each of our 21 species to full model class ID and prototype IDs
        self.species_to_global_class: dict[str, int] = {}
        self.global_class_to_species: dict[int, str] = {}
        self.target_class_indices: list[int] = []
        self.target_prototype_indices: list[int] = []
        self.species_prototype_map: dict[str, list[int]] = {}

        for local_idx, sp in enumerate(self.species_list):
            if sp not in self.label2id:
                raise ValueError(f"Species '{sp}' not found in pretrained model vocabulary!")
            global_cls = self.label2id[sp]
            self.species_to_global_class[sp] = global_cls
            self.global_class_to_species[global_cls] = sp
            self.target_class_indices.append(global_cls)

            # Each class c has 5 prototypes: [5c, 5c+1, 5c+2, 5c+3, 5c+4]
            proto_start = global_cls * self.prototypes_per_class
            proto_ids = list(range(proto_start, proto_start + self.prototypes_per_class))
            self.species_prototype_map[sp] = proto_ids
            self.target_prototype_indices.extend(proto_ids)

        self.target_class_tensor = torch.tensor(self.target_class_indices, dtype=torch.long, device=self.device)
        self.target_prototype_tensor = torch.tensor(self.target_prototype_indices, dtype=torch.long, device=self.device)

        logger.info(
            f"Initialized HSN classifier with {self.num_target_classes} target species and "
            f"{len(self.target_prototype_indices)} prototypes ({self.prototypes_per_class} per class)."
        )

    @torch.no_grad()
    def forward(
        self,
        input_values: torch.Tensor,
        return_heatmaps: bool = True,
    ) -> dict[str, Any]:
        """Forward pass for a batch of spectrograms.

        Args:
            input_values: Tensor of shape (batch, 1, 256, 626).
            return_heatmaps: Whether to extract spatial prototype activation maps.

        Returns:
            Dictionary containing:
              - 'logits': Tensor (batch, 21) containing raw logits for HSN species
              - 'probs': Tensor (batch, 21) sigmoid probabilities
              - 'all_logits': Tensor (batch, 9736) all model logits
              - 'prototype_activations': Tensor (batch, 21, 5, 16, 38) activation heatmaps
              - 'prototype_max_activations': Tensor (batch, 21, 5) pooled prototype scores
              - 'class_max_activations': Tensor (batch, 21) max prototype score per class
              - 'conv_features': Tensor (batch, 1024, 8, 19) ConvNeXt last hidden state
        """
        input_values = input_values.to(self.device)
        output = self.model(
            input_values=input_values,
            output_prototypical_activations=True,
            output_hidden_states=None,
        )

        all_logits = output.logits  # (B, 9736)
        hsn_logits = all_logits[:, self.target_class_tensor]  # (B, 21)
        hsn_probs = torch.sigmoid(hsn_logits)  # (B, 21)

        result: dict[str, Any] = {
            "logits": hsn_logits,
            "probs": hsn_probs,
            "all_logits": all_logits,
            "conv_features": output.last_hidden_state,
        }

        if output.prototype_activations is not None:
            # Full activations: (B, 48680, H, W) where H=16, W=38
            all_proto_act = output.prototype_activations
            B, _, H, W = all_proto_act.shape

            # Slice to 105 HSN prototypes: (B, 105, H, W) -> reshape to (B, 21, 5, H, W)
            hsn_proto_act = all_proto_act[:, self.target_prototype_tensor, :, :]
            hsn_proto_act = hsn_proto_act.view(B, self.num_target_classes, self.prototypes_per_class, H, W)

            # Max pooling over spatial dimensions: (B, 21, 5)
            proto_max = torch.amax(hsn_proto_act, dim=(-2, -1))
            # Max over 5 prototypes per class: (B, 21)
            class_max = torch.amax(proto_max, dim=-1)

            result["prototype_max_activations"] = proto_max
            result["class_max_activations"] = class_max

            if return_heatmaps:
                result["prototype_activations"] = hsn_proto_act

        return result
