from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import T5EncoderModel


@dataclass
class PrefixModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class PrefixTunedProT5(nn.Module):
    """
    Prefix tuning implemented directly on encoder inputs.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        prefix_length: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = T5EncoderModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.d_model
        self.prefix_length = prefix_length

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.prefix_embeddings = nn.Parameter(torch.empty(prefix_length, self.hidden_size))
        nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> PrefixModelOutput:
        batch_size = input_ids.size(0)
        device = input_ids.device

        token_embed = self.backbone.get_input_embeddings()(input_ids)
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prefix, token_embed], dim=1)

        prefix_mask = torch.ones(batch_size, self.prefix_length, dtype=attention_mask.dtype, device=device)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        token_hidden = outputs.last_hidden_state[:, self.prefix_length :, :]

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return PrefixModelOutput(loss=loss, logits=logits)
