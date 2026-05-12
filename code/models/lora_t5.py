from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5EncoderModel


class LoRA(nn.Module):
    def __init__(self, base_module: nn.Linear, r: int = 4, alpha: float = 1.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.base = base_module
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        in_dim = base_module.in_features
        out_dim = base_module.out_features
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * ((self.dropout(x) @ self.A.T) @ self.B.T)


def add_lora(model: nn.Module, r: int = 4, alpha: float = 1.0, dropout: float = 0.05) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            child_name = name.split(".")[-1]
            if child_name in ("q", "k", "v", "o"):
                parent_name = ".".join(name.split(".")[:-1])
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRA(module, r=r, alpha=alpha, dropout=dropout))


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.blk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blk(x)


class T5LoRAClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 512,
        num_classes: int = 10,
        r: int = 4,
        alpha: float = 1.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        add_lora(self.encoder, r=r, alpha=alpha, dropout=lora_dropout)
        d_model = self.encoder.config.d_model
        self.head = MLPHead(input_dim=d_model, hidden_dim=hidden_dim, num_classes=num_classes)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return self.head(pooled)
