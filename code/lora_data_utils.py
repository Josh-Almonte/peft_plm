from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer


class ProteinDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = (input_ids != 0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_mask[idx],
        }


def _load_pkl(data_path: str | Path, split: str) -> pd.DataFrame:
    path = os.path.join(data_path, split) + ".pkl"
    return pd.read_pickle(path)


def _tokenize_split(df: pd.DataFrame, tokenizer: T5Tokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [" ".join(list(s)) for s in df[df.columns[0]].tolist()]
    input_ids = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    labels = torch.tensor(df["loc_num"].tolist())
    return input_ids, labels


def build_lora_dataloaders(
    data_path: str | Path,
    tokenizer: T5Tokenizer,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_df = _load_pkl(data_path, "train")
    val_df = _load_pkl(data_path, "valid")

    train_ids, train_labels = _tokenize_split(train_df, tokenizer, max_length)
    val_ids, val_labels = _tokenize_split(val_df, tokenizer, max_length)

    train_loader = DataLoader(ProteinDataset(train_ids, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ProteinDataset(val_ids, val_labels), batch_size=eval_batch_size, shuffle=False)
    return train_loader, val_loader
