from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


INVALID_AA_PATTERN = re.compile(r"[UZOB]")


def normalize_sequence(sequence: str) -> str:
    cleaned = INVALID_AA_PATTERN.sub("X", sequence.strip().upper())
    return " ".join(list(cleaned))


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class ProteinLocalizationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, label2id: dict[str, int], tokenizer, max_length: int) -> None:
        self.sequences = frame["sequence"].tolist()
        self.labels = [label2id[label] for label in frame["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = normalize_sequence(self.sequences[idx])
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def read_split_csv(path: str | Path) -> SplitData:
    frame = pd.read_csv(path)
    required = {"sequence", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    split_map: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        part = frame.loc[frame["split"].str.lower() == name].copy()
        if part.empty:
            raise ValueError(f"No rows found for split='{name}' in {path}")
        split_map[name] = part.reset_index(drop=True)

    return SplitData(train=split_map["train"], val=split_map["val"], test=split_map["test"])


def build_label_index(frame: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(frame["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_dataloader(
    frame: pd.DataFrame,
    label2id: dict[str, int],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = ProteinLocalizationDataset(frame, label2id, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
