#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:
    from code.config_utils import load_yaml_config
    from code.data_utils import build_dataloader, read_split_csv
    from code.models import PrefixTunedProT5
    from code.train_utils import accuracy_from_logits, choose_device, save_json
except ImportError:
    from config_utils import load_yaml_config
    from data_utils import build_dataloader, read_split_csv
    from models import PrefixTunedProT5
    from train_utils import accuracy_from_logits, choose_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Prefix-Tuned ProT5 checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path override.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path override.")
    return parser.parse_args()


def evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += output.loss.item()
            total_acc += accuracy_from_logits(output.logits, batch["labels"])
            steps += 1
    return {"loss": total_loss / max(steps, 1), "q10": 100.0 * (total_acc / max(steps, 1))}


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    exp_cfg = cfg["experiment"]

    checkpoint = args.checkpoint if args.checkpoint else Path(exp_cfg["output_dir"]) / "best_checkpoint.pt"
    output_path = args.output if args.output else Path(exp_cfg["output_dir"]) / "test_metrics.json"

    loaded = torch.load(checkpoint, map_location="cpu")
    label2id = loaded["label2id"]

    split = read_split_csv(data_cfg["dataset_csv"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], do_lower_case=False, use_fast=False)
    test_loader = build_dataloader(
        split.test,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    device = choose_device()
    model = PrefixTunedProT5(
        model_name=model_cfg["model_name"],
        num_labels=len(label2id),
        prefix_length=model_cfg["prefix_length"],
        dropout=model_cfg["dropout"],
        freeze_backbone=model_cfg["freeze_backbone"],
    )
    model.load_state_dict(loaded["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, test_loader, device)
    save_json(output_path, metrics)
    print(f"test_q10={metrics['q10']:.3f}")


if __name__ == "__main__":
    main()
