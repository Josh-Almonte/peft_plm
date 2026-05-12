#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5Tokenizer

try:
    from code.config_utils import load_yaml_config
    from code.lora_data_utils import build_lora_dataloaders
    from code.models.lora_t5 import T5LoRAClassifier
    from code.train_utils import accuracy_from_logits, choose_device, save_json, set_seed
except ImportError:
    from config_utils import load_yaml_config
    from lora_data_utils import build_lora_dataloaders
    from models.lora_t5 import T5LoRAClassifier
    from train_utils import accuracy_from_logits, choose_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA-adapted ProT5 on sub-cellular localization.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory from config.")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            total_loss += criterion(logits, labels).item()
            total_acc += accuracy_from_logits(logits, labels)
            count += 1
    return {"loss": total_loss / max(count, 1), "q10": 100.0 * (total_acc / max(count, 1))}


def train_one_run(cfg: dict, seed: int, output_dir: Path) -> dict[str, float]:
    set_seed(seed)
    device = choose_device()

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    tokenizer = T5Tokenizer.from_pretrained(model_cfg["model_name"], do_lower_case=False)
    train_loader, val_loader = build_lora_dataloaders(
        data_path=data_cfg["data_path"],
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        batch_size=train_cfg["batch_size"],
        eval_batch_size=train_cfg["eval_batch_size"],
    )

    model = T5LoRAClassifier(
        model_name=model_cfg["model_name"],
        hidden_dim=model_cfg["hidden_dim"],
        num_classes=model_cfg["num_classes"],
        r=model_cfg["r"],
        alpha=model_cfg["alpha"],
        lora_dropout=model_cfg["lora_dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_q10 = -1.0
    best_ckpt = output_dir / "best_checkpoint.pt"
    history: list[dict] = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy_from_logits(logits, labels)
            steps += 1

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "train_q10": 100.0 * (running_acc / max(steps, 1)),
            "val_loss": val_metrics["loss"],
            "val_q10": val_metrics["q10"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}  train_loss={epoch_metrics['train_loss']:.4f}"
            f"  val_loss={val_metrics['loss']:.4f}  val_q10={val_metrics['q10']:.2f}"
        )

        if val_metrics["q10"] > best_val_q10:
            best_val_q10 = val_metrics["q10"]
            torch.save(
                {"model_state_dict": model.state_dict(), "model_cfg": model_cfg, "seed": seed},
                best_ckpt,
            )

    save_json(
        output_dir / "train_history.json",
        {"history": history, "best_val_q10": best_val_q10, "seed": seed},
    )
    return {"best_val_q10": best_val_q10, "checkpoint": str(best_ckpt)}


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["experiment"]["seed"])
    output_dir = args.output_dir if args.output_dir is not None else Path(cfg["experiment"]["output_dir"])
    result = train_one_run(cfg, seed=seed, output_dir=output_dir)
    save_json(output_dir / "train_summary.json", result)
    print(f"best_val_q10={result['best_val_q10']:.3f}")
    print(f"checkpoint={result['checkpoint']}")


if __name__ == "__main__":
    main()
