#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from code.config_utils import load_yaml_config
    from code.data_utils import build_dataloader, build_label_index, read_split_csv
    from code.models import PrefixTunedProT5
    from code.train_utils import accuracy_from_logits, choose_device, save_json, set_seed
except ImportError:
    from config_utils import load_yaml_config
    from data_utils import build_dataloader, build_label_index, read_split_csv
    from models import PrefixTunedProT5
    from train_utils import accuracy_from_logits, choose_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prefix-Tuned ProT5 on sub-cellular localization.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory from config.")
    return parser.parse_args()


def evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
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
            count += 1
    return {"loss": total_loss / max(count, 1), "q10": 100.0 * (total_acc / max(count, 1))}


def train_one_run(cfg: dict, seed: int, output_dir: Path) -> dict[str, float]:
    set_seed(seed)
    device = choose_device()

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    split = read_split_csv(data_cfg["dataset_csv"])
    label2id, id2label = build_label_index(split.train)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], do_lower_case=False)
    train_loader = build_dataloader(
        split.train,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = build_dataloader(
        split.val,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=model_cfg["max_length"],
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = PrefixTunedProT5(
        model_name=model_cfg["model_name"],
        num_labels=len(label2id),
        prefix_length=model_cfg["prefix_length"],
        dropout=model_cfg["dropout"],
        freeze_backbone=model_cfg["freeze_backbone"],
    ).to(device)

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_q10 = -1.0
    best_ckpt = output_dir / "best_checkpoint.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        steps = 0

        bar = tqdm(train_loader, desc=f"epoch={epoch}", leave=False)
        for batch in bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()

            steps += 1
            running_loss += output.loss.item()
            running_acc += accuracy_from_logits(output.logits, batch["labels"])
            bar.set_postfix(loss=f"{running_loss / steps:.4f}", q10=f"{100.0 * running_acc / steps:.2f}")

        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "train_q10": 100.0 * (running_acc / max(steps, 1)),
            "val_loss": val_metrics["loss"],
            "val_q10": val_metrics["q10"],
        }
        history.append(epoch_metrics)

        if val_metrics["q10"] > best_val_q10:
            best_val_q10 = val_metrics["q10"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "label2id": label2id,
                    "id2label": id2label,
                    "seed": seed,
                },
                best_ckpt,
            )

    save_json(output_dir / "train_history.json", {"history": history, "best_val_q10": best_val_q10, "seed": seed})
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
