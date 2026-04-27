#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer

from config_utils import load_yaml_config
from data_utils import build_dataloader, read_split_csv
from models import PrefixTunedProT5
from train_prefix_tuning import train_one_run
from train_utils import choose_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-seed Prefix Tuning experiment and aggregate Q10.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def evaluate_test_q10(checkpoint_path: Path, cfg: dict) -> float:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    loaded = torch.load(checkpoint_path, map_location="cpu")
    label2id = loaded["label2id"]

    split = read_split_csv(data_cfg["dataset_csv"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], do_lower_case=False)
    loader = build_dataloader(
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
    ).to(device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            all_labels.extend(labels.tolist())
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            all_preds.extend(output.logits.argmax(dim=-1).cpu().tolist())
    return 100.0 * accuracy_score(all_labels, all_preds)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    seeds = cfg["experiment"]["seeds"]
    base_output = Path(cfg["experiment"]["output_dir"])
    base_output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in seeds:
        run_dir = base_output / f"seed_{seed}"
        train_result = train_one_run(cfg, seed=seed, output_dir=run_dir)
        q10 = evaluate_test_q10(Path(train_result["checkpoint"]), cfg)
        rows.append({"PEFT method": "Prefix tuning", "seed": int(seed), "value": q10 / 100.0})
        print(f"seed={seed} test_q10={q10:.3f}")

    df = pd.DataFrame(rows)
    runs_csv = Path(cfg["results"]["prefix_runs_csv"])
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(runs_csv, index=False)

    summary = {
        "mean_q10": float(df["value"].mean() * 100.0),
        "std_q10": float(df["value"].std(ddof=1) * 100.0),
        "count": int(len(df)),
        "ci95_q10": float(1.96 * df["value"].std(ddof=1) / max(len(df), 1) ** 0.5 * 100.0),
    }
    save_json(cfg["results"]["summary_json"], summary)
    print(f"wrote {runs_csv}")


if __name__ == "__main__":
    main()
