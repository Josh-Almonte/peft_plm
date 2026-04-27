#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["LoRA", "DoRA", "IA3", "Prefix tuning"]
METHOD_COLORS = {
    "LoRA": "#4C93CF",
    "DoRA": "#69B3FF",
    "IA3": "#F47B20",
    "Prefix tuning": "#C62828",
}
PRETRAINED_BASELINE_Q10 = 61.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2 reimplementation from run outputs.")
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=Path("data/reference/figure2_peft_methods_paper.csv"),
        help="CSV from paper source-data containing LoRA/DoRA/IA3/Prefix values.",
    )
    parser.add_argument(
        "--prefix-csv",
        type=Path,
        default=Path("results/prefix_tuning_runs.csv"),
        help="CSV from this repo runs; when present it replaces Prefix tuning rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/figure2_reimplementation.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.assign(accuracy_q10=lambda d: d["value"] * 100.0)
        .groupby("PEFT method", as_index=False)["accuracy_q10"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"PEFT method": "method"})
    )
    grouped["ci95"] = 1.96 * grouped["std"] / grouped["count"].pow(0.5)
    grouped["method"] = pd.Categorical(grouped["method"], METHOD_ORDER, ordered=True)
    return grouped.sort_values("method").reset_index(drop=True)


def build_plot_df(paper_csv: Path, prefix_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(paper_csv)
    if prefix_csv.exists():
        ours = pd.read_csv(prefix_csv)
        ours = ours.loc[ours["PEFT method"] == "Prefix tuning"].copy()
        base = base.loc[base["PEFT method"] != "Prefix tuning"].copy()
        base = pd.concat([base, ours], ignore_index=True)
    return base


def plot(df: pd.DataFrame, out_path: Path) -> None:
    stats = summarize(df)
    y_positions = {method: len(METHOD_ORDER) - 1 - i for i, method in enumerate(METHOD_ORDER)}

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=120)

    for method in METHOD_ORDER:
        y = y_positions[method]
        row = stats.loc[stats["method"] == method].iloc[0]
        values = df.loc[df["PEFT method"] == method, "value"] * 100.0
        color = METHOD_COLORS[method]

        ax.errorbar(
            row["mean"],
            y,
            xerr=row["ci95"],
            fmt="none",
            ecolor=color,
            elinewidth=2.0,
            capsize=6,
            zorder=2,
        )

        for x in values:
            ax.vlines(x, y - 0.23, y + 0.23, color=color, linewidth=2, zorder=3)
            ax.scatter(
                x,
                y,
                s=48,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=4,
            )

        ax.text(51.0, y, f"{row['mean']:.1f} ± {row['ci95']:.1f}", va="center", ha="left", fontsize=20, color="black")

    ax.axvline(PRETRAINED_BASELINE_Q10, color="gray", linestyle=(0, (4, 3)), linewidth=2.0, zorder=1)
    ax.text(
        PRETRAINED_BASELINE_Q10 - 0.2,
        1.5,
        "pretrained embedding",
        rotation=90,
        va="center",
        ha="right",
        fontsize=13,
        color="dimgray",
    )

    ax.set_yticks([y_positions[m] for m in METHOD_ORDER], labels=["LoRA", "DoRA", "IA3", "Prefix\ntuning"])
    ax.set_xlim(49, 71)
    ax.set_xticks([50, 55, 60, 65, 70])
    ax.set_xlabel("Accuracy (Q10)", fontsize=18, color="black")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=16, colors="#444444")
    ax.grid(True, axis="both", color="#D3D3D3", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = build_plot_df(args.paper_csv, args.prefix_csv)
    plot(df, args.output)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
