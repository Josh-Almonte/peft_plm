# CS 4782 Final Project: Figure 2 Reimplementation

## 1. Introduction
This repository re-implements the PEFT comparison experiment from Schmirler et al. (2024), with emphasis on a from-scratch Prefix Tuning implementation for ProT5. A from-scratch LoRA implementation has also been added.

Paper: Schmirler, Heinzinger, Rost, *Fine-tuning protein language models boosts predictions across diverse tasks*, Nature Communications (2024), DOI: [10.1038/s41467-024-51844-2](https://doi.org/10.1038/s41467-024-51844-2).

## 2. Chosen Result
Chosen target: Figure 2 (PEFT comparison on sub-cellular localization), evaluated as Q10 accuracy on the test split.  
This repo trains Prefix Tuning directly and plots Figure 2 using your run outputs (and optional paper values for non-prefix methods).

## 3. GitHub Contents
- `code/`: training, evaluation, experiment orchestration, and plotting scripts.
- `configs/`: YAML experiment configuration files for prefix tuning runs. `lora_default.yaml` added for LoRA.
- `data/`: dataset placement + reference source-data.
- `results/`: generated metrics, per-seed outputs, and figure artifacts.
- `poster/`: in-class presentation poster PDF.
- `report/`: final 2-page report PDF.

## 4. Re-implementation Details
`code/models/prefix_t5.py` implements Prefix Tuning from scratch (learned virtual prefix tokens prepended to encoder embeddings), without PEFT helper libraries.  
Training/evaluation uses ProT5 encoder features with a classification head and reports Q10 on validation/test splits.

### LoRA (added)
`code/models/lora_t5.py` implements LoRA from scratch: low-rank matrices **A** (random init, scale 0.01) and **B** (zero init) are injected into each attention projection (`q`, `k`, `v`, `o`) of the frozen ProT5 encoder. The effective weight update is `ΔW = (α/r) · BA`. A two-layer MLP head (`LayerNorm → Linear → ReLU → Dropout → Linear`) classifies mean-pooled encoder outputs. No PEFT helper libraries are used.

## 5. Reproduction Steps
1. Create environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. The preprocessed dataset is already included at `data/raw/subcellular_localization.csv`. If you need to regenerate it, download the raw splits (`train.pkl`, `valid.pkl`, `test.pkl`) from [HannesStark/protein-localization](https://github.com/HannesStark/protein-localization) and run `data/data_preprocess_to_csv.ipynb` (updating the input paths to match your local copies).

3. Train one Prefix Tuning run (uses `experiment.seed` and `experiment.output_dir` from config):
```bash
python3 code/train_prefix_tuning.py --config configs/prefix_tuning_default.yaml
```

4. Evaluate a trained checkpoint on the test split:
```bash
python3 code/evaluate_prefix_tuning.py --config configs/prefix_tuning_default.yaml
```
This writes `test_metrics.json` under `experiment.output_dir` from the config.

Optional: evaluate a specific checkpoint and output path:
```bash
python3 code/evaluate_prefix_tuning.py \
  --config configs/prefix_tuning_default.yaml \
  --checkpoint results/runs/prefix_tuning/seed_97/best_checkpoint.pt \
  --output results/runs/prefix_tuning/seed_97/test_metrics.json
```

5. Run the full three-seed experiment (seeds from `experiment.seeds`):
```bash
python3 code/run_prefix_experiment.py --config configs/prefix_tuning_default.yaml
```

6. Generate Figure 2:
```bash
python3 code/plot_figure2.py \
  --paper-csv data/reference/figure2_peft_methods_paper.csv \
  --prefix-csv results/prefix_tuning_runs.csv \
  --output results/figures/figure2_reimplementation.png
```

Compute resources: one CUDA GPU (>=24GB VRAM recommended for full ProT5); CPU mode works for small smoke tests only.

### LoRA (added)
7. Update `data.data_path` in `configs/lora_default.yaml` to point to the directory containing `train.pkl`, `valid.pkl`, `test.pkl`.

8. Train one LoRA run:
```bash
python3 -m code.train_lora --config configs/lora_default.yaml
```
Results are written to `results/runs/lora/` by default (`best_checkpoint.pt`, `train_history.json`, `train_summary.json`).

Optional: override seed or output directory:
```bash
python3 -m code.train_lora --config configs/lora_default.yaml --seed 97 --output-dir results/runs/lora/seed_97
```

## 6. Results / Insights
Expected outputs after running the experiment:
- `results/prefix_tuning_runs.csv`: per-seed Prefix Tuning Q10.
- `results/summary_prefix_tuning.json`: aggregate mean/std/95% CI.
- `results/figures/figure2_reimplementation.png`: reproduced Figure 2 style plot.

LoRA outputs (after step 8):
- `results/runs/lora/best_checkpoint.pt`: best checkpoint by val Q10.
- `results/runs/lora/train_history.json`: per-epoch train/val loss and Q10.
- `results/runs/lora/train_summary.json`: best val Q10 and checkpoint path.

## 7. Conclusion
This repo emphasizes reproducible, scriptable re-implementation: training, evaluation, and figure generation are connected end-to-end, and Prefix Tuning logic is implemented directly in-code.

## 8. References
- Schmirler D, Heinzinger M, Rost B. 2024. Nature Communications. DOI: [10.1038/s41467-024-51844-2](https://doi.org/10.1038/s41467-024-51844-2).
- ProT5 checkpoint: [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).
- SubLoc dataset: [HannesStark/protein-localization](https://github.com/HannesStark/protein-localization).

## 9. Acknowledgements
Developed as a final project for CS 4782 at Cornell University, instructed by Prof. Kilian Q. Weinberger and Prof. Wei-Chiu Ma. Team members: Joshua Almonte, Minh Triet Vu.
