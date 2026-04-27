# CS 4782 Final Project: Figure 2 Reimplementation

## 1. Introduction
This repository re-implements the PEFT comparison experiment from Schmirler et al. (2024), with emphasis on a from-scratch Prefix Tuning implementation for ProT5.

Paper: Schmirler, Heinzinger, Rost, *Fine-tuning protein language models boosts predictions across diverse tasks*, Nature Communications (2024), DOI: [10.1038/s41467-024-51844-2](https://doi.org/10.1038/s41467-024-51844-2).

## 2. Chosen Result
Chosen target: Figure 2 (PEFT comparison on sub-cellular localization), evaluated as Q10 accuracy on the test split.  
This repo trains Prefix Tuning directly and plots Figure 2 using your run outputs (and optional paper values for non-prefix methods).

## 3. GitHub Contents
- `code/`: training, evaluation, experiment orchestration, and plotting scripts.
- `configs/`: experiment config files.
- `data/`: dataset placement + reference source-data.
- `results/`: generated metrics, per-seed outputs, and figure artifacts.
- `poster/`: poster PDF placeholder.
- `report/`: final report PDF placeholder.

## 4. Re-implementation Details
`code/models/prefix_t5.py` implements Prefix Tuning from scratch (learned virtual prefix tokens prepended to encoder embeddings), without PEFT helper libraries.  
Training/evaluation uses ProT5 encoder features with a classification head and reports Q10 on validation/test splits.

## 5. Reproduction Steps
1. Create environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Put localization data into `data/raw/subcellular_localization.csv` with columns: `sequence,label,split` where split is `train|val|test`.
3. Run three-seed Prefix Tuning experiment:
```bash
python3 code/run_prefix_experiment.py --config configs/prefix_tuning_default.yaml
```
4. Generate Figure 2:
```bash
python3 code/plot_figure2.py \
  --paper-csv data/reference/figure2_peft_methods_paper.csv \
  --prefix-csv results/prefix_tuning_runs.csv \
  --output results/figures/figure2_reimplementation.png
```

Compute resources: one CUDA GPU (>=24GB VRAM recommended for full ProT5); CPU mode works for small smoke tests only.

## 6. Results / Insights
Expected outputs after running the experiment:
- `results/prefix_tuning_runs.csv`: per-seed Prefix Tuning Q10.
- `results/summary_prefix_tuning.json`: aggregate mean/std/95% CI.
- `results/figures/figure2_reimplementation.png`: reproduced Figure 2 style plot.

## 7. Conclusion
This repo emphasizes reproducible, scriptable re-implementation: training, evaluation, and figure generation are connected end-to-end, and Prefix Tuning logic is implemented directly in-code.

## 8. References
- Schmirler D, Heinzinger M, Rost B. 2024. Nature Communications. DOI: [10.1038/s41467-024-51844-2](https://doi.org/10.1038/s41467-024-51844-2).
- ProT5 checkpoint: [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50).

## 9. Acknowledgements
Developed as coursework for CS 4782 final project.
