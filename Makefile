.PHONY: setup train-prefix eval-prefix run-prefix-seeds figure2

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train-prefix:
	python3 code/train_prefix_tuning.py --config configs/prefix_tuning_default.yaml

eval-prefix:
	python3 code/evaluate_prefix_tuning.py --config configs/prefix_tuning_default.yaml

run-prefix-seeds:
	python3 code/run_prefix_experiment.py --config configs/prefix_tuning_default.yaml

figure2:
	python3 code/plot_figure2.py
