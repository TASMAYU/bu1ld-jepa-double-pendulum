PYTHON ?= python3

.PHONY: help all data baselines verify clean

help:
	@echo "make all        - full M1 pipeline: generate data + run baselines (train/val only)"
	@echo "make data       - simulate, render, hash and manifest all three splits"
	@echo "make baselines  - fit and score the frozen baseline family (train/val only)"
	@echo "make verify     - re-derive every array and check it against the manifest hashes"
	@echo "make clean      - remove generated data and results"

all:
	$(PYTHON) -m src.run_pipeline --stage all

data:
	$(PYTHON) -m src.run_pipeline --stage data

baselines:
	$(PYTHON) -m src.run_pipeline --stage baselines

verify:
	$(PYTHON) -m src.run_pipeline --stage verify

clean:
	rm -rf data/raw
	rm -f results/baselines_m1.json
