# centrifuge_new

This repository tracks code only. Raw datasets, processed arrays, training logs, plots, and model weights stay on the server and are not pushed to GitHub.

## Project Layout

- `1_data/`: dataset config and preprocessing scripts
- `3_train/`: training entrypoints and experiment config
- `_2_models/`: model definitions
- `readme_setup/`: environment setup and helper scripts

## Data Location

Training on the server uses local data stored outside Git:

- raw data: `1_data/datasets/`
- processed data: `1_data/processed/`
- results and weights: `4_results/`

These folders are intentionally excluded from version control. If you clone this repo on a new machine, you need to prepare those directories locally before training.

## Typical Server Workflow

```bash
cd /home/chenjingwen/Projects/centrifuge_new/code
python 1_data/data_prep.py
python 3_train/train.py --task experiment_to_centrifuge_2170 --model bsan
```

## Git Usage

This repo is connected to:

- `git@github.com:wynnwin-c/centrifuge_new.git`

Common commands:

```bash
git status
git add <files>
git commit -m "message"
git pull --rebase
git push
```
