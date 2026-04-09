# 3_train/run_task.py
import argparse
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_task(cfg: dict, target_rpm: int) -> str:
    name = f"exp_to_{target_rpm}"
    for task in cfg["tasks"]:
        if task["name"] == name:
            return task["name"]
    raise ValueError(f"任务未找到: {name}")


def run_one(task_name: str, model: str, extra_args=None):
    if model == "fewshot_binary":
        script = ROOT / "3_train" / "train_fewshot_binary.py"
        cmd = [sys.executable, str(script), "--task", task_name]
    else:
        script = ROOT / "3_train" / "train.py"
        cmd = [sys.executable, str(script), "--task", task_name, "--model", model]
    if extra_args:
        cmd += extra_args
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "3_train" / "config_train.yaml"))
    parser.add_argument("--models", type=str, nargs="+", default=["fewshot_binary"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--per_class", type=int, default=None)
    parser.add_argument("--split_mode", type=str, choices=["repeated", "logo"], default=None)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_name = find_task(cfg, args.target)

    extra = []
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra += ["--batch_size", str(args.batch_size)]
    if args.lr is not None:
        extra += ["--lr", str(args.lr)]
    if args.per_class is not None:
        extra += ["--per_class", str(args.per_class)]
    if args.split_mode is not None:
        extra += ["--split_mode", args.split_mode]
    if args.repeats is not None:
        extra += ["--repeats", str(args.repeats)]

    for model in args.models:
        run_one(task_name, model, extra_args=extra)


if __name__ == "__main__":
    main()
