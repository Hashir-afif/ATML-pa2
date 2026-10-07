from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}"


def run_beta_study(config_path: str = "configs/dpo.yaml", betas: list[float] | None = None,
                   skip_train: bool = False, skip_eval: bool = False) -> None:
    cfg = load_yaml(config_path)
    betas = betas or [float(b) for b in cfg["betas"]]
    n_short = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n_short)

    # Every fork: fresh LoRA on the original SFT policy, same first-n_short pairs, same seed,
    # optimizer, LoRA config and evaluation protocol. Only beta changes.
    for beta in betas:
        tag = beta_tag(beta)
        adapter = f"outputs/task1_dpo/{tag}"
        if not skip_train:
            result = run_training(config_path, tag, output_path=adapter, beta=beta, max_examples=n_short)
            model = result.pop("model")
            del model, result
            clear_gpu()
        elif not repo_path(adapter).exists():
            raise SystemExit(f"--skip-train but {adapter} does not exist")
        if not skip_eval:
            evaluate_model(config_path, adapter, tag, beta=beta)
            clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+", help="default: cfg['betas']")
    ap.add_argument("--skip-train", action="store_true", help="evaluate existing fork adapters only")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    run_beta_study(args.config, args.betas, args.skip_train, args.skip_eval)


if __name__ == "__main__":
    main()
