from __future__ import annotations

import argparse

from common.data import load_yaml
from common.models import clear_gpu
from task2_ppo.continue_train import fork_name, run_or_reuse_fork
from task2_ppo.evaluate import ensure_evaluated


def run_kl_study(config_path: str = "configs/ppo.yaml", force: bool = False) -> None:
    """Matched short forks from the identical midpoint: only kl_beta changes.

    eps, prompts (the first fork_updates prompts of the fixed order), seed, optimizer, critic start state and
    evaluation protocol are shared with every other fork. The kl_beta=0.10 fork is the same configuration
    as the eps=0.20 clipping fork and is trained once (run_or_reuse_fork).
    """
    cfg = load_yaml(config_path)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    eps = float(cfg["clip_epsilon"])
    for beta in [float(b) for b in cfg["kl_values"]]:
        adapter = run_or_reuse_fork(config_path, eps, beta, force=force)
        ensure_evaluated(config_path, adapter, fork_name(eps, beta), force=force)
        clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if completed outputs exist")
    args = ap.parse_args()
    run_kl_study(args.config, args.force)


if __name__ == "__main__":
    main()
