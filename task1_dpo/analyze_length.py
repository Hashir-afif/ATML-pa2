from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def print_comparison(cfg: dict, names: list[str]) -> None:
    root = repo_path(cfg["results_dir"]) / "eval"
    print("\nPreference accuracy on the length-stratified held-out set")
    print(f"{'model':18s}" + "".join(f"{s:>18s}" for s in STRATA) + f"{'all':>10s}")
    for name in names:
        m = load_json(root / name / "metrics.json")
        lsp = m["length_stratified_preference"]
        print(f"{name:18s}" + "".join(f"{lsp[s]['preference_accuracy']:>18.4f}" for s in STRATA)
              + f"{lsp['all']['preference_accuracy']:>10.4f}")
    print("\nGeneration (held-out prompts) and word-limit compliance")
    for name in names:
        m = load_json(root / name / "metrics.json")
        g, wl = m["generation_eval_prompts"], m["word_limit"]
        if g.get("status") == "NOT RUN":
            print(f"{name:18s} generation NOT RUN")
            continue
        print(f"{name:18s} tokens mean={g['response_tokens']['mean']:.1f} iqr={g['response_tokens']['iqr']:.1f} "
              f"| word-limit compliance={wl['compliance_rate']:.3f} words mean={wl['word_count']['mean']:.1f}")


def run_length_study(config_path: str = "configs/dpo.yaml", skip_train: bool = False, skip_eval: bool = False) -> None:
    cfg = load_yaml(config_path)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    # Same budget/optimizer/beta/seed/LoRA as the standard one-epoch run; only the training file changes.
    adapter = cfg["length_output"]
    if not skip_train:
        result = run_training(config_path, "length_balanced",
                              dataset_path=cfg["paths"]["dpo_length_train"], output_path=adapter)
        model = result.pop("model")
        del model, result
        clear_gpu()
    if not skip_eval:
        evaluate_model(config_path, adapter, "length_balanced")
        clear_gpu()
        if not (repo_path(cfg["results_dir"]) / "eval" / "standard" / "metrics.json").exists():
            evaluate_model(config_path, cfg["standard_output"], "standard")
            clear_gpu()
    print_comparison(cfg, ["standard", "length_balanced"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="reuse the existing length-balanced adapter")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    run_length_study(args.config, args.skip_train, args.skip_eval)


if __name__ == "__main__":
    main()
