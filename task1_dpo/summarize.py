"""Consolidate Task 1 results: results.json, CSV tables, figures, qualitative candidates.

Reads only what train.py / evaluate.py saved to disk; never recomputes or invents numbers.
Missing runs are recorded as NOT RUN.
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task1_dpo.ablate_beta import beta_tag

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]
STRATUM_LABEL = {"preferred_longer": "Preferred longer", "length_matched": "Length matched",
                 "rejected_longer": "Rejected longer"}
# Fixed entity -> colour mapping (never re-assigned by rank or by which runs exist).
COLORS = {"sft": "#52514e", "standard": "#2a78d6", "beta_0.03": "#eb6834", "beta_0.1": "#1baf7a",
          "beta_0.3": "#4a3aa7", "length_balanced": "#e34948"}
LABELS = {"sft": "SFT (no DPO)", "standard": "Standard DPO (1 epoch)", "beta_0.03": "β=0.03 (short)",
          "beta_0.1": "β=0.10 (short)", "beta_0.3": "β=0.30 (short)", "length_balanced": "Length-balanced DPO"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.8, "lines.linewidth": 2})


def _save(fig, path_stem):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def collect(cfg: dict, names: list[str]):
    root = repo_path(cfg["results_dir"])
    evals, runs, curves = {}, {}, {}
    for name in names:
        mp = root / "eval" / name / "metrics.json"
        evals[name] = load_json(mp) if mp.exists() else {"status": "NOT RUN", "note": f"{mp.name} missing"}
        rp = root / "runs" / name / "run_info.json"
        if name == "sft":
            runs[name] = {"status": "NOT APPLICABLE", "note": "untouched base policy, no training"}
        elif rp.exists():
            runs[name] = load_json(rp)
            curves[name] = read_jsonl(root / "runs" / name / "train_log.jsonl")
        else:
            runs[name] = {"status": "NOT RUN", "note": "run_info.json missing"}
    return evals, runs, curves


def summary_rows(evals, runs):
    rows = []
    for name, m in evals.items():
        r = runs.get(name, {})
        row = {"model": name, "label": LABELS.get(name, name), "status": m.get("status"),
               "train_pairs": r.get("num_pairs_used"), "updates": r.get("updates"), "beta": r.get("beta")}
        if m.get("status") in ("COMPLETED", "SMOKE_TEST"):
            hp = m["heldout_preference"]
            row.update({"heldout_n": hp["n"], "heldout_pref_acc": hp["preference_accuracy"],
                        "heldout_pref_acc_excl_ties": hp["preference_accuracy_excluding_ties"],
                        "heldout_dpo_loss": hp["dpo_loss"], "beta_for_loss": m["beta_for_loss"],
                        "heldout_margin_mean": hp["margin_mean"]})
            g = m["generation_eval_prompts"]
            if g and g.get("status") != "NOT RUN":
                row.update({"kl_token_mean": g["kl_token_mean_pooled"], "kl_seq_sum_mean": g["kl_per_sequence_sum_mean"],
                            "reward_mean": g["reward_mean"], "reward_std": g["reward_std"],
                            "resp_tokens_mean": g["response_tokens"]["mean"], "resp_tokens_std": g["response_tokens"]["std"],
                            "resp_tokens_median": g["response_tokens"]["median"], "resp_tokens_iqr": g["response_tokens"]["iqr"],
                            "truncation_rate": g["truncation_rate"]})
        rows.append(row)
    return rows


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def strata_rows(evals, names):
    rows = []
    for name in names:
        m = evals.get(name, {})
        if m.get("status") not in ("COMPLETED", "SMOKE_TEST"):
            continue
        for s in STRATA + ["all"]:
            d = m["length_stratified_preference"][s]
            rows.append({"model": name, "stratum": s, "n": d["n"], "pref_acc": d.get("preference_accuracy"),
                         "pref_acc_excl_ties": d.get("preference_accuracy_excluding_ties"),
                         "margin_mean": d.get("margin_mean"), "chosen_logratio_mean": d.get("chosen_logratio_mean"),
                         "rejected_logratio_mean": d.get("rejected_logratio_mean")})
    return rows


def per_token_strata_rows(cfg, names):
    """DIAGNOSTIC (not the manual's summed-log-prob metric): per-token-normalised margin by stratum.

    margin_pt = chosen_logratio / chosen_tokens - rejected_logratio / rejected_tokens. Separates a length
    effect of summing over tokens from a per-token preference.
    """
    root = repo_path(cfg["results_dir"]) / "eval"
    rows = []
    for name in names:
        p = root / name / "pairs_length_stratified_eval.jsonl"
        if not p.exists():
            continue
        pairs = read_jsonl(p)
        for s in STRATA:
            R = [r for r in pairs if r["length_stratum"] == s]
            cl = np.array([r["chosen_logratio"] for r in R]); rl = np.array([r["rejected_logratio"] for r in R])
            ct = np.array([r["chosen_tokens_scored"] for r in R]); rt = np.array([r["rejected_tokens_scored"] for r in R])
            rows.append({"model": name, "stratum": s, "n": len(R),
                         "summed_pref_acc": float(np.mean(cl - rl > 0)),
                         "per_token_pref_acc": float(np.mean(cl / ct - rl / rt > 0)),
                         "per_token_chosen_logratio_mean": float(np.mean(cl / ct)),
                         "per_token_rejected_logratio_mean": float(np.mean(rl / rt)),
                         "chosen_tokens_mean": float(ct.mean()), "rejected_tokens_mean": float(rt.mean())})
    return rows


def word_limit_rows(evals):
    rows = []
    for name, m in evals.items():
        wl = m.get("word_limit") if m.get("status") in ("COMPLETED", "SMOKE_TEST") else None
        if not wl or wl.get("status") == "NOT RUN":
            continue
        rows.append({"model": name, "compliance_rate": wl["compliance_rate"], "n_samples": wl["n_samples"],
                     "prompts_all_compliant": wl["prompts_all_samples_compliant"],
                     "words_mean": wl["word_count"]["mean"], "words_std": wl["word_count"]["std"],
                     "words_over_limit_ratio_mean": wl["word_count_over_limit_ratio_mean"],
                     "tokens_mean": wl["response_tokens"]["mean"], "reward_mean": wl["reward_mean"]})
    return rows


# --------------------------------------------------------------------------- figures

def fig_training_curves(curves, out):
    if not curves:
        return
    panels = [("loss", "DPO loss (train, per update)"), ("accuracy", "Preference accuracy (train batch)"),
              ("margin", "Margin m = Δlogratio (nats)"), ("grad_norm", "Grad norm (pre-clip)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, log in curves.items():
            ax.plot([r["update"] for r in log], [r[key] for r in log], color=COLORS.get(name), label=LABELS.get(name, name))
        ax.set_title(title)
        ax.set_xlabel("Optimizer update")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Task 1 DPO training curves (standard: 1 epoch; β forks: short budget)")
    fig.tight_layout()
    _save(fig, out / "task1_training_curves")


def fig_beta(evals, betas, out):
    names = [beta_tag(b) for b in betas]
    ok = [n for n in names if evals.get(n, {}).get("status") == "COMPLETED"
          and evals[n]["generation_eval_prompts"].get("status") != "NOT RUN"]
    if not ok:
        return
    xs = [b for b, n in zip(betas, names) if n in ok]
    panels = [("Held-out preference accuracy", lambda m: m["heldout_preference"]["preference_accuracy"]),
              ("KL from reference (nats/token)", lambda m: m["generation_eval_prompts"]["kl_token_mean_pooled"]),
              ("Reward-model score (mean)", lambda m: m["generation_eval_prompts"]["reward_mean"]),
              ("Response length (tokens, mean)", lambda m: m["generation_eval_prompts"]["response_tokens"]["mean"])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    sft = evals.get("sft", {})
    for ax, (title, f) in zip(axes, panels):
        ax.plot(xs, [f(evals[n]) for n in ok], marker="o", markersize=7, color=COLORS["beta_0.1"], label="Short β forks")
        if sft.get("status") == "COMPLETED":
            try:
                ax.axhline(f(sft), color=COLORS["sft"], linestyle="--", linewidth=1.2, label="SFT baseline")
            except (KeyError, TypeError):
                pass
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_xlabel("β")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out / "task1_beta_study")


def fig_strata(evals, out):
    names = [n for n in ["sft", "standard", "length_balanced"] if evals.get(n, {}).get("status") == "COMPLETED"]
    if len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(names)
    x = np.arange(len(STRATA))
    for i, n in enumerate(names):
        vals = [evals[n]["length_stratified_preference"][s]["preference_accuracy"] for s in STRATA]
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, width=w * 0.92, color=COLORS[n], label=LABELS[n])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABEL[s] for s in STRATA])
    ax.set_ylabel("Preference accuracy (m > 0)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Length-stratified held-out preference accuracy (82 pairs/stratum before filtering)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, out / "task1_length_strata")


def fig_lengths(cfg, evals, out):
    names = [n for n in ["sft", "standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]
             if evals.get(n, {}).get("status") == "COMPLETED"
             and evals[n]["generation_eval_prompts"].get("status") != "NOT RUN"]
    if not names:
        return
    root = repo_path(cfg["results_dir"]) / "eval"
    gen = {n: read_jsonl(root / n / "generations_eval_prompts.jsonl") for n in names}
    wl = {n: read_jsonl(root / n / "generations_word_limit.jsonl") for n in names}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, data, ylabel, title in [
        (axes[0], {n: [r["response_tokens"] for r in gen[n]] for n in names}, "Generated tokens",
         f"Held-out prompts (cap {cfg['max_generation_tokens']} tokens)"),
        (axes[1], {n: [r["word_count"] / r["word_limit"] for r in wl[n]] for n in names}, "Words / word limit",
         "Word-limit prompts (≤ 1.0 is compliant)")]:
        bp = ax.boxplot([data[n] for n in names], patch_artist=True, widths=0.6, medianprops={"color": "#0b0b0b"})
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(COLORS[n])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([LABELS[n] for n in names], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].axhline(1.0, color="#52514e", linestyle="--", linewidth=1.2)
    fig.tight_layout()
    _save(fig, out / "task1_response_lengths")


# --------------------------------------------------------------------------- qualitative candidates

def qualitative_candidates(cfg, evals, top_k):
    """Rank (not judge) candidates for manual reading; selection/annotation is done by a human."""
    root = repo_path(cfg["results_dir"]) / "eval"
    need = ["sft", "standard"]
    if any(evals.get(n, {}).get("status") != "COMPLETED" for n in need):
        return {"status": "NOT RUN", "note": "requires completed sft and standard evaluations"}
    out = {}
    sft = {r["item_key"]: r for r in read_jsonl(root / "sft" / "generations_eval_prompts.jsonl")}
    for name in ["standard", "length_balanced"]:
        p = root / name / "generations_eval_prompts.jsonl"
        if not p.exists():
            continue
        pol = {r["item_key"]: r for r in read_jsonl(p)}
        paired = []
        for k, r in pol.items():
            if k in sft:
                paired.append({"item_key": k, "reward_gain_vs_sft": r["reward"] - sft[k]["reward"],
                               "token_gain_vs_sft": r["response_tokens"] - sft[k]["response_tokens"],
                               "policy": {x: r[x] for x in ("reward", "response_tokens", "truncated", "response")},
                               "sft": {x: sft[k][x] for x in ("reward", "response_tokens", "truncated", "response")}})
        paired.sort(key=lambda d: d["reward_gain_vs_sft"], reverse=True)
        out[f"{name}_largest_reward_gain_vs_sft"] = paired[:top_k]
        out[f"{name}_reward_gain_with_truncation"] = [d for d in paired if d["policy"]["truncated"]][:top_k]
        out[f"{name}_largest_reward_drop_vs_sft"] = paired[::-1][:top_k]
    for name in ["sft", "standard", "length_balanced"]:
        p = root / name / "generations_word_limit.jsonl"
        if p.exists():
            rows = [r for r in read_jsonl(p) if r["compliant"] == 0.0]
            rows.sort(key=lambda r: r["reward"], reverse=True)
            out[f"{name}_word_limit_violations_by_reward"] = rows[:top_k]
    # held-out pairs the standard model gets wrong despite a large rating gap, and right on rating ties
    pairs = read_jsonl(root / "standard" / "pairs_standard_eval.jsonl")
    wrong = sorted([p for p in pairs if not p["correct"]],
                   key=lambda p: p["score_chosen"] - p["score_rejected"], reverse=True)
    out["standard_heldout_wrong_with_large_rating_gap"] = wrong[:top_k]
    return out


def build_summary(config_path: str = "configs/dpo.yaml") -> dict:
    cfg = load_yaml(config_path)
    betas = [float(b) for b in cfg["betas"]]
    names = ["sft", "standard"] + [beta_tag(b) for b in betas] + ["length_balanced"]
    evals, runs, curves = collect(cfg, names)
    root = repo_path(cfg["results_dir"])
    fig_dir = root / "figures"

    rows = summary_rows(evals, runs)
    write_csv(root / "summary_table.csv", rows)
    write_csv(root / "length_strata.csv", strata_rows(evals, ["sft", "standard", "length_balanced"]))
    write_csv(root / "word_limit.csv", word_limit_rows(evals))
    pt_rows = per_token_strata_rows(cfg, names)
    write_csv(root / "length_strata_per_token_diagnostic.csv", pt_rows)
    fig_training_curves(curves, fig_dir)
    fig_beta(evals, betas, fig_dir)
    fig_strata(evals, fig_dir)
    fig_lengths(cfg, evals, fig_dir)
    save_json(root / "qualitative_candidates.json", qualitative_candidates(cfg, evals, int(cfg["qualitative_top_k"])))

    budgets = {n: {"train_pairs": runs[n].get("num_pairs_used"), "updates": runs[n].get("updates")}
               for n in names if runs[n].get("status") not in ("NOT RUN", "NOT APPLICABLE")}
    results = {
        "task": "task1_dpo",
        "note": "Standard and length-balanced runs use the one-epoch budget; beta forks use the short "
                f"budget (first {cfg['short_ablation_examples']} pairs). Budgets differ; see 'budgets'.",
        "budgets": budgets,
        "training_runs": runs,
        "training_curves": curves,
        "evaluation": evals,
        "tables": {"summary": rows, "length_strata": strata_rows(evals, ["sft", "standard", "length_balanced"]),
                   "word_limit": word_limit_rows(evals),
                   "length_strata_per_token_diagnostic": pt_rows},
    }
    save_json(root / "results.json", results)
    print(f"Wrote {root / 'results.json'}")
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("model", "status", "heldout_pref_acc", "heldout_dpo_loss", "kl_token_mean",
                                   "reward_mean", "resp_tokens_mean")}))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    build_summary(args.config)


if __name__ == "__main__":
    main()
