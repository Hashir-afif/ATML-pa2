# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.evaluate --config configs/dpo.yaml --name sft                      # untouched SFT baseline row
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard                 # Step 1: one epoch
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml                               # Step 2: beta in {0.03,0.10,0.30}, short budget
python -m task1_dpo.analyze_length --config configs/dpo.yaml                            # Step 3: length-balanced run + strata
python -m task1_dpo.summarize --config configs/dpo.yaml                                 # results.json, CSV tables, figures
```

Outputs: adapters in `outputs/task1_dpo/<run>/`; training logs and run metadata in
`results/task1_dpo/runs/<run>/`; per-pair margins, generations and metrics in
`results/task1_dpo/eval/<model>/`; consolidated `results/task1_dpo/results.json`, `summary_table.csv`,
`length_strata.csv`, `word_limit.csv` and `figures/`.

Defect fix: `task1_dpo/dpo.py` used `beta * (policy_margin + ref_margin)`; the DPO objective requires
`beta * (policy_margin - ref_margin)`. Pairs whose prompt alone exceeds `max_sequence_length` (768) are
skipped by the fixed rule `overlong_prompt_policy: skip` and their `source_index` values are logged.

### Task 2 - PPO

```bash
export HF_HUB_OFFLINE=1   # all models are cached after the first run; avoids slow hub checks
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard                 # Step 1: 20 updates
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml   # Step 2: cached batch + eps forks (8 updates each)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml          # Step 3: kl_beta forks (8 updates each)
python -m task2_ppo.summarize --config configs/ppo.yaml          # results.json, CSV tables, figures
```

Outputs: policy adapters in `outputs/task2_ppo/<run>/` (critic adapter in `<run>/value/`); per-update logs and
run metadata in `results/task2_ppo/runs/<run>/`; held-out generations + metrics in `results/task2_ppo/eval/<name>/`;
cached-batch study in `results/task2_ppo/clipping/`; consolidated `results/task2_ppo/results.json`, CSV tables
(including paired per-prompt reward differences vs the midpoint) and `figures/`. Fork runs are named
`fork_eps<eps>_kl<beta>`; the eps=0.2/kl=0.1 fork is shared by both studies.

Defect fix: `task2_ppo/ppo.py::ppo_policy_loss` used `torch.maximum` of the two surrogates; the clipped
objective requires `torch.minimum` (pessimistic bound). Implementation choices (all in `configs/ppo.yaml`):
dropout disabled during PPO, raw (un-whitened) GAE advantages, prompts longer than 256 tokens skipped and logged,
critic score head trained with fp32 master weights under fp16 autocast.

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
