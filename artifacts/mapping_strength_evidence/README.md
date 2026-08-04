# Original-v1 MS evidence

This directory contains the surviving original-v1 evidence used as the frozen
baseline for the active `ms_v3_counterfactual_zero_gate_v8` method.

## Contents

- `cache/original_prompt/openai_gpt_oss_120b/{validation,test}/{id}/mapping_extractor.json`
- `cache/original_prompt/openai_gpt_oss_120b/{validation,test}/{id}/ms_judge.json`

There are 74 examples (12 validation and 62 test), two cached agent responses
per example, and 148 cache files in total. Each cache entry records the model,
prompt version, prompt hash, timestamp, token usage, and structured result.

## Original request settings

```text
model: openai/gpt-oss-120b
temperature: 0.2
top_p: 1.0
reasoning_effort: medium
max_tokens: 2200
seed: 42 on the first attempt
max_retries: 3
prompt_version: v1
```

The raw prompt source is frozen in
`analogy_agents/original_mapping_strength_prompts.py`. Rebuilding the prompts
from the dataset and archived mapping results reproduces every stored prompt
hash: 74/74 for MappingExtractor and 74/74 for MSJudge.

Verify the complete chain with:

```bash
.venv/bin/python scripts/verify_mapping_strength_archive.py
```

Replay the archived responses without an API call with:

```bash
.venv/bin/python run_text_agents.py \
  --mode mapping-strength \
  --split test \
  --cache-dir artifacts/mapping_strength_evidence/cache \
  --output-dir runs/mapping_strength_replay
```

Add `--refresh-cache` and use a writable cache directory such as `.agent_cache`
to make fresh Together calls. Fresh inference may not reproduce every label.

## Active v8 correction

The active submission does not use the archived v1 score as the final value in
two test rows. Three blind source reconstructions and three counterfactual
audits are run with medium reasoning. Python requires at least two zero votes
before replacing the v1 score with zero; otherwise it preserves the archived
score. The active code has no ID-specific scoring branch.

The resulting test changes are `id=15` (`recursion`, votes `[1,0,0]`) and
`id=22` (`backpropagation`, votes `[0,0,0]`). The frozen corrected vector is
`artifacts/frozen/mapping_strength_predictions.csv`; the uncorrected archive
vector remains in
`artifacts/frozen/mapping_strength_v1_baseline_predictions.csv`.
