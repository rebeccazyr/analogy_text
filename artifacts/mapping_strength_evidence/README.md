# Original-v1 MS evidence

This directory contains the surviving evidence for the Mapping Strength values
used by the best submission.

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
