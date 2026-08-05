# rrrrr1030 latest text submission

This repository contains the current TCC, MS, and M methods, their prompts and
fixed scoring rules, the frozen per-metric predictions, and the latest combined
submission for `rrrrr1030`.

Snapshot date: **2026-08-04**

Model: **`openai/gpt-oss-120b`**

## Latest leaderboard result

| Metric | Kendall | Spearman | Reasoning effort |
| --- | ---: | ---: | --- |
| TCC | 0.3359301817 | 0.3359301817 | medium |
| MS | 0.4280255989 | 0.4407342051 | medium |
| M | 0.5909945085 | 0.6266820770 | **high** |
| **Text average** | **0.4516500964** | **0.4677821546** | — |
| Video average | 0 | 0 | not run |
| **Overall average** | **0.2258250482** | **0.2338910773** | — |

**Only M uses high reasoning. TCC and MS use medium reasoning.** VC, VA, and VE
are all zero; this repository covers the text pipeline only. The exact metric
snapshot is stored in
[`artifacts/frozen/leaderboard_metrics.json`](artifacts/frozen/leaderboard_metrics.json),
and the versioned 62-row submission is
[`artifacts/frozen/submission.csv`](artifacts/frozen/submission.csv).

## Active text components

| Metric | Active method | Test distribution |
| --- | --- | --- |
| TCC | `tcc_v1_facet_conservative_v1` | `0: 0, 1: 32, 2: 30` |
| MS | `ms_v3_counterfactual_zero_gate_v8` over the verified v1 baseline | `0: 2, 1: 10, 2: 50` |
| M | `v7_1_role_audit_loo`, run with high reasoning | `0: 8, 1: 6, 2: 48` |

Every row contains `TARGET`, `DESCRIPTION`, and `ANALOGY`. TCC, MS, and M are
predicted independently and combined by ID. No active scoring branch contains
a test-ID override.

## TCC — Target Concept Coverage

TCC asks how much of the reference description is represented by the analogy.
It does not judge mapping correctness or source/target distance.

1. The recovered exact-v1 `ConceptDecomposer` splits `DESCRIPTION` into topics.
2. The exact-v1 `TCCJudge` marks each topic absent, partial, or covered.
3. A topic-relation judge filters entailed restatements and contextual details.
4. A facet auditor checks the unresolved blockers.
5. Python permits only a conservative `1 -> 2` correction when no substantive
   or scope blocker remains.

The active TCC run uses **medium reasoning**.

## MS — Mapping Strength

MS asks whether the source-to-target correspondences preserve the relevant
roles, relations, operations, and causal direction. It deliberately ignores
topic completeness (TCC), domain distance (M), and prose quality.

The verified original-v1 MappingExtractor and MSJudge remain the default
baseline. Their prompt hashes match all 74 archived validation/test examples,
and the 148 original structured responses are retained under
`artifacts/mapping_strength_evidence/`. The corresponding uncorrected vector is
retained separately as
`artifacts/frozen/mapping_strength_v1_baseline_predictions.csv`.

The active v8 correction replaces the former two-cell manual candidate with a
prompt-driven, sample-independent zero gate:

1. Load the frozen, medium-reasoning v1 mapping and MS judgment.
2. Run three blind source framers. They see the analogy but not its target and
   quarantine labels, equations, or target-native operations that do not
   naturally belong to the source story.
3. Run three counterfactual auditors against those independently reconstructed
   source frames.
4. Python converts each audit to either `0` or “keep the v1 score.”
5. The final score becomes `0` only when at least two of the three votes are
   zero. The policy cannot promote a row or otherwise replace the v1 `1/2`
   boundary.

The deterministic zero rule is restricted to three decisive cases:

- an imported formal mechanism dominates the explanation;
- a recursion/self-reference target is represented only by a linear handoff;
- the source operation is impossible or reverses the core target relation.

On test, this process changes only `id=15` (`recursion`, votes `[1,0,0]`) and
`id=22` (`backpropagation`, votes `[0,0,0]`) from `MS=1` to `MS=0`. All other
test MS values remain at the verified v1 baseline. On validation it also
identifies the symbolic-computation target-injection case; validation Kendall
is `0.7655473322` and Spearman is `0.7715167498`.

The active MS correction uses **medium reasoning**. Its prompt, schemas, and
fixed Python conversion are in:

- `analogy_agents/ms_corrective_prompts.py`
- `analogy_agents/ms_corrective_schemas.py`
- `analogy_agents/ms_native_prompts.py`
- `analogy_agents/ms_native_schemas.py`
- `analogy_agents/pipeline.py`

## M — Metaphoricity

M asks how much semantic type translation is required between the literal
source mechanism and the target. It ignores coverage, mapping quality, length,
and fluency.

1. Reconstruct the smallest literal source mechanism after removing headings,
   parenthetical glosses, and explicit mapping language.
2. Apply a strict literal-instance gate.
3. Check whether the source and target preserve a native relation and how many
   central roles change semantic type.
4. Apply the fixed boundary:

```text
literal instance                                      -> 0
native relation and at most one central role change   -> 1
otherwise                                             -> 2
```

Validation uses leave-one-out calibration anchors. Test inference uses all 12
validation anchors. The active M v7.1 run is the **only component run with high
reasoning**.

### Experimental M cosine baseline

`--mode m-cosine` keeps the current literal-instance judge for the `M=0`
boundary, but replaces the nonliteral `M=1/2` LLM judgment with deterministic
cosine scoring:

1. Reuse `DomainAnalysis` to form symmetric source/target concept texts and
   source/target domain texts.
2. Embed the four texts locally with sentence-transformers and
   `BAAI/bge-large-en-v1.5` by default.
3. Compute `1 - cosine_similarity` for concept and domain separately.
4. Average the two distances by default. A nonliteral distance at or below
   `0.35` receives `M=1`; a larger distance receives `M=2`.

The weight and threshold are explicit experimental parameters, not claimed
calibrated constants. Run the 12-row validation split first:

```bash
.venv/bin/python run_text_agents.py \
  --mode m-cosine \
  --split validation \
  --reasoning-effort high \
  --embedding-device cuda:0 \
  --m-concept-weight 0.5 \
  --m-cosine-threshold 0.35 \
  --output-dir runs/m_cosine_validation
```

Together is used only for the structured domain analysis and literal-instance
gate; embedding inference does not call the Together API. The local model is
loaded once per process, uses CUDA automatically when available, and can be
pinned to the single H100 with `--embedding-device cuda:0`. The structured LLM
evidence and embedding vectors are cached separately. Each
details row records the concept, domain, and combined distances, the literal
gate result, threshold margin, embedding model, weight, and cutoff. This mode
does not change the active frozen submission.

## Install and test

Run commands from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Configure a Together API key:

```bash
cp .env.example .env
```

Put the key in `.env`, then run the active pipeline:

```bash
bash scripts/run_agents.sh
```

The script explicitly runs TCC and MS with `medium` reasoning and M with
`high` reasoning, then combines only those newly generated prediction files.
Results are written under `runs/recomputed/`. Set `REFRESH_CACHE=1` to force
fresh model calls. Fresh inference can differ from the frozen snapshot even
with the same prompts and settings.

The original-v1 MS prompt/evidence chain can be verified independently:

```bash
.venv/bin/python scripts/verify_mapping_strength_archive.py
```

## Repository guide

- `artifacts/frozen/submission.csv`: latest versioned 62-row submission.
- `artifacts/frozen/`: latest component predictions, validation summaries, and
  leaderboard metrics.
- `artifacts/mapping_strength_evidence/`: tracked original-v1 MS evidence used
  as the v8 baseline.
- `analogy_agents/`: prompts, schemas, agent orchestration, and fixed scoring
  rules.
- `run_text_agents.py`: CLI entry point for individual modes.
- `scripts/run_agents.sh`: active end-to-end reasoning configuration.
- `scripts/combine_recomputed_metrics.py`: strict ID-based result combiner.
- `docs/METHOD.md`: compact method snapshot.
- `docs/GENERALIZATION.md`: leakage and generalization limitations.

## Limitations

These are competition results, not proof of out-of-distribution performance.
The M method uses 12 labeled validation anchors. The MS v8 policy is free of
test-ID branches, but its version was selected after public leaderboard
feedback, so method-selection bias remains. Video metrics are not evaluated in
this repository.
