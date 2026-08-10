# rrrrr1030 latest text submission

This repository contains the current TCC, MS, and M methods, their prompts and
fixed scoring rules, the frozen per-metric predictions, and the latest combined
submission for `rrrrr1030`.

Snapshot date: **2026-08-11**

Model: **`openai/gpt-oss-120b`**

## Latest leaderboard result

| Leaderboard column | Kendall average |
| --- | ---: |
| **Text** | **0.4850** |
| Video | 0.4099 |
| **Overall** | **0.4474** |

These are the four-decimal values shown by the leaderboard for rank 1. This
repository contains the text pipeline; the video result came from a separate
pipeline and the frozen CSV here deliberately leaves `VC`, `VA`, and `VE` at
zero. TCC and MS are unchanged from the previous text submission. Given their
known component scores, the rounded `0.4850` text average implies M Kendall is
approximately `0.6910` (the exact component value is not exposed in the
screenshot). Spearman values are also not shown.

**Only M uses high reasoning. TCC and MS use medium reasoning.** The reported
metric snapshot and its precision caveat are stored in
[`artifacts/frozen/leaderboard_metrics.json`](artifacts/frozen/leaderboard_metrics.json),
and the versioned 62-row submission is
[`artifacts/frozen/submission.csv`](artifacts/frozen/submission.csv).

## Active text components

| Metric | Active method | Test distribution |
| --- | --- | --- |
| TCC | `tcc_v1_facet_conservative_v1` | `0: 0, 1: 32, 2: 30` |
| MS | `ms_v3_counterfactual_zero_gate_v8` over the verified v1 baseline | `0: 2, 1: 10, 2: 50` |
| M | `m_v79_existing_evidence_reconciliation_v1` over v7.1 | `0: 9, 1: 5, 2: 48` |

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
validation anchors. The active M run is the **only component run with high
reasoning**.

The v79 reconciliation layer reuses the three v7.1 evidence agents and keeps
the frozen v7.1 label unless their structured outputs meet a narrow,
sample-independent rule. It does not make another model call. Three test rows
change:

| ID | Target | v7.1 | v79 | Evidence rule |
| ---: | --- | ---: | ---: | --- |
| 2 | Hallucination | 1 | 2 | identical target/source group resolves a role-count inconsistency |
| 44 | Debugging | 2 | 1 | related-domain native relation with at most one central role shift |
| 52 | Large language models (LLMs) | 1 | 0 | two judges agree it is a literal in-scope instance |

This improved validation M Kendall from `0.7454` to `0.8666`. The old v7.1
test and validation vectors remain tracked as immutable baselines under
`artifacts/frozen/`.

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

## Zero-loss integrated champion

Before changing any shared semantic extraction, rebuild the current champion
from its frozen TCC, MS, and M components and require byte-identical parity:

```bash
make integrated-best
```

This writes:

- `runs/integrated-best/submission.csv`;
- `runs/integrated-best/parity_audit.json`.

The builder verifies every component hash and the complete `id=0..61` set,
combines the components with fixed LF line endings, and fails unless the result
is byte-for-byte identical to `artifacts/frozen/submission.csv`. The expected
champion SHA256 is
`38339e05b5458e10b9a9be0e323215d2c366997a0064f4d7b4a4a79b107a5cc6`.
This is the immutable control for all subsequent shared-extraction challengers.

## Shared semantic front end (validation candidate)

The three metrics can also run from one reusable semantic analysis instead of
performing separate topic, concept, source, role, and relation extraction. The
shared front end preserves two information firewalls:

1. `target_frame` sees only `TARGET + DESCRIPTION` and extracts topics,
   concepts, roles, relations, and constraints;
2. `source_frame` sees only `ANALOGY` and reconstructs the literal native
   source concept, roles, operations, and relations;
3. `mapping_frame` receives both blind frames and extracts cross-domain
   alignments.

TCC, MS, and M then run concurrently from this one structure. TCC and MS use
medium reasoning; M uses high reasoning. Identical `TARGET + DESCRIPTION`
groups share one target frame. On the 62-row test split there are 47 such
groups. Excluding conditional facet-audit calls, this changes the planned model
call count from about 744 to 543 (201 fewer, approximately 27%).

```bash
SPLIT=validation bash scripts/run_shared_agents.sh
```

Or run a small smoke test directly:

```bash
.venv/bin/python run_text_agents.py \
  --mode shared-active \
  --split validation \
  --ids 0 \
  --max-retries 8 \
  --reasoning-effort medium \
  --m-reasoning-effort high \
  --shared-max-tokens 5000 \
  --output-dir runs/shared-smoke
```

This is a validation candidate, not the frozen leaderboard method. It reuses
the active deterministic score boundaries but changes their upstream evidence,
so it must match or improve validation and public results before replacing the
three independently run active components.

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
- `scripts/run_integrated_best.py`: zero-loss frozen champion rebuild and
  parity proof.
- `scripts/run_shared_agents.sh`: candidate end-to-end shared-front-end run.
- `scripts/generate_augmentation_data.py`: label-safe rewrite and relative-pair
  augmentation with three-reviewer acceptance gates.
- `data_augmentation/`: augmentation inputs, pilot evidence, accepted rows, and
  quality report; the pilot accepted all nine invariant rewrites and rejected
  all 18 counterfactual pairs rather than assigning unsafe pseudo-labels.
- `scripts/combine_recomputed_metrics.py`: strict ID-based result combiner.
- `docs/METHOD.md`: compact method snapshot.
- `docs/GENERALIZATION.md`: leakage and generalization limitations.

## Limitations

These are competition results, not proof of out-of-distribution performance.
The M method uses 12 labeled validation anchors. The MS v8 policy is free of
test-ID branches, but its version was selected after public leaderboard
feedback, so method-selection bias remains. Video metrics are not evaluated in
this repository.
