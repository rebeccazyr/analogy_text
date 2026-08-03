# rrrrr1030 best text submission

This repository contains the three text-metric methods used by the best known
`rrrrr1030` run, together with their prompts, scoring rules, archived evidence,
and runnable agent pipeline.

| Text Kendall | Text Spearman | Video Kendall | Video Spearman |
| ---: | ---: | ---: | ---: |
| 0.4109 | 0.4267 | 0 | 0 |

The run used:

- TCC: `tcc_v1_facet_conservative_v1`
- MS: hash-verified original-v1 MappingExtractor + MSJudge
- M: `v7_1_role_audit_loo`
- VC, VA, VE: all zero in the leaderboard run; video processing is outside this
  repository's scope.

Run all commands below from the repository root. Python 3.10 or newer is
recommended.

## How the three text metrics are calculated

Every row has the same three inputs: `TARGET`, its reference `DESCRIPTION`, and
the generated `ANALOGY`. TCC, MS, and M are predicted independently. Each agent
writes its own prediction CSV; the full rerun script then combines those newly
generated files into the submission format.

### TCC — Target Concept Coverage

TCC asks **how much of the reference description is represented by the
analogy**, without judging whether the mappings are logical or metaphorically
distant.

1. The recovered exact-v1 `ConceptDecomposer` splits `DESCRIPTION` into 2–6
   topics.
2. The exact-v1 `TCCJudge` labels every topic `absent`, `partial`, or `covered`
   and applies the official scale:
   - `0`: no topic is covered;
   - `1`: some, but not all, topics are covered;
   - `2`: all topics are covered.
3. A separate topic-relation judge sees `TARGET`, `DESCRIPTION`, and the topic
   list—but not `ANALOGY`—and marks each topic as an independent requirement,
   an entailed restatement, or contextual detail. Only independent requirements
   are retained as possible blockers.
4. When the original score is `1`, a facet auditor splits remaining blockers
   into atomic facets and marks each one covered or missing. Missing substantive
   functions/relations and scope/constraints remain blockers. A residual gap
   that is only category/medium, terminology, an illustration, or a measurement
   convention does not block full coverage.
5. Python applies one sample-independent correction:

```text
if original_v1_TCC == 1 and no retained blocker remains:
    final_TCC = 2
else:
    final_TCC = original_v1_TCC
```

Therefore this policy can only promote `1 -> 2`. It never demotes a score,
changes `0`, or uses an example-ID override. Archived test distribution:
`0: 0`, `1: 32`, `2: 30`.

### MS — Mapping Strength

MS asks **whether the source-to-target correspondences are logically sound and
internally consistent**. It deliberately ignores topic completeness (TCC),
source/target distance (M), and writing fluency.

The original-v1 workflow extracted explicit or clearly implied mappings, then
classified each mapping as:

- `sound`: roles, relation, and mechanism align;
- `stretch`: understandable, but structure is lost or a logical leap is needed;
- `inconsistent`: the mapping contradicts or cannot support the claimed target
  relation.

The ordinal rubric is:

- `0`: mappings are far-fetched or highly inconsistent;
- `1`: some logical stretches or inconsistencies remain;
- `2`: mappings are well aligned, sound, and consistent.

At archive time, the original-v1 cache retained the prompt version and hash but
not a separate copy of the prompt source. The surviving MappingExtractor and
MSJudge templates were later checked against every archived example. Rebuilt
hashes match `74/74` MappingExtractor calls and `74/74` MSJudge calls. The
verified source is now frozen in
`analogy_agents/original_mapping_strength_prompts.py`, and all 148 original
structured responses are stored in `artifacts/mapping_strength_evidence/cache/`.

The submitted value is the original judge's structured recommendation:

```text
mapping = MappingExtractor_v1(TARGET, DESCRIPTION, ANALOGY)
judgment = MSJudge_v1(TARGET, DESCRIPTION, ANALOGY, mapping)
final_MS = judgment.recommended_score
```

Verify the full prompt → cache → final-column chain:

```bash
.venv/bin/python scripts/verify_mapping_strength_archive.py
```

Replay the archived test run without an API call:

```bash
.venv/bin/python run_text_agents.py \
  --mode mapping-strength \
  --split test \
  --cache-dir artifacts/mapping_strength_evidence/cache \
  --output-dir runs/mapping_strength_replay
```

This recreates
`runs/mapping_strength_replay/test_mapping_strength_predictions.csv`, which
matches the archived MS predictions for all 62 rows. Adding `--refresh-cache` with a
writable cache such as `.agent_cache` makes fresh Together calls; fresh
inference can differ despite identical prompts. Archived test distribution:
`0: 0`, `1: 12`, `2: 50`.

### M — Metaphoricity

M asks **how much semantic type translation is required between the literal
source mechanism and the target concept**. It ignores coverage, mapping quality,
analogy length, and fluency.

1. A source-domain analyzer reconstructs the smallest literal source mechanism
   after removing titles, parenthetical target glosses, and explicit “X is Y”
   wording. It identifies the central source/target roles and type changes.
2. A literal-instance judge checks whether that source mechanism is genuinely
   an instance, implementation, measurement, or application of the target.
3. A calibrated ordinal judge evaluates literal applicability, whether the two
   sides share a conventionally recognized native relation, and whether at most
   one central role changes semantic type. It uses 12 labeled validation
   anchors; validation evaluation physically leaves out the row being scored,
   while test inference uses all 12 anchors.
4. Python—not the model's free-form recommendation—sets the final score using
   this fixed rule:

```text
if literal_instance == yes:
    final_M = 0
elif native_relation_match == yes and role_change_degree == none_or_one:
    final_M = 1
else:
    final_M = 2
```

Interpretation:

- `0`: the source is a literal instance/application of the target;
- `1`: it is not literal, but an adjacent/native relation is preserved with at
  most one central role-type change;
- `2`: understanding requires cross-domain projection or multiple central
  role-type changes.

Archived test distribution: `0: 8`, `1: 10`, `2: 44`.

## 1. Install dependencies and run tests

The model pipeline and policy tests need the packages in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Equivalent test shortcut after creating the environment:

```bash
make test
```

The tests check the active TCC/M decision boundaries, all 148 archived MS prompt
and response records, validation leave-one-out behavior, and dataset row counts.

## 2. Run the active model agents

First complete the dependency installation above, then configure your own
Together API key:

```bash
cp .env.example .env
```

Edit `.env` so it contains your real key, then run:

```bash
bash scripts/run_agents.sh
```

This script:

1. runs `tcc-v1-facet-conservative` on all 62 test examples;
2. replays the exact archived `ms-v1` path on all 62 test examples;
3. runs `m` (`v7_1_role_audit_loo`) on all 62 test examples;
4. combines the three newly generated metric files into `submission.csv`.

The combiner is intentionally part of this full rerun. It has no default paths
to archived predictions, so the repository no longer exposes a separate command
for rebuilding a submission from previously saved metric files.

New model outputs are written under:

```text
runs/recomputed/target_coverage/ TCC details and predictions
runs/recomputed/mapping_strength/ MS mapping/judge details and predictions
runs/recomputed/metaphoricity/   M details and predictions
runs/recomputed/submission.csv   combined result of this full rerun
```

The rerun never overwrites the archived metric results. To force fresh calls
instead of reusing `.agent_cache/`:

```bash
REFRESH_CACHE=1 bash scripts/run_agents.sh
```

This can increase API cost. With `REFRESH_CACHE=1`, MS uses the recovered exact
v1 prompts but writes fresh responses to `.agent_cache`; it never overwrites
the archived evidence under `artifacts/mapping_strength_evidence/`.

## What is actually required?

The repository keeps two layers:

| Layer | Purpose | Main files |
| --- | --- | --- |
| Active model rerun | Calls TCC, MS, and M, then combines only those newly generated outputs | `run_text_agents.py`, `analogy_agents/`, `challenge-dataset/data/`, `requirements.txt`, `scripts/run_agents.sh`, `scripts/combine_recomputed_metrics.py` |
| Audit and traceability | Records the archived per-metric outputs, why the methods are trustworthy, and where they are limited | `artifacts/`, `tests/`, `docs/`, `manifest.json` |

Some source files under `analogy_agents/`, such as
`metaphoricity_pairwise.py` and the old taxonomy utilities, support earlier
experimental modes exposed by the shared runner. They do not determine the
active metric outputs and are retained only for source traceability.

## Repository contents

```text
README.md
    Setup, metric definitions, run instructions, and repository guide.

Makefile
    Shortcuts: `make test`, `make verify-mapping-strength`, and `make check`.

requirements.txt
    Python dependencies for model inference and tests. The small post-run
    combiner itself uses only the Python standard library.

.env.example
    API-key template. Copy it to the ignored `.env`; never commit a real key.

manifest.json
    Snapshot metadata, leaderboard values, component versions, and SHA-256
    hashes for the archived per-metric predictions.

run_text_agents.py
    CLI entry point for loading a split, selecting a scoring mode, running
    model agents, and writing per-example details and prediction CSVs.

analogy_agents/
    pipeline.py
        Data loading, Together API calls, caching, agent orchestration, fixed
        scoring rules, validation metrics, and output writers.
    prompts.py / schemas.py
        Current model prompts and their strict structured-output schemas.
    original_target_coverage_prompts.py / original_target_coverage_schemas.py
        Recovered original-v1 TCC prompt and schema definitions used by
        the active TCC path.
    original_mapping_strength_prompts.py / original_mapping_strength_schemas.py
        Hash-verified original-v1 MappingExtractor/MSJudge prompts and their
        structured-output contracts.
    metaphoricity_taxonomy.py / metaphoricity_pairwise.py
        Supporting utilities for earlier M experiments; they are retained for
        source traceability but are not part of the active metric paths.

challenge-dataset/data/
    The 12-row validation and 62-row test text Parquet files. Videos are
    intentionally excluded because the submitted video scores are zero.

artifacts/frozen/
    target_concept_coverage_predictions.csv
        Frozen active TCC predictions.
    mapping_strength_predictions.csv
        Frozen original Mapping Strength predictions.
    metaphoricity_predictions.csv
        Frozen active M predictions.
    *_validation_scores.json
        Preserved validation audit summaries.

artifacts/mapping_strength_evidence/
    The 148 original MappingExtractor/MSJudge cache records for 12 validation
    and 62 test rows, plus provenance and replay instructions.

scripts/
    run_agents.sh
        Runs the active TCC, MS, and M paths, then combines the new outputs.
    combine_recomputed_metrics.py
        Validates and combines only explicitly supplied, newly computed metric
        files; it has no archived-input defaults.
    verify_mapping_strength_archive.py
        Recomputes all v1 MS prompt hashes, validates archived schemas, and
        checks both validation and test predictions against archived records.

tests/
    test_pipeline.py
        Tests the active scoring policies and validation/data invariants.
    test_combine_recomputed_metrics.py
        Tests ID-based assembly of newly generated metric outputs.

docs/
    METHOD.md
        Exact TCC, MS, and M method snapshot.
    GENERALIZATION.md
        Generalization evidence, leakage checks, risks, and limitations.

output/, runs/, .agent_cache/
    Generated local files. These directories are ignored by Git.
```

## Scope and limitations

The archived metric outputs are a competition baseline, not proof of performance
on a new distribution. In particular, active TCC has no complete independent
validation run, M uses 12 validation anchors, and a fresh MS model call may
differ even though its archived run and prompt source are reproducible. See
[docs/METHOD.md](docs/METHOD.md) and
[docs/GENERALIZATION.md](docs/GENERALIZATION.md) for details.
