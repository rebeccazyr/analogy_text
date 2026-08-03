# rrrrr1030 best text submission

This repository is a self-contained, reproducible snapshot of the best known
text-only submission for `rrrrr1030`.

| Text Kendall | Text Spearman | Video Kendall | Video Spearman |
| ---: | ---: | ---: | ---: |
| 0.4109 | 0.4267 | 0 | 0 |

The submitted file combines:

- TCC: `tcc_v1_facet_conservative_v1`
- MS: frozen original-v1 prediction
- M: `v7_1_role_audit_loo`
- VC, VA, VE: all zero

## Choose the run you need

There are two different workflows:

1. **Reproduce the submitted CSV** from frozen, audited predictions. This is
   deterministic, free, and does not call an API.
2. **Run the model agents again** for TCC and M. This needs a Together API key,
   incurs API cost, and may differ because model inference is not guaranteed to
   be deterministic.

Run all commands below from the repository root. Python 3.10 or newer is
recommended.

## How the three text metrics are calculated

Every row has the same three inputs: `TARGET`, its reference `DESCRIPTION`, and
the generated `ANALOGY`. TCC, MS, and M are predicted independently and remain
three separate columns in the submission; their raw ordinal values are not
averaged together by this repository.

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
changes `0`, or uses an example-ID override. Frozen test distribution:
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

For this best submission, the exact original-v1 MS prompt was not preserved
independently. The honest reproducible rule in this repository is therefore:

```text
final_MS(id) = audited frozen original-v1 MS prediction for that id
```

The repository does not claim that MS can be regenerated exactly from the
current prompt source. Frozen test distribution: `0: 0`, `1: 12`, `2: 50`.

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

Frozen test distribution: `0: 8`, `1: 10`, `2: 44`.

### Final CSV assembly

The deterministic builder joins the three predictions by `id` and sets the
unused video metrics to zero:

```text
id,TCC,MS,M,VC,VA,VE
...,tcc_score,ms_score,m_score,0,0,0
```

## 1. Reproduce the exact submitted CSV

No virtual environment, API key, or third-party package is required:

```bash
python3 scripts/build_submission.py
```

Equivalent shortcut:

```bash
make reproduce
```

The command reads the three frozen text components, joins them by `id`, and
creates:

```text
output/submission.csv       final 62-row competition file
output/build_audit.json     source hashes, distributions, and verification result
```

The builder checks the ID set, columns, score ranges, zero video fields, and
value-for-value equality with the known-good submission. The expected SHA-256
of `output/submission.csv` is:

```text
eaaed257e856be97f59601dd17ae41f3bccba9356cb140bdd862efe7a38293ee
```

## 2. Install dependencies and run tests

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

The tests check the active TCC/M decision boundaries, validation leave-one-out
behavior, dataset row counts, and exact frozen-submission reproduction.

## 3. Run the active model agents again

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
2. runs `m` (`v7_1_role_audit_loo`) on all 62 test examples;
3. combines those new predictions with the frozen MS predictions;
4. compares the result with the known-good submission.

New model outputs are written under:

```text
runs/recomputed/tcc/             TCC details and predictions
runs/recomputed/m/               M details and predictions
runs/recomputed/submission.csv   recomputed combined submission
runs/recomputed/build_audit.json recomputed verification report
```

The rerun never overwrites `artifacts/frozen/`. If new model calls produce
different labels, the final comparison exits with an error by design; the new
files remain under `runs/recomputed/` for inspection. To force fresh calls
instead of reusing `.agent_cache/`:

```bash
REFRESH_CACHE=1 bash scripts/run_agents.sh
```

This can increase API cost. MS is not rerun: its exact original-v1 prompt was
not independently preserved, so the audited MS predictions are treated as an
immutable input.

## What is actually required?

Not every tracked file participates in the final score. The repository keeps
three layers so that the result can be reproduced and audited honestly:

| Layer | Purpose | Main files |
| --- | --- | --- |
| Exact submission | Required to rebuild the leaderboard CSV without an API | `artifacts/frozen/*.csv`, `scripts/build_submission.py` |
| Active model rerun | Required to call the active TCC and M agents again | `run_text_agents.py`, `analogy_agents/`, `challenge-dataset/data/`, `requirements.txt`, `scripts/run_agents.sh` |
| Audit and traceability | Not needed to construct the CSV, but records why the snapshot is trustworthy and where it is limited | `tests/`, `docs/`, `manifest.json`, validation-score JSON files |

Some source files under `analogy_agents/`, such as `m_pairwise.py` and the old
taxonomy utilities, support earlier experimental modes exposed by the shared
runner. They do not determine the frozen best submission and are retained only
for source traceability.

## Repository contents

```text
README.md
    Setup, reproduction, rerun, and repository guide.

Makefile
    Shortcuts: `make reproduce`, `make test`, and `make check`.

requirements.txt
    Python dependencies for model inference and tests. The deterministic CSV
    builder itself uses only the Python standard library.

.env.example
    API-key template. Copy it to the ignored `.env`; never commit a real key.

manifest.json
    Snapshot metadata, leaderboard values, component versions, and SHA-256
    hashes for the frozen artifacts.

run_text_agents.py
    CLI entry point for loading a split, selecting a scoring mode, running
    model agents, and writing per-example details and prediction CSVs.

analogy_agents/
    pipeline.py
        Data loading, Together API calls, caching, agent orchestration, fixed
        scoring rules, validation metrics, and output writers.
    prompts.py / schemas.py
        Current model prompts and their strict structured-output schemas.
    v1_prompts.py / v1_schemas.py
        Recovered, frozen original-v1 TCC prompt and schema definitions used by
        the active TCC path.
    m_taxonomy.py / m_pairwise.py
        Supporting utilities for earlier M experiments; they are retained for
        source traceability but are not part of the final frozen build.

challenge-dataset/data/
    The 12-row validation and 62-row test text Parquet files. Videos are
    intentionally excluded because the submitted video scores are zero.

artifacts/frozen/
    tcc_predictions.csv
        Frozen active TCC predictions.
    ms_v1_base_submission.csv
        Frozen original-v1 MS values plus zero video columns.
    m_v7_1_predictions.csv
        Frozen active M predictions.
    known_good_submission.csv
        Exact leaderboard submission used as the reproduction oracle.
    *_validation_scores.json
        Preserved validation audit summaries.

scripts/
    build_submission.py
        Deterministically merges and validates the three frozen components.
    reproduce.sh
        Shell wrapper for the deterministic builder.
    run_agents.sh
        Reruns only the active TCC and M inference paths, then verifies them.

tests/
    test_pipeline.py
        Tests the active scoring policies and validation/data invariants.
    test_reproduction.py
        Tests byte-stable reconstruction of the known-good submission.

docs/
    METHOD.md
        Exact TCC, MS, and M method snapshot.
    GENERALIZATION.md
        Generalization evidence, leakage checks, risks, and limitations.

output/, runs/, .agent_cache/
    Generated local files. These directories are ignored by Git.
```

## Scope and limitations

The frozen file is the current competition baseline, not proof of performance
on a new distribution. In particular, active TCC has no complete independent
validation run, M uses 12 validation anchors, and MS can be reproduced only as
a frozen artifact. See [docs/METHOD.md](docs/METHOD.md) and
[docs/GENERALIZATION.md](docs/GENERALIZATION.md) for details.
