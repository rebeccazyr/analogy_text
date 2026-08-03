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
