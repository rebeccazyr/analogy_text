# rrrrr1030 best text submission

This repository is a reproducible snapshot of the best known text-only
submission:

| Text Kendall | Text Spearman | Video Kendall | Video Spearman |
| ---: | ---: | ---: | ---: |
| 0.4109 | 0.4267 | 0 | 0 |

The final file combines:

- TCC: `tcc_v1_facet_conservative_v1`
- MS: frozen original v1 prediction
- M: `v7_1_role_audit_loo`
- VC/VA/VE: zero

## Reproduce the submitted CSV

No API or third-party package is required for the deterministic build:

```bash
python3 scripts/build_submission.py
```

or:

```bash
make reproduce
```

Generated files:

```text
output/submission.csv
output/build_audit.json
```

The builder validates IDs, schemas, score ranges, video zeros, source hashes,
and value-for-value equality with the frozen known-good file. Expected SHA-256:

```text
eaaed257e856be97f59601dd17ae41f3bccba9356cb140bdd862efe7a38293ee
```

## Run the model agents again

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Put your own Together API key in .env.
bash scripts/run_agents.sh
```

This reruns the active TCC and M paths into `runs/recomputed/` and then checks
whether the new predictions still reproduce the frozen submission. It does not
overwrite the frozen artifacts. Setting `REFRESH_CACHE=1` forces fresh model
calls and can incur API cost.

MS cannot be regenerated exactly because its original v1 prompt source was not
preserved independently. The audited prediction is intentionally frozen.

## Test

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Repository map

```text
analogy_agents/       agent prompts, schemas, and scoring pipeline
challenge-dataset/    text parquet splits only; videos are not needed
artifacts/frozen/     immutable component predictions and known-good CSV
scripts/              deterministic builder and agent runners
tests/                local policy and reproduction tests
docs/                 method and generalization notes
manifest.json         versions, leaderboard values, and SHA-256 hashes
```

See [docs/METHOD.md](docs/METHOD.md) for the active method and
[docs/GENERALIZATION.md](docs/GENERALIZATION.md) for the audit limitations.
