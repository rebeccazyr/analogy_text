# Label-safe text augmentation

This pipeline deliberately does not assign new absolute `0/1/2` pseudo-labels.
It produces two auditable datasets:

1. `gold_preserving_invariance`: surface rewrites of original labeled rows. A
   rewrite inherits gold labels only after unanimous blind confirmation that
   TCC, MS, and M are all unchanged.
2. `relative_counterfactual_pair`: A/B pairs for new concepts with only an
   operation-defined `higher/lower/same` relation for one metric. The other two
   metrics must remain invariant.

Run a small pilot from the repository root:

```bash
python scripts/generate_augmentation_data.py \
  --validation-ids 0,1,4 \
  --concept-ids C001,C002,C005 \
  --output-dir data_augmentation/pilot
```

The script uses the parent workspace `.env` when available, caches every model
response, and keeps raw generations, all reviews, accepted rows, disagreements,
and rejections. It is safe to resume without `--refresh`.

Important outputs:

```text
data_augmentation/pilot/accepted/gold_invariance.jsonl
data_augmentation/pilot/accepted/relative_pairs.jsonl
data_augmentation/pilot/reviewed/invariance_all.jsonl
data_augmentation/pilot/reviewed/relative_all.jsonl
data_augmentation/pilot/quality_report.json
```

The accepted relative pairs are also split by `concept_id`; variants from one
concept never cross train, validation, and test partitions.
