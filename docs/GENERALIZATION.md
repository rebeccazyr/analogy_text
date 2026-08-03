# Generalization audit

## What was checked

- No active TCC or M scoring branch hard-codes test IDs.
- Example IDs are used for cache addressing and validation leave-one-out only.
- The active TCC correction is one sample-agnostic `1 -> 2` policy.
- M is converted to `0/1/2` by a fixed Python rule with no manual overrides.
- Frozen component hashes and final submission hash are verified at build time.

## Evidence

| Component | Validation Kendall | Public Kendall | Assessment |
| --- | ---: | ---: | --- |
| TCC facet | not independently measured | 0.3358 | medium risk |
| MS v1 | 0.3714 | 0.4017 | lowest risk |
| M v7.1 | 0.8666 | 0.4952 | medium risk |

## Known limitations

1. The exact active TCC facet policy has no complete independent validation
   run. Two reasonable conservative TCC policies disagree on 10 of 62 rows.
2. M uses only 12 labeled validation anchors. Seven test rows share a target
   with validation, and five share the exact target and description.
3. Several method versions were compared using validation or public results,
   so leave-one-out does not remove method-selection bias.
4. The archived MS run, prompts, and structured responses are reproducible,
   but a fresh model call can still differ because inference is not guaranteed
   to be deterministic.

The frozen file is suitable as the current competition baseline. It should not
be presented as proof of out-of-distribution performance on new datasets.
Further public-leaderboard tuning should be avoided.
