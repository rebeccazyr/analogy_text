# Generalization audit

## What was checked

- No active TCC, MS, or M scoring branch hard-codes test IDs.
- IDs are used only for dataset alignment, cache addressing, and validation
  leave-one-out.
- TCC has one fixed `1 -> 2` correction policy.
- MS v8 has one fixed two-of-three zero gate over the frozen v1 baseline.
- M is converted to `0/1/2` by a fixed Python boundary.
- Frozen per-metric predictions and exact leaderboard metrics are tracked.

## Evidence

| Component | Validation Kendall | Public Kendall | Public Spearman |
| --- | ---: | ---: | ---: |
| TCC facet, medium | not independently measured | 0.3359301817 | 0.3359301817 |
| MS v8 zero gate, medium | 0.7655473322 | 0.4280255989 | 0.4407342051 |
| M v7.1, high | 0.7453559925 | 0.5909945085 | 0.6266820770 |

## Known limitations

1. The exact active TCC facet policy has no complete independent validation
   run.
2. M uses only 12 labeled validation anchors. Seven test rows share a target
   with validation, and five share the exact target and description.
3. Several method versions were compared using validation or public results.
   Leave-one-out does not remove method-selection bias.
4. The MS v8 prompt and scoring rule are sample-independent, but v8 was selected
   after public leaderboard feedback suggested two original-v1 false positives.
   Replacing the manual candidate with a prompt does not erase that selection
   history.
5. Fresh model inference is not guaranteed to reproduce the frozen labels even
   when prompts, model, reasoning effort, and seed are unchanged.

The archived results are a competition snapshot and should not be presented as
proof of performance on a new distribution.
