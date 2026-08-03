# Method snapshot

The submitted text score is the average of three independently predicted
ordinal metrics. Video metrics are fixed to zero.

## TCC

Active policy: `tcc_v1_facet_conservative_v1`.

1. Run the recovered original v1 concept decomposition and TCC judgment.
2. Classify description topics as independent, overlapping, or contextual.
3. Audit unresolved blockers as atomic facets.
4. Python permits only a conservative `1 -> 2` promotion when all retained
   substantive and scope facets are covered.

The active path begins at
`SixAgentPipeline.evaluate_exact_v1_tcc_with_facet_audit` in
`analogy_agents/pipeline.py`. The facet contract is in
`facet_coverage_audit_prompt` in `analogy_agents/prompts.py`.

## MS

Active artifact: the original v1 predictions. The pipeline extracts explicit
source-to-target mappings and judges their logical soundness independently of
coverage and conceptual distance.

The exact original MS v1 prompt source was not independently preserved at the
time of the run. For honest reproducibility, this repository treats the
audited 62-row MS prediction as an immutable input rather than claiming it can
be regenerated exactly.

## M

Active policy: `v7_1_role_audit_loo`.

1. Reconstruct the literal source operation.
2. Apply a strict literal-instance gate.
3. Check whether source and target preserve a native relation.
4. Count only central ontological role changes.
5. Apply the fixed Python boundary:

```text
literal instance                                      -> 0
native relation and at most one central role change   -> 1
otherwise                                             -> 2
```

Validation uses physical leave-one-out calibration anchors. Test inference
uses all 12 validation anchors.

## Frozen output distributions

| Metric | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| TCC | 0 | 32 | 30 |
| MS | 0 | 12 | 50 |
| M | 8 | 10 | 44 |
