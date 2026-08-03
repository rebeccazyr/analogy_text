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

Active method: the original-v1 `MappingExtractor` followed by `MSJudge`.

1. `MappingExtractor` identifies the source concept, explicit or clearly
   implied source-to-target correspondences, the shared process, and potential
   breaks. It does not assign a score.
2. `MSJudge` labels every extracted mapping `sound`, `stretch`, or
   `inconsistent`, records structural issues, and recommends the official
   ordinal score: `0` for far-fetched/highly inconsistent mappings, `1` for
   some stretches or inconsistencies, and `2` for well-aligned and consistent
   mappings.
3. The final MS column is `MSJudge.recommended_score`; there is no manual or
   ID-specific correction.

The original archive stored prompt hashes rather than a separate prompt-source
copy. The surviving templates have now been verified against all archived
calls: 74/74 MappingExtractor hashes and 74/74 MSJudge hashes match. The repo
also contains all 148 original responses.
`scripts/verify_mapping_strength_archive.py` checks the prompt, schema, cache,
validation prediction, and frozen test-column chain.

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
