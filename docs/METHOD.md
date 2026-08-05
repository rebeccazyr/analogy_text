# Metric method snapshot

Snapshot: 2026-08-04. Model: `openai/gpt-oss-120b`.

| Metric | Method | Reasoning effort |
| --- | --- | --- |
| TCC | `tcc_v1_facet_conservative_v1` | medium |
| MS | `ms_v3_counterfactual_zero_gate_v8` | medium |
| M | `v7_1_role_audit_loo` | high |

Only M uses high reasoning; all other active text inference uses medium.

## TCC

1. Run the recovered original-v1 concept decomposition and TCC judgment.
2. Classify description topics as independent, overlapping, or contextual.
3. Audit unresolved blockers as atomic facets.
4. Permit only a conservative `1 -> 2` promotion when all retained
   substantive and scope facets are covered.

The active path begins at
`SixAgentPipeline.evaluate_exact_v1_tcc_with_facet_audit`.

## MS

1. Load the hash-verified medium-reasoning v1 MappingExtractor and MSJudge
   evidence as the baseline.
2. Reconstruct the literal source three times without revealing the target.
   Quarantine non-native target labels, equations, calculations, and causal
   operations.
3. Run three target-aware counterfactual zero-gate audits.
4. Convert each audit with the fixed `ms_score_from_zero_gate` function.
5. Change the baseline to zero only when at least two votes are zero; otherwise
   preserve the v1 score.

The zero gate detects only decisive target injection, missing recursive
identity, impossible source operations, or reversed core relations. It cannot
promote a score and does not distinguish the baseline `1/2` boundary.

Test changes relative to v1:

| ID | Target | v1 | v8 | Votes |
| ---: | --- | ---: | ---: | --- |
| 15 | recursion | 1 | 0 | `[1,0,0]` |
| 22 | backpropagation | 1 | 0 | `[0,0,0]` |

There is no manual or ID-specific correction in the active code.

## M

1. Reconstruct the literal source operation.
2. Apply a strict literal-instance gate.
3. Check whether source and target preserve a native relation.
4. Count only central ontological role changes.
5. Apply the fixed boundary:

```text
literal instance                                      -> 0
native relation and at most one central role change   -> 1
otherwise                                             -> 2
```

Validation anchors are physically leave-one-out. Test inference uses all 12
anchors. The active test run uses high reasoning.

Experimental `m-cosine` keeps the same literal-instance gate for `M=0`, then
uses a configurable weighted average of concept and domain `1-cosine`
distances for the `M=1/2` boundary. It uses no ordinal LLM judge after the
structured domain and literal evidence has been extracted. Embeddings run
locally through sentence-transformers (`BAAI/bge-large-en-v1.5` by default),
while Together is used only for those two structured LLM calls. The initial
weight and threshold are research defaults and must be evaluated on validation
before any test run; this mode is not part of the frozen submission.

The follow-up `m-features` ablation keeps that literal gate and compares five
pre-registered feature groups. E1 uses mechanism cosine only; E2 uses the
LLM-extracted native-relation mismatch only; E3 combines mechanism and
relation; E4 adds role-type shift; and E5 adds low-weight concept and domain
cosines. Cosine distances are clipped to `[0,1]`; categorical evidence uses
`0`, `0.5`, and `1` for M1-directed, unclear, and M2-directed values. One run
shares the same two LLM calls and local embeddings across all five variants,
so the ablation itself does not multiply API calls.

## Frozen output distributions

| Metric | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| TCC | 0 | 32 | 30 |
| MS | 2 | 10 | 50 |
| M | 8 | 6 | 48 |
