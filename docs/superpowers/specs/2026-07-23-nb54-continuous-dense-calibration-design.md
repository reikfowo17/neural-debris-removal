# NB54 Continuous Dense Calibration

## Objective

Build one standalone Kaggle notebook that reproduces the proven NB51 training and ranking pipeline, then replaces its discontinuous two-level confidence composer with dense, continuous confidence calibration. The primary goal is to improve on LB 180.3030 without removing any detector candidate or sacrificing the coverage whose value was demonstrated by NB52.

## Pipeline

1. Run the proven NB51 setup, legacy-ranker training, real-anchor mining, four source-disjoint scrub folds, and hard gates end to end.
2. Collect teacher confidence, legacy poison response, and cross-fitted scrub poison response for every test candidate.
3. Form the proven A065 poison score: `0.35 * legacy + 0.65 * scrub`.
4. Rank all candidates globally from poison-like to clean-like with deterministic tie-breaking.
5. Convert clean percentile rank into confidence continuously between `0.02` and `min(1, teacher_confidence + 0.15)`.
6. Apply only a mild soft duplicate penalty to lower-ranked candidates that overlap a stronger clean-like candidate. Never remove a candidate.
7. Export three gamma variants after all gates and structural assertions pass.

## Calibration

For clean percentile `q` in `[0, 1]`:

`confidence = 0.02 + q^gamma * (min(1, teacher_confidence + 0.15) - 0.02)`

Candidates with lower poison score receive larger `q`. The transform is deterministic and monotonic in clean rank for equal teacher confidence. Confidence remains in `[0.02, 1.0]`.

The variants are:

- `CONTINUOUS_G2`, gamma 2.0, primary.
- `CONTINUOUS_G1`, gamma 1.0, aggressive.
- `CONTINUOUS_G3`, gamma 3.0, conservative hedge.

## Duplicate Damping

Within each image, compare candidates in clean-rank order. A candidate receives a mild multiplicative damping only if it substantially overlaps a stronger clean-like candidate. The floor remains `0.02`, the stronger candidate is unchanged, and no candidate is omitted. This preserves isolated tail coverage and avoids reproducing NB52 sparse behavior.

## Safety And Audit

No CSV is exported unless every NB51 training gate passes. Before export, assert that each variant preserves every baseline candidate and box coordinate, has the same per-image candidate counts, contains finite confidence in `[0.02, 1.0]`, and is deterministic under stable ties. The report records confidence quantiles, total confidence mass, duplicate-damped count, and candidate count for each variant.

The OOF audit can verify ranking preservation but cannot validate hidden-clean confidence calibration. The primary CSV is therefore an explicitly controlled leaderboard experiment. NB51 LB 180.3030 remains the fallback.

## Submission Policy

Submit `nb54_CONTINUOUS_G2.csv` first. Do not submit the hedge files until its LB is known. Use the reset submissions conditionally rather than submitting all variants blindly.
