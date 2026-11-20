# VISTA — Multimodal Data Quality & Validation Report

- **Generated (UTC):** `2026-11-20T00:24:08.497040+00:00`
- **Dataset:** `mvtec_3d_ad` version `1.0.0`
- **Ingested Categories:** `cable_gland`
- **Total Samples Validated:** `354`

## Validation Status Breakdown (Q2)

| Status | Sample Count | Share |
| --- | ---: | ---: |
| `PASS` | 253 | 71.47% |
| `WARN` | 101 | 28.53% |
| `QUARANTINE` | 0 | 0.00% |
| `FAIL` | 0 | 0.00% |

### Validation Rule Breakdown by Category and Status

Source SQL: `analytics/sql/02_validation_failures.sql`

| category_name | total_samples | pass_count | pass_rate_pct | warn_count | quarantine_count | fail_count | misaligned_count | exact_duplicate_count | near_duplicate_count | cross_split_leak_count | non_finite_3d_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cable_gland | 354 | 253 | 71.4700 | 101 | 0 | 0 | 0 | 0 | 101 | 0 | 0 |


## Split Integrity & Label-Leakage Safeguards

- **Training-ready Gold dataset (`training_ready_samples.parquet`) & dbt mart (`mart_training_ready_samples`)**:
  - Includes ONLY `split IN ('train', 'validation')` with `validation_status IN ('PASS', 'WARN')` and `is_cross_split_leak = FALSE`.
  - Excludes all `test` split rows (`verified test rows in mart = 0`).
  - Drops `defect_type` and `is_anomalous_benchmark` columns entirely from the training export schema.
- **Immutability:** Bronze source archives and sample manifests are never modified or silently repaired. Any detected corruption or split violation is routed to `quarantine/mvtec_3d_ad/<category>/quarantine_manifest.json`.

## Pipeline Stage Summary

| stage_name | runs | avg_duration_s | total_duration_s |
| --- | --- | --- | --- |
| ingest | 5 | 113.6839 | 568.4196 |
| publish_gold | 7 | 103.7071 | 518.5355 |
| transform | 5 | 125.3072 | 501.2289 |
| validate | 4 | 59.8896 | 239.5582 |
| discover | 7 | 11.7644 | 82.3508 |
| quality_metrics | 3 | 15.9947 | 47.9840 |
