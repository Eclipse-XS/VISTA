# VISTA — Multimodal Data Quality, Leakage & Operational Telemetry Report

- **Generated (UTC):** `2026-11-24T20:33:57.095491+00:00`
- **Immutable Gold Snapshot ID:** `snap_1_0_0_silver_v1_172131d3_a3cd04efd735`
- **Snapshot Manifest SHA-256:** `1c50ca3d4956c6fc1c024d4a9ea7126c14364687c909fedc155d2fc5061b2440`
- **Operational Telemetry Cutoff (UTC):** `2026-11-24T20:33:13.666710+00:00`
- **Telemetry Consistency Audit:** `PASS (0 inconsistencies)`
- **Dataset:** `mvtec_3d_ad` version `1.0.0`
- **Ingested Categories:** `cable_gland, cookie, dowel`
- **Total Samples Validated:** `1169`

## 1. Validation Status Breakdown (Q2)

| Status | Sample Count | Share |
| --- | ---: | ---: |
| `PASS` | 814 | 69.63% |
| `WARN` | 355 | 30.37% |
| `QUARANTINE` | 0 | 0.00% |
| `FAIL` | 0 | 0.00% |

### Validation Rule Breakdown by Category and Status

Source SQL: `analytics/sql/02_validation_failures.sql`

| category_name | total_samples | pass_count | pass_rate_pct | warn_count | quarantine_count | fail_count | misaligned_count | exact_duplicate_count | near_duplicate_count | cross_split_leak_count | non_finite_3d_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cookie | 363 | 248 | 68.3200 | 115 | 0 | 0 | 0 | 0 | 115 | 0 | 0 |
| dowel | 452 | 313 | 69.2500 | 139 | 0 | 0 | 0 | 0 | 139 | 0 | 0 |
| cable_gland | 354 | 253 | 71.4700 | 101 | 0 | 0 | 0 | 0 | 101 | 0 | 0 |


## 2. Leakage Prevention & ML Readiness Contracts

- **Train-only normalization (`normalization_artifacts`):** fitted strictly on eligible `split = 'train'` samples (`status IN ('PASS', 'WARN')`, `is_cross_split_leak = FALSE`). Frozen parameters are applied identically to `train`, `validation`, and `test`.
- **ML partitions:**
  - `training_ready_samples.parquet` / `mart_training_ready_samples`: `train` + `validation` ONLY (`verified test rows = 0`). Excludes `defect_type`, `is_anomalous_benchmark`, and within-split descriptive EDA z-scores.
  - `evaluation_samples.parquet`: held-out `test` partition explicitly tagged `usage = 'benchmark_evaluation_only'`.
- **Immutability:** Bronze archives and committed Gold snapshots (`gold/mvtec_3d_ad/snapshots/<snapshot_id>/`) are immutable and SHA-256 verified.

## 3. Authoritative Operational Telemetry Summary

Canonical statuses:
- Terminal: `SUCCEEDED`, `FAILED`, `INTERRUPTED`, `SKIPPED`
- Intermediate: `RUNNING`
- Duration averages exclude `NULL` durations (never coerced to `0`). Historical failures and retries are preserved transparently.

| stage_name | total_attempts | succeeded | failed | interrupted | retries | avg_succeeded_s |
| --- | --- | --- | --- | --- | --- | --- |
| analytics | 4 | 4 | 0 | 0 | 0 | 31.9110 |
| discover | 10 | 7 | 3 | 0 | 3 | 14.9610 |
| ingest | 8 | 7 | 1 | 0 | 1 | 119.4940 |
| publish_gold | 11 | 5 | 5 | 1 | 4 | 356.7740 |
| quality_metrics | 6 | 6 | 0 | 0 | 0 | 31.4270 |
| transform | 8 | 6 | 1 | 1 | 1 | 257.7710 |
| validate | 7 | 7 | 0 | 0 | 0 | 75.0440 |
