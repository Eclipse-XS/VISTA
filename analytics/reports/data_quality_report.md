# VISTA — Multimodal Data Quality, Leakage & Operational Telemetry Report

- **Generated (UTC):** `2026-11-20T10:28:11.920290+00:00`
- **Immutable Gold Snapshot ID:** `snap_1_0_0_silver_v1_172131d3_907737715593`
- **Snapshot Manifest SHA-256:** `e7922b010b8788a8a33b351a17541dca9a1d05dc744d237894bc3fc8b711c97b`
- **Operational Telemetry Cutoff (UTC):** `2026-11-20T10:27:50.491166+00:00`
- **Telemetry Consistency Audit:** `PASS (0 inconsistencies)`
- **Dataset:** `mvtec_3d_ad` version `1.0.0`
- **Ingested Categories:** `cable_gland`
- **Total Samples Validated:** `354`

## 1. Validation Status Breakdown (Q2)

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
| analytics | 2 | 1 | 0 | 0 | 0 | 15.1660 |
| discover | 8 | 5 | 3 | 0 | 3 | 10.7970 |
| ingest | 6 | 5 | 1 | 0 | 1 | 19.0940 |
| publish_gold | 8 | 2 | 5 | 1 | 4 | 179.7160 |
| quality_metrics | 4 | 4 | 0 | 0 | 0 | 18.6730 |
| transform | 6 | 4 | 1 | 1 | 1 | 100.7070 |
| validate | 5 | 5 | 0 | 0 | 0 | 58.3910 |
