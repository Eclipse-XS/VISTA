-- Ensures mart_training_ready_samples preserves strict 1-row-per-sample_pk grain
-- (catches accidental many-to-many join multiplication across prep/validation versions).
SELECT
    sample_pk,
    COUNT(*) AS row_count
FROM {{ ref('mart_training_ready_samples') }}
GROUP BY sample_pk
HAVING COUNT(*) > 1
