-- Fails if operational stage telemetry contains inconsistent statuses or invalid durations.
SELECT
    stage_run_id,
    run_id,
    stage_name,
    stage_status,
    run_status,
    duration_seconds
FROM {{ ref('fct_ingestion_runs') }}
WHERE (stage_status = 'SUCCEEDED' AND (duration_seconds IS NULL OR duration_seconds < 0))
   OR (stage_status = 'RUNNING' AND run_status IN ('SUCCEEDED', 'FAILED', 'INTERRUPTED'))
   OR attempt_number < 1
