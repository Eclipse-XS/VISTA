-- Q9: Authoritative pipeline stage execution performance (separates statuses, retries, and non-NULL durations).
SELECT
    stage_name,
    COUNT(*) AS total_attempts,
    COUNT(DISTINCT run_id) AS logical_runs,
    SUM(CASE WHEN stage_status = 'SUCCEEDED' THEN 1 ELSE 0 END) AS succeeded_attempts,
    SUM(CASE WHEN stage_status = 'FAILED' THEN 1 ELSE 0 END) AS failed_attempts,
    SUM(CASE WHEN stage_status = 'INTERRUPTED' THEN 1 ELSE 0 END) AS interrupted_attempts,
    SUM(CASE WHEN stage_status = 'RUNNING' THEN 1 ELSE 0 END) AS in_progress_attempts,
    SUM(CASE WHEN is_retry THEN 1 ELSE 0 END) AS retried_attempts,
    ROUND(SUM(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL), 3) AS total_observed_duration_s,
    ROUND(AVG(duration_seconds) FILTER (WHERE stage_status = 'SUCCEEDED' AND duration_seconds IS NOT NULL), 3) AS avg_succeeded_duration_s,
    ROUND(MAX(duration_seconds) FILTER (WHERE stage_status = 'SUCCEEDED' AND duration_seconds IS NOT NULL), 3) AS max_succeeded_duration_s,
    SUM(items_processed) FILTER (WHERE stage_status = 'SUCCEEDED') AS succeeded_items_processed,
    ROUND(
        SUM(items_processed) FILTER (WHERE stage_status = 'SUCCEEDED' AND duration_seconds > 0)
        / NULLIF(SUM(duration_seconds) FILTER (WHERE stage_status = 'SUCCEEDED' AND items_processed > 0 AND duration_seconds > 0), 0),
        2
    ) AS succeeded_items_per_sec
FROM main.fct_ingestion_runs
GROUP BY stage_name
ORDER BY total_observed_duration_s DESC NULLS LAST, stage_name;
