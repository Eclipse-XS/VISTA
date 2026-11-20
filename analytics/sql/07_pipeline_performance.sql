-- Q9: Which ingestion or transformation stages consume the most time?
SELECT
    stage_name,
    COUNT(*) AS execution_count,
    SUM(CASE WHEN stage_status = 'SUCCEEDED' THEN 1 ELSE 0 END) AS succeeded_runs,
    SUM(CASE WHEN stage_status = 'FAILED' THEN 1 ELSE 0 END) AS failed_runs,
    ROUND(SUM(COALESCE(duration_seconds, 0)), 3) AS total_duration_s,
    ROUND(AVG(COALESCE(duration_seconds, 0)), 3) AS avg_duration_s,
    ROUND(MAX(COALESCE(duration_seconds, 0)), 3) AS max_duration_s,
    SUM(items_processed) AS total_items_processed,
    ROUND(AVG(items_per_second), 2) AS avg_items_per_sec
FROM main.fct_ingestion_runs
GROUP BY stage_name
ORDER BY total_duration_s DESC;
