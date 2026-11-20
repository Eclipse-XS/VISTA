SELECT
    stage_run_id,
    run_id,
    stage_name,
    category_name,
    attempt_number,
    is_retry,
    status AS stage_status,
    CAST(started_at AS TIMESTAMPTZ) AS started_at,
    CAST(finished_at AS TIMESTAMPTZ) AS finished_at,
    duration_seconds,
    items_processed,
    items_failed,
    checkpoint_json,
    error_message
FROM read_parquet('{{ env_var("VISTA_TELEMETRY_DIR", env_var("VISTA_GOLD_DIR")) }}/stage_runs.parquet')
