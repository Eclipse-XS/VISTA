SELECT
    run_id,
    flow_name,
    category_name,
    dataset_version,
    preprocessing_version,
    status AS run_status,
    CAST(started_at AS TIMESTAMPTZ) AS started_at,
    CAST(finished_at AS TIMESTAMPTZ) AS finished_at,
    duration_seconds,
    config_json,
    error_message
FROM read_parquet('{{ env_var("VISTA_GOLD_DIR") }}/pipeline_runs.parquet')
