-- Grain: 1 row per pipeline stage attempt joined to parent pipeline run
SELECT
    st.stage_run_id,
    st.run_id,
    pr.flow_name,
    COALESCE(st.category_name, pr.category_name) AS category_name,
    pr.dataset_version,
    pr.preprocessing_version,
    st.stage_name,
    st.attempt_number,
    st.is_retry,
    st.stage_status,
    pr.run_status,
    st.started_at,
    st.finished_at,
    st.duration_seconds,
    st.items_processed,
    st.items_failed,
    CASE
        WHEN st.stage_status = 'SUCCEEDED'
         AND st.duration_seconds IS NOT NULL
         AND st.duration_seconds > 0
         AND st.items_processed > 0
        THEN st.items_processed / st.duration_seconds
        ELSE NULL
    END AS items_per_second,
    st.error_message
FROM {{ ref('stg_stage_runs') }} st
LEFT JOIN {{ ref('stg_pipeline_runs') }} pr
    ON st.run_id = pr.run_id
