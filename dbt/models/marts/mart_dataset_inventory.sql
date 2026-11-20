-- Grain: 1 row per category x split
SELECT
    s.category_name,
    s.material_type,
    s.split,
    COUNT(*) AS sample_count,
    SUM(CASE WHEN v.status = 'PASS' THEN 1 ELSE 0 END) AS passed_count,
    SUM(CASE WHEN v.status = 'WARN' THEN 1 ELSE 0 END) AS warned_count,
    SUM(CASE WHEN v.status = 'QUARANTINE' THEN 1 ELSE 0 END) AS quarantined_count,
    SUM(CASE WHEN v.status = 'FAIL' THEN 1 ELSE 0 END) AS failed_count,
    ROUND(100.0 * SUM(CASE WHEN v.status = 'PASS' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS pass_pct,
    ROUND(SUM(s.total_raw_bytes) / (1024.0 * 1024.0), 2) AS raw_size_mb,
    SUM(CASE WHEN sa.artifact_id IS NOT NULL THEN 1 ELSE 0 END) AS silver_ready_count,
    SUM(CASE WHEN sa.normal_reliable THEN 1 ELSE 0 END) AS reliable_normal_count
FROM {{ ref('dim_sample') }} s
LEFT JOIN {{ ref('fct_validation_results') }} v
    ON s.sample_pk = v.sample_pk
LEFT JOIN {{ ref('stg_silver_artifacts') }} sa
    ON s.sample_pk = sa.sample_pk
GROUP BY
    s.category_name,
    s.material_type,
    s.split
ORDER BY
    s.category_name,
    s.split
