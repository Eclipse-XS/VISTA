-- Grain: 1 row per category x defect_type in labeled benchmark TEST split ONLY.
-- NOTE: Benchmark test proportions reflect dataset curation, NOT real factory defect rates.
WITH test_samples AS (
    SELECT
        s.sample_pk,
        s.category_name,
        s.defect_type,
        s.is_anomalous_benchmark
    FROM {{ ref('dim_sample') }} s
    WHERE s.split = 'test'
),
category_totals AS (
    SELECT
        category_name,
        COUNT(*) AS category_test_total
    FROM test_samples
    GROUP BY category_name
)
SELECT
    t.category_name,
    t.defect_type,
    t.is_anomalous_benchmark,
    COUNT(*) AS sample_count,
    ct.category_test_total,
    ROUND(100.0 * COUNT(*) / NULLIF(ct.category_test_total, 0), 2) AS benchmark_test_share_pct,
    AVG(q.rgb_brightness_mean) AS mean_rgb_brightness,
    AVG(q.rgb_laplacian_var) AS mean_rgb_sharpness,
    AVG(q.missing_point_ratio) AS mean_missing_3d_ratio,
    AVG(q.z_mean) AS mean_z_depth,
    AVG(q.z_std) AS mean_z_std
FROM test_samples t
INNER JOIN category_totals ct
    ON t.category_name = ct.category_name
LEFT JOIN {{ ref('fct_data_quality') }} q
    ON t.sample_pk = q.sample_pk
GROUP BY
    t.category_name,
    t.defect_type,
    t.is_anomalous_benchmark,
    ct.category_test_total
ORDER BY
    t.category_name,
    sample_count DESC
