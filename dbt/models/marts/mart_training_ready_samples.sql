-- Grain: 1 row per validated, non-leaking train/validation sample ready for future ML training.
-- STRICT LEAKAGE GUARD: Excludes `test` split entirely and omits benchmark anomaly label columns.
SELECT
    s.sample_pk,
    s.sample_key,
    s.dataset_version,
    s.category_name,
    s.split,
    s.sample_id,
    v.status AS validation_status,
    sa.preprocessing_version,
    sa.config_hash,
    sa.depth_remote_uri AS silver_remote_uri,
    sa.depth_sha256 AS silver_sha256,
    sa.normal_reliable,
    q.rgb_brightness_mean,
    q.rgb_laplacian_var,
    q.valid_point_ratio,
    q.missing_point_ratio
FROM {{ ref('dim_sample') }} s
INNER JOIN {{ ref('fct_validation_results') }} v
    ON s.sample_pk = v.sample_pk
INNER JOIN {{ ref('stg_silver_artifacts') }} sa
    ON s.sample_pk = sa.sample_pk
LEFT JOIN {{ ref('fct_data_quality') }} q
    ON s.sample_pk = q.sample_pk
   AND q.preprocessing_version = sa.preprocessing_version
   AND q.config_hash = sa.config_hash
WHERE s.split IN ('train', 'validation')
  AND v.status IN ('PASS', 'WARN')
  AND v.is_cross_split_leak = FALSE
  AND sa.artifact_status = 'SUCCEEDED'
