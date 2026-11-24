-- Grain: 1 row per validated, non-leaking train/validation sample ready for ML training.
-- STRICT LEAKAGE GUARD:
--   1. Excludes `test` split entirely.
--   2. Omits `defect_type`, `is_anomalous_benchmark`, and within-split descriptive EDA z-scores.
--   3. Joins ONLY frozen train-only normalized features (`fct_ml_normalized_features`).
WITH latest_norm_per_cat AS (
    SELECT
        category_name,
        preprocessing_version,
        config_hash,
        MAX(norm_artifact_id) AS norm_artifact_id
    FROM {{ ref('stg_normalization_artifacts') }}
    GROUP BY category_name, preprocessing_version, config_hash
)
SELECT
    s.sample_pk,
    s.sample_key,
    s.dataset_version,
    s.category_name,
    s.split AS partition,
    s.split,
    s.sample_id,
    v.status AS validation_status,
    sa.preprocessing_version,
    sa.config_hash,
    sa.depth_remote_uri AS silver_remote_uri,
    sa.depth_sha256 AS silver_sha256,
    sa.normal_reliable,
    nf.norm_artifact_id,
    nf.train_manifest_hash,
    q.rgb_brightness_mean,
    q.rgb_laplacian_var,
    q.valid_point_ratio,
    q.missing_point_ratio,
    q.fg_area_ratio,
    q.fg_rgb_3d_iou,
    q.fg_rgb_brightness_mean,
    q.fg_rgb_laplacian_var,
    q.fg_missing_3d_ratio,
    nf.norm_fg_rgb_brightness,
    nf.norm_fg_rgb_contrast_rms,
    nf.norm_fg_rgb_laplacian_var,
    nf.norm_fg_rgb_saturation,
    nf.norm_fg_missing_3d_ratio,
    nf.norm_fg_z_std,
    nf.norm_fg_area_ratio,
    nf.norm_fg_rgb_3d_iou,
    nf.train_ref_rgb_outlier_flag,
    nf.train_ref_geometry_outlier_flag
FROM {{ ref('dim_sample') }} s
INNER JOIN {{ ref('fct_validation_results') }} v
    ON s.sample_pk = v.sample_pk
INNER JOIN {{ ref('stg_silver_artifacts') }} sa
    ON s.sample_pk = sa.sample_pk
LEFT JOIN {{ ref('fct_data_quality') }} q
    ON s.sample_pk = q.sample_pk
   AND q.preprocessing_version = sa.preprocessing_version
   AND q.config_hash = sa.config_hash
LEFT JOIN latest_norm_per_cat ln
    ON s.category_name = ln.category_name
   AND sa.preprocessing_version = ln.preprocessing_version
   AND sa.config_hash = ln.config_hash
LEFT JOIN {{ ref('fct_ml_normalized_features') }} nf
    ON s.sample_pk = nf.sample_pk
   AND nf.norm_artifact_id = ln.norm_artifact_id
WHERE s.split IN ('train', 'validation')
  AND v.status IN ('PASS', 'WARN')
  AND v.is_cross_split_leak = FALSE
  AND sa.artifact_status = 'SUCCEEDED'
