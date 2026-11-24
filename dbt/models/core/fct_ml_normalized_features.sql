-- Grain: 1 row per sample x frozen train-only normalization artifact
SELECT
    nf.norm_feature_id,
    nf.sample_pk,
    s.sample_key,
    nf.norm_artifact_id,
    na.artifact_key AS norm_artifact_key,
    nf.dataset_version,
    nf.category_name,
    nf.split,
    nf.preprocessing_version,
    nf.config_hash,
    nf.norm_version,
    nf.train_manifest_hash,
    na.train_sample_count,
    na.params_sha256,
    nf.norm_fg_rgb_brightness,
    nf.norm_fg_rgb_contrast_rms,
    nf.norm_fg_rgb_laplacian_var,
    nf.norm_fg_rgb_saturation,
    nf.norm_fg_missing_3d_ratio,
    nf.norm_fg_z_std,
    nf.norm_fg_area_ratio,
    nf.norm_fg_rgb_3d_iou,
    nf.norm_global_missing_3d_ratio,
    nf.norm_global_rgb_brightness,
    nf.train_ref_rgb_outlier_flag,
    nf.train_ref_geometry_outlier_flag,
    nf.computed_at
FROM {{ ref('stg_ml_normalized_features') }} nf
INNER JOIN {{ ref('stg_normalization_artifacts') }} na
    ON nf.norm_artifact_id = na.norm_artifact_id
INNER JOIN {{ ref('stg_samples') }} s
    ON nf.sample_pk = s.sample_pk
