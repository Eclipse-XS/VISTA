-- Grain: 1 row per category x preprocessing_version x config_hash
SELECT
    q.category_name,
    c.material_type,
    q.preprocessing_version,
    q.config_hash,
    COUNT(*) AS sample_count,
    AVG(q.rgb_brightness_mean) AS mean_rgb_brightness,
    MEDIAN(q.rgb_brightness_mean) AS median_rgb_brightness,
    AVG(q.rgb_contrast_rms) AS mean_rgb_contrast_rms,
    AVG(q.rgb_laplacian_var) AS mean_rgb_sharpness_laplacian,
    MEDIAN(q.rgb_laplacian_var) AS median_rgb_sharpness_laplacian,
    AVG(q.rgb_saturation_mean) AS mean_rgb_saturation,
    AVG(q.rgb_underexposed_ratio) AS mean_underexposed_ratio,
    AVG(q.rgb_overexposed_ratio) AS mean_overexposed_ratio,
    AVG(q.valid_point_ratio) AS mean_valid_3d_ratio,
    AVG(q.missing_point_ratio) AS mean_missing_3d_ratio,
    MEDIAN(q.missing_point_ratio) AS median_missing_3d_ratio,
    MAX(q.missing_point_ratio) AS max_missing_3d_ratio,
    AVG(q.depth_coverage_ratio) AS mean_depth_coverage_ratio,
    AVG(q.z_mean) AS mean_z_depth,
    AVG(q.z_std) AS mean_z_std,
    SUM(CASE WHEN q.rgb_quality_outlier_flag THEN 1 ELSE 0 END) AS rgb_outlier_count,
    SUM(CASE WHEN q.geometry_outlier_flag THEN 1 ELSE 0 END) AS geometry_outlier_count
FROM {{ ref('fct_data_quality') }} q
LEFT JOIN {{ ref('dim_category') }} c
    ON q.category_name = c.category_name
GROUP BY
    q.category_name,
    c.material_type,
    q.preprocessing_version,
    q.config_hash
