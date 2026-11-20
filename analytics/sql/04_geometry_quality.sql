-- Q3: Which categories have the highest missing 3D-point ratios and how do depth stats compare?
SELECT
    category_name,
    material_type,
    preprocessing_version,
    sample_count,
    ROUND(100.0 * mean_valid_3d_ratio, 2) AS mean_valid_3d_pct,
    ROUND(100.0 * mean_missing_3d_ratio, 2) AS mean_missing_3d_pct,
    ROUND(100.0 * median_missing_3d_ratio, 2) AS median_missing_3d_pct,
    ROUND(100.0 * max_missing_3d_ratio, 2) AS max_missing_3d_pct,
    ROUND(mean_z_depth, 4) AS mean_z_meters,
    ROUND(mean_z_std, 4) AS mean_z_std_meters,
    geometry_outlier_count
FROM main.mart_category_quality
ORDER BY mean_missing_3d_ratio DESC, category_name;
