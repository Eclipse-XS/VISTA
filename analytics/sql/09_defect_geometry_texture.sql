-- Q8: How do different defect types relate to measured geometry and texture characteristics?
-- Evaluated descriptively on the labeled benchmark test split only.
SELECT
    category_name,
    defect_type,
    is_anomalous_benchmark,
    sample_count,
    ROUND(mean_rgb_brightness, 2) AS mean_brightness,
    ROUND(mean_rgb_sharpness, 2) AS mean_laplacian_var,
    ROUND(100.0 * mean_missing_3d_ratio, 2) AS mean_missing_3d_pct,
    ROUND(mean_z_depth, 4) AS mean_z_m,
    ROUND(mean_z_std, 4) AS mean_z_std_m
FROM main.mart_defect_distribution
ORDER BY category_name, is_anomalous_benchmark, defect_type;
