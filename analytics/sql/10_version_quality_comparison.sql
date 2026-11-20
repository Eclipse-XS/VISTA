-- Q10: How do quality distributions change between dataset or preprocessing versions?
SELECT
    s.dataset_version,
    q.preprocessing_version,
    q.config_hash,
    q.category_name,
    COUNT(*) AS sample_count,
    ROUND(AVG(q.rgb_brightness_mean), 2) AS mean_brightness,
    ROUND(AVG(q.rgb_laplacian_var), 2) AS mean_sharpness,
    ROUND(100.0 * AVG(q.valid_point_ratio), 2) AS mean_valid_3d_pct,
    ROUND(100.0 * AVG(q.missing_point_ratio), 2) AS mean_missing_3d_pct,
    ROUND(AVG(q.z_mean), 4) AS mean_z_m,
    SUM(CASE WHEN q.geometry_outlier_flag THEN 1 ELSE 0 END) AS geometry_outliers
FROM main.fct_data_quality q
INNER JOIN main.dim_sample s
    ON q.sample_pk = s.sample_pk
GROUP BY
    s.dataset_version,
    q.preprocessing_version,
    q.config_hash,
    q.category_name
ORDER BY
    s.dataset_version,
    q.category_name,
    q.preprocessing_version;
