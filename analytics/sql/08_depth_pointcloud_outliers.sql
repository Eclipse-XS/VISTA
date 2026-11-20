-- Q6: Which samples exhibit unusual depth or point-cloud statistics?
SELECT
    q.category_name,
    q.split,
    s.defect_type,
    q.sample_key,
    ROUND(q.missing_point_ratio, 4) AS missing_3d_ratio,
    ROUND(q.missing_ratio_robust_z, 3) AS missing_robust_z,
    ROUND(q.z_mean, 4) AS z_mean_m,
    ROUND(q.z_mean_robust_z, 3) AS z_mean_robust_z,
    ROUND(q.z_std, 4) AS z_std_m,
    ROUND(q.bbox_volume, 6) AS bbox_vol_m3,
    q.geometry_outlier_flag,
    q.rgb_quality_outlier_flag
FROM main.fct_data_quality q
INNER JOIN main.dim_sample s
    ON q.sample_pk = s.sample_pk
WHERE q.geometry_outlier_flag = TRUE
   OR q.rgb_quality_outlier_flag = TRUE
ORDER BY GREATEST(ABS(COALESCE(q.z_mean_robust_z, 0)), ABS(COALESCE(q.missing_ratio_robust_z, 0)), ABS(COALESCE(q.sharpness_robust_z, 0))) DESC
LIMIT 30;
