-- P1 Q2: Does RGB-3D quality correlation persist after isolating foreground from background?
SELECT
    category_name,
    COUNT(*) AS sample_count,
    ROUND(CORR(rgb_brightness_mean, missing_point_ratio), 4) AS global_corr_bright_vs_miss3d,
    ROUND(CORR(fg_rgb_brightness_mean, fg_missing_3d_ratio), 4) AS fg_corr_bright_vs_miss3d,
    ROUND(CORR(rgb_laplacian_var, missing_point_ratio), 4) AS global_corr_sharp_vs_miss3d,
    ROUND(CORR(fg_rgb_laplacian_var, fg_missing_3d_ratio), 4) AS fg_corr_sharp_vs_miss3d,
    ROUND(CORR(bg_rgb_brightness_mean, bg_missing_3d_ratio), 4) AS bg_corr_bright_vs_miss3d,
    ROUND(CORR(fg_area_ratio, missing_point_ratio), 4) AS corr_footprint_vs_global_miss3d,
    ROUND(CORR(fg_area_ratio, fg_missing_3d_ratio), 4) AS corr_footprint_vs_fg_miss3d
FROM main.fct_data_quality
GROUP BY category_name
ORDER BY category_name;
