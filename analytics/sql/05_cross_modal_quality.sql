-- Q5: Are RGB quality indicators associated with geometric missingness?
SELECT
    category_name,
    COUNT(*) AS sample_count,
    ROUND(CORR(rgb_brightness_mean, missing_point_ratio), 4) AS corr_brightness_vs_missing3d,
    ROUND(CORR(rgb_contrast_rms, missing_point_ratio), 4) AS corr_contrast_vs_missing3d,
    ROUND(CORR(rgb_laplacian_var, missing_point_ratio), 4) AS corr_sharpness_vs_missing3d,
    ROUND(CORR(rgb_saturation_mean, missing_point_ratio), 4) AS corr_saturation_vs_missing3d,
    ROUND(CORR(rgb_underexposed_ratio, missing_point_ratio), 4) AS corr_underexp_vs_missing3d,
    ROUND(CORR(rgb_overexposed_ratio, missing_point_ratio), 4) AS corr_overexp_vs_missing3d
FROM main.fct_data_quality
GROUP BY category_name
ORDER BY category_name;
