-- Q4: How do RGB brightness, sharpness, and contrast vary between categories?
SELECT
    q.category_name,
    c.material_type,
    q.split,
    COUNT(*) AS sample_count,
    ROUND(AVG(q.rgb_brightness_mean), 2) AS mean_brightness,
    ROUND(MEDIAN(q.rgb_brightness_mean), 2) AS median_brightness,
    ROUND(STDDEV_SAMP(q.rgb_brightness_mean), 2) AS std_brightness,
    ROUND(AVG(q.rgb_contrast_rms), 4) AS mean_contrast_rms,
    ROUND(AVG(q.rgb_contrast_p5_p95), 2) AS mean_contrast_p5_p95,
    ROUND(AVG(q.rgb_laplacian_var), 2) AS mean_laplacian_var,
    ROUND(MEDIAN(q.rgb_laplacian_var), 2) AS median_laplacian_var,
    ROUND(AVG(q.rgb_saturation_mean), 4) AS mean_saturation,
    ROUND(100.0 * AVG(q.rgb_underexposed_ratio), 2) AS mean_underexposed_pct,
    ROUND(100.0 * AVG(q.rgb_overexposed_ratio), 2) AS mean_overexposed_pct
FROM main.fct_data_quality q
LEFT JOIN main.dim_category c
    ON q.category_name = c.category_name
GROUP BY q.category_name, c.material_type, q.split
ORDER BY q.category_name, q.split;
