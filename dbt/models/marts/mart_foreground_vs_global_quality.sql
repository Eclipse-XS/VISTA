-- Grain: 1 row per category x split comparing global, foreground-aware, and background metrics
SELECT
    q.category_name,
    c.material_type,
    q.split,
    COUNT(*) AS sample_count,
    ROUND(AVG(q.rgb_brightness_mean), 2) AS global_rgb_brightness,
    ROUND(AVG(q.fg_rgb_brightness_mean), 2) AS fg_rgb_brightness,
    ROUND(AVG(q.bg_rgb_brightness_mean), 2) AS bg_rgb_brightness,
    ROUND(AVG(q.rgb_laplacian_var), 2) AS global_sharpness_lap,
    ROUND(AVG(q.fg_rgb_laplacian_var), 2) AS fg_sharpness_lap,
    ROUND(100.0 * AVG(q.missing_point_ratio), 2) AS global_missing_3d_pct,
    ROUND(100.0 * AVG(q.fg_missing_3d_ratio), 2) AS fg_missing_3d_pct,
    ROUND(100.0 * AVG(q.bg_missing_3d_ratio), 2) AS bg_missing_3d_pct,
    ROUND(100.0 * AVG(q.fg_area_ratio), 2) AS fg_area_pct,
    ROUND(AVG(q.fg_rgb_3d_iou), 4) AS mean_fg_rgb_3d_iou,
    ROUND(AVG(q.fg_center_offset_norm), 4) AS mean_fg_center_offset,
    ROUND(AVG(q.fg_plane_inlier_ratio), 4) AS mean_plane_inlier_ratio
FROM {{ ref('fct_data_quality') }} q
LEFT JOIN {{ ref('dim_category') }} c
    ON q.category_name = c.category_name
GROUP BY q.category_name, c.material_type, q.split
ORDER BY q.category_name, q.split
