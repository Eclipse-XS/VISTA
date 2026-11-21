-- Grain: 1 row per material_type x category_name comparing organic vs manufactured quality & segmentation
SELECT
    c.material_type,
    q.category_name,
    COUNT(*) AS sample_count,
    ROUND(AVG(q.rgb_brightness_mean), 2) AS global_brightness,
    ROUND(AVG(q.fg_rgb_brightness_mean), 2) AS fg_brightness,
    ROUND(AVG(q.bg_rgb_brightness_mean), 2) AS bg_brightness,
    ROUND(AVG(q.fg_rgb_laplacian_var), 2) AS fg_sharpness_lap,
    ROUND(100.0 * AVG(q.missing_point_ratio), 2) AS global_missing_3d_pct,
    ROUND(100.0 * AVG(q.fg_missing_3d_ratio), 2) AS fg_missing_3d_pct,
    ROUND(100.0 * AVG(q.bg_missing_3d_ratio), 2) AS bg_missing_3d_pct,
    ROUND(100.0 * AVG(q.fg_area_ratio), 2) AS mean_fg_area_pct,
    ROUND(100.0 * AVG(q.fg_3d_area_ratio), 2) AS mean_fg_3d_area_pct,
    ROUND(AVG(q.fg_rgb_3d_iou), 4) AS mean_fg_rgb_3d_iou,
    ROUND(MEDIAN(q.fg_rgb_3d_iou), 4) AS median_fg_rgb_3d_iou,
    ROUND(MIN(q.fg_rgb_3d_iou), 4) AS min_fg_rgb_3d_iou,
    SUM(CASE WHEN q.fg_rgb_3d_iou IS NULL OR q.fg_rgb_3d_iou < 0.50 THEN 1 ELSE 0 END) AS low_iou_failure_count,
    ROUND(
        100.0 * SUM(CASE WHEN q.fg_rgb_3d_iou IS NULL OR q.fg_rgb_3d_iou < 0.50 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS low_iou_failure_pct,
    ROUND(AVG(q.fg_plane_inlier_ratio), 4) AS mean_plane_inlier_ratio
FROM {{ ref('fct_data_quality') }} q
INNER JOIN {{ ref('dim_category') }} c
    ON q.category_name = c.category_name
GROUP BY c.material_type, q.category_name
ORDER BY c.material_type, q.category_name
