-- P1 Q1: How much do foreground-aware quality metrics differ from global metrics?
SELECT
    category_name,
    material_type,
    split,
    sample_count,
    global_rgb_brightness,
    fg_rgb_brightness,
    bg_rgb_brightness,
    global_sharpness_lap,
    fg_sharpness_lap,
    global_missing_3d_pct,
    fg_missing_3d_pct,
    bg_missing_3d_pct,
    ROUND(global_missing_3d_pct - fg_missing_3d_pct, 2) AS global_minus_fg_missing_pp,
    fg_area_pct,
    mean_fg_rgb_3d_iou,
    mean_fg_center_offset
FROM main.mart_foreground_vs_global_quality
ORDER BY category_name, split;
