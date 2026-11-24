-- Q16: Cross-category comparison of organic vs manufactured RGB/3D quality & foreground segmentation reliability
SELECT
    material_type,
    category_name,
    sample_count,
    global_brightness,
    fg_brightness,
    bg_brightness,
    fg_sharpness_lap,
    global_missing_3d_pct,
    fg_missing_3d_pct,
    bg_missing_3d_pct,
    mean_fg_area_pct,
    mean_fg_3d_area_pct,
    mean_fg_rgb_3d_iou,
    median_fg_rgb_3d_iou,
    min_fg_rgb_3d_iou,
    low_iou_failure_count,
    low_iou_failure_pct,
    mean_plane_inlier_ratio
FROM main.mart_material_type_comparison
ORDER BY material_type, category_name;
