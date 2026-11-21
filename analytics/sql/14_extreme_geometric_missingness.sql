-- P1 Q4: Which samples exhibit extreme global vs foreground geometric missingness?
SELECT
    q.category_name,
    q.split,
    s.defect_type,
    q.sample_key,
    ROUND(100.0 * q.missing_point_ratio, 2) AS global_missing_3d_pct,
    ROUND(100.0 * q.fg_missing_3d_ratio, 2) AS fg_missing_3d_pct,
    ROUND(100.0 * q.bg_missing_3d_ratio, 2) AS bg_missing_3d_pct,
    ROUND(100.0 * q.fg_area_ratio, 2) AS fg_area_pct,
    ROUND(q.fg_rgb_3d_iou, 4) AS fg_rgb_3d_iou,
    nf.train_ref_geometry_outlier_flag
FROM main.fct_data_quality q
INNER JOIN main.dim_sample s
    ON q.sample_pk = s.sample_pk
LEFT JOIN main.fct_ml_normalized_features nf
    ON q.sample_pk = nf.sample_pk
ORDER BY q.fg_missing_3d_ratio DESC NULLS LAST, q.missing_point_ratio DESC
LIMIT 20;
