-- Grain: 1 row per category x split x feature showing standardized shift relative to train split
WITH base AS (
    SELECT category_name, split, 'global_missing_3d_ratio' AS feature_name, '3d_global' AS modality_group, missing_point_ratio AS val FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'bg_missing_3d_ratio', '3d_background', bg_missing_3d_ratio FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_missing_3d_ratio', '3d_foreground', fg_missing_3d_ratio FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'global_rgb_brightness_mean', 'rgb_global', rgb_brightness_mean FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'bg_rgb_brightness_mean', 'rgb_background', bg_rgb_brightness_mean FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_rgb_brightness_mean', 'rgb_foreground', fg_rgb_brightness_mean FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_rgb_laplacian_var', 'rgb_foreground', fg_rgb_laplacian_var FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_area_ratio', 'framing', fg_area_ratio FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_center_offset_norm', 'framing', fg_center_offset_norm FROM {{ ref('fct_data_quality') }}
    UNION ALL
    SELECT category_name, split, 'fg_rgb_3d_iou', 'cross_modal', fg_rgb_3d_iou FROM {{ ref('fct_data_quality') }}
),
train_ref AS (
    SELECT
        category_name,
        feature_name,
        modality_group,
        AVG(val) AS train_mean,
        STDDEV_SAMP(val) AS train_std,
        MEDIAN(val) AS train_median
    FROM base
    WHERE split = 'train' AND val IS NOT NULL
    GROUP BY category_name, feature_name, modality_group
),
split_stats AS (
    SELECT
        category_name,
        split,
        feature_name,
        modality_group,
        COUNT(val) AS sample_count,
        AVG(val) AS split_mean,
        MEDIAN(val) AS split_median,
        STDDEV_SAMP(val) AS split_std
    FROM base
    WHERE val IS NOT NULL
    GROUP BY category_name, split, feature_name, modality_group
)
SELECT
    ss.category_name,
    ss.split,
    ss.feature_name,
    ss.modality_group,
    ss.sample_count,
    ROUND(tr.train_mean, 4) AS train_mean,
    ROUND(ss.split_mean, 4) AS split_mean,
    ROUND(tr.train_median, 4) AS train_median,
    ROUND(ss.split_median, 4) AS split_median,
    ROUND(
        CASE
            WHEN tr.train_std IS NOT NULL AND tr.train_std > 1e-12
            THEN (ss.split_mean - tr.train_mean) / tr.train_std
            ELSE NULL
        END,
        4
    ) AS shift_in_train_sd
FROM split_stats ss
INNER JOIN train_ref tr
    ON ss.category_name = tr.category_name
   AND ss.feature_name = tr.feature_name
ORDER BY ss.category_name, ss.split, ABS(shift_in_train_sd) DESC
