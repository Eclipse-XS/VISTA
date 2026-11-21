-- P1 Q6 & Q7: Assess quality indicators as adaptive-fusion candidates vs spurious shortcut risks.
WITH test_shifts AS (
    SELECT
        category_name,
        feature_name,
        modality_group,
        shift_in_train_sd AS test_shift_sd
    FROM main.mart_split_distribution_shift
    WHERE split = 'test'
),
val_shifts AS (
    SELECT
        category_name,
        feature_name,
        shift_in_train_sd AS val_shift_sd
    FROM main.mart_split_distribution_shift
    WHERE split = 'validation'
)
SELECT
    t.category_name,
    t.feature_name,
    t.modality_group,
    v.val_shift_sd,
    t.test_shift_sd,
    CASE
        WHEN t.modality_group IN ('3d_global', '3d_background', 'rgb_global', 'rgb_background')
         AND ABS(t.test_shift_sd) >= 1.5
            THEN 'HIGH_SHORTCUT_RISK (background/session shift)'
        WHEN t.modality_group = 'framing'
            THEN 'CONFOUNDER_CONTROL (monitor framing/scale)'
        WHEN ABS(t.test_shift_sd) < 1.0
            THEN 'PLAUSIBLE_FUSION_CANDIDATE (low split shift)'
        ELSE 'MODERATE_SHIFT_INVESTIGATE'
    END AS engineering_assessment
FROM test_shifts t
LEFT JOIN val_shifts v
    ON t.category_name = v.category_name
   AND t.feature_name = v.feature_name
ORDER BY t.category_name, ABS(t.test_shift_sd) DESC;
