-- Fails if foreground valid + missing 3D ratios deviate from 1.0 beyond floating-point tolerance.
SELECT
    metric_id,
    sample_pk,
    fg_valid_3d_ratio,
    fg_missing_3d_ratio
FROM {{ ref('fct_data_quality') }}
WHERE fg_valid_3d_ratio IS NOT NULL
  AND fg_missing_3d_ratio IS NOT NULL
  AND ABS((fg_valid_3d_ratio + fg_missing_3d_ratio) - 1.0) > 1e-5
