-- Fails if valid_point_ratio + missing_point_ratio deviates from 1.0 beyond floating-point tolerance.
SELECT
    metric_id,
    sample_pk,
    valid_point_ratio,
    missing_point_ratio
FROM {{ ref('fct_data_quality') }}
WHERE valid_point_ratio IS NOT NULL
  AND missing_point_ratio IS NOT NULL
  AND ABS((valid_point_ratio + missing_point_ratio) - 1.0) > 1e-5
