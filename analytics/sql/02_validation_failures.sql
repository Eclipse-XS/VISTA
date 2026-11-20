-- Q2: What proportion of samples pass all validation rules, and what issues occur?
WITH per_category AS (
    SELECT
        category_name,
        COUNT(*) AS total_samples,
        SUM(CASE WHEN status = 'PASS' THEN 1 ELSE 0 END) AS pass_count,
        SUM(CASE WHEN status = 'WARN' THEN 1 ELSE 0 END) AS warn_count,
        SUM(CASE WHEN status = 'QUARANTINE' THEN 1 ELSE 0 END) AS quarantine_count,
        SUM(CASE WHEN status = 'FAIL' THEN 1 ELSE 0 END) AS fail_count,
        SUM(CASE WHEN NOT spatial_aligned THEN 1 ELSE 0 END) AS misaligned_count,
        SUM(CASE WHEN is_exact_duplicate THEN 1 ELSE 0 END) AS exact_duplicate_count,
        SUM(CASE WHEN is_near_duplicate THEN 1 ELSE 0 END) AS near_duplicate_count,
        SUM(CASE WHEN is_cross_split_leak THEN 1 ELSE 0 END) AS cross_split_leak_count,
        SUM(CASE WHEN nan_3d_count > 0 OR inf_3d_count > 0 THEN 1 ELSE 0 END) AS non_finite_3d_samples
    FROM main.fct_validation_results
    GROUP BY category_name
)
SELECT
    category_name,
    total_samples,
    pass_count,
    ROUND(100.0 * pass_count / NULLIF(total_samples, 0), 2) AS pass_rate_pct,
    warn_count,
    quarantine_count,
    fail_count,
    misaligned_count,
    exact_duplicate_count,
    near_duplicate_count,
    cross_split_leak_count,
    non_finite_3d_samples
FROM per_category
ORDER BY pass_rate_pct ASC, category_name;
