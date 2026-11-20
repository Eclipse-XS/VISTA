-- Q7: What are the defect-type distributions in labeled benchmark test data?
-- NOTE: Reflects MVTec 3D-AD benchmark construction, NOT real factory defect rates.
SELECT
    category_name,
    defect_type,
    is_anomalous_benchmark,
    sample_count,
    category_test_total,
    benchmark_test_share_pct
FROM main.mart_defect_distribution
ORDER BY category_name, sample_count DESC, defect_type;
