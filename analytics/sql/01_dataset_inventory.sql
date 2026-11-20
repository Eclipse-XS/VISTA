-- Q1: How many samples exist per category and dataset split?
SELECT
    category_name,
    material_type,
    split,
    sample_count,
    passed_count,
    warned_count,
    quarantined_count,
    failed_count,
    pass_pct,
    raw_size_mb,
    silver_ready_count,
    reliable_normal_count
FROM main.mart_dataset_inventory
ORDER BY category_name, split;
