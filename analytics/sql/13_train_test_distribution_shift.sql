-- P1 Q3 & Q5: Which modality exhibits stronger train-to-test shifts and framing/footprint effects?
SELECT
    category_name,
    split,
    feature_name,
    modality_group,
    sample_count,
    train_mean,
    split_mean,
    train_median,
    split_median,
    shift_in_train_sd
FROM main.mart_split_distribution_shift
WHERE split IN ('validation', 'test')
ORDER BY category_name, split, ABS(shift_in_train_sd) DESC;
