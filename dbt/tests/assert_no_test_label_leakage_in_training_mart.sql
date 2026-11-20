-- Fails if any test-split sample leaks into the training-ready mart.
SELECT
    sample_pk,
    sample_key,
    split
FROM {{ ref('mart_training_ready_samples') }}
WHERE split NOT IN ('train', 'validation')
