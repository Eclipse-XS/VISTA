-- Grain: 1 row per MVTec 3D-AD category
SELECT
    c.category_id,
    c.category_name,
    c.material_type,
    c.source_url,
    c.archive_filename,
    c.approx_size_mb,
    COUNT(s.sample_pk) AS total_samples,
    SUM(CASE WHEN s.split = 'train' THEN 1 ELSE 0 END) AS train_samples,
    SUM(CASE WHEN s.split = 'validation' THEN 1 ELSE 0 END) AS validation_samples,
    SUM(CASE WHEN s.split = 'test' THEN 1 ELSE 0 END) AS test_samples,
    c.created_at
FROM {{ ref('stg_categories') }} c
LEFT JOIN {{ ref('stg_samples') }} s
    ON c.category_name = s.category_name
GROUP BY
    c.category_id,
    c.category_name,
    c.material_type,
    c.source_url,
    c.archive_filename,
    c.approx_size_mb,
    c.created_at
