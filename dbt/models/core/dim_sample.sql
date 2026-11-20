-- Grain: 1 row per dataset sample
SELECT
    s.sample_pk,
    s.sample_key,
    s.dataset_version,
    s.category_name,
    c.material_type,
    s.split,
    s.defect_type,
    s.is_anomalous_benchmark,
    s.sample_id,
    s.rgb_relpath,
    s.xyz_relpath,
    s.gt_relpath,
    s.rgb_sha256,
    s.xyz_sha256,
    s.gt_sha256,
    s.rgb_size_bytes,
    s.xyz_size_bytes,
    s.gt_size_bytes,
    COALESCE(s.rgb_size_bytes, 0) + COALESCE(s.xyz_size_bytes, 0) + COALESCE(s.gt_size_bytes, 0) AS total_raw_bytes,
    s.bronze_archive_id,
    s.discovered_at
FROM {{ ref('stg_samples') }} s
LEFT JOIN {{ ref('stg_categories') }} c
    ON s.category_name = c.category_name
